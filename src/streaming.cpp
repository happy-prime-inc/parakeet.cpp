#include "streaming.hpp"
#include "tokenizer.hpp"
#include "mel.hpp"
#include <algorithm>
#include <cassert>

namespace pk {

StreamingSession::StreamingSession(const ModelLoader& ml, const std::string& target_lang)
    : ml_(ml), enc_(ml), pred_(ml), joint_(ml), prompt_(ml), acc_(ml) {
    const ParakeetConfig& cfg = ml.config();
    d_model_  = (int)cfg.d_model;
    blank_id_ = (int)cfg.blank_id;

    // Resolve the language prompt index for multilingual (nemotron) models. The
    // one-hot is constant over time (one language per utterance), so the prompt
    // projection is applied per chunk in feed_mel_chunk using this fixed index.
    // Non-prompt models leave prompt_index_ = -1 and skip prompt_.apply().
    if (cfg.prompt.present) {
        // Empty target_lang -> the model default; an unknown locale THROWS
        // std::runtime_error (same message as Model::resolve_prompt_index), so a
        // typo (e.g. --lang xx) fails loudly instead of silently mis-transcribing.
        // Matches the offline path and the parakeet_capi_stream_begin_lang
        // contract (NULL + ctx last_error on an unknown locale).
        prompt_index_ = cfg.prompt.resolve_index_or_throw(target_lang);
    }
    // Greedy max symbols per frame, from model metadata (NeMo default 10);
    // matches the offline pk::transcribe path in model.cpp.
    max_symbols_ = (int)cfg.max_symbols;
    assert(joint_.num_durations() == 0 && "StreamingSession is RNN-T only (no TDT durations)");

    // <EOU>/<EOB> ids and the per-encoder-frame stride are resolved by
    // TranscriptAccumulator from the same model, for both streaming paths.

    reset();
}

void StreamingSession::reset() {
    enc_.reset();
    state_ = rnnt_decode_init(pred_);
    acc_.reset();
}

std::vector<int32_t> StreamingSession::feed_mel_chunk(const std::vector<float>& mel_chunk,
                                                      int n_frames, bool is_last) {
    // 1. Encoder step: the chunk's valid encoder frames, row-major [valid, d_model]
    //    (d_model fastest) — exactly the orientation rnnt_decode_frames expects.
    int n_valid = 0;
    std::vector<float> enc_frames = enc_.step(mel_chunk, n_frames, is_last, n_valid);

    if (n_valid <= 0) {
        acc_.absorb({}, {}, nullptr, acc_.encoder_frame());  // clears the EOU flag
        return {};
    }

    // 1b. Prompt conditioning (nemotron multilingual): project the chunk's
    //     encoder frames through prompt_kernel for the resolved language before
    //     the RNN-T decode. The one-hot is constant over time, so applying it
    //     per chunk is exact (== the offline forward's single application).
    //     prompt_.apply() wants channels-first [d_model, valid]; enc_.step gives
    //     time-major [valid, d_model], so transpose in, apply, transpose back.
    //     No-op for non-prompt models (prompt_.present()==false): enc_frames is
    //     left byte-identical.
    if (prompt_.present()) {
        std::vector<float> chunk_cf((size_t)d_model_ * n_valid);  // [d_model, valid]
        for (int t = 0; t < n_valid; ++t)
            for (int c = 0; c < d_model_; ++c)
                chunk_cf[(size_t)c * n_valid + t] = enc_frames[(size_t)t * d_model_ + c];
        std::vector<float> projected;
        prompt_.apply(chunk_cf, d_model_, n_valid, prompt_index_, projected);  // [d_model, valid]
        for (int t = 0; t < n_valid; ++t)
            for (int c = 0; c < d_model_; ++c)
                enc_frames[(size_t)t * d_model_ + c] = projected[(size_t)c * n_valid + t];
    }

    // 2. RNN-T greedy over the new encoder frames, carrying the decoder state
    //    across chunks (do NOT reset). Appends to state_.hyp and returns the ids
    //    emitted in this chunk, with their LOCAL frame index in [0, n_valid) and
    //    per-token TokenInfo (LOCAL frame, max_prob conf, span==1).
    const int base_frame = acc_.encoder_frame();
    std::vector<int32_t> local_frames;
    std::vector<TokenInfo> chunk_tokens;
    std::vector<int32_t> emitted =
        rnnt_decode_frames(pred_, joint_, enc_frames, n_valid, d_model_,
                           state_, blank_id_, max_symbols_, &local_frames,
                           &chunk_tokens);
    acc_.advance_frames(n_valid);

    // 3. Update text, EOU events (absolute frames from the decoder's per-token
    //    local frame + base) and per-word timestamps. rnnt_decode_frames reports
    //    LOCAL frames, so local_frames is passed and base_frame added.
    acc_.absorb(emitted, chunk_tokens, &local_frames, base_frame);
    acc_.regroup_words(/*flush_all=*/false);

    // 4. End-of-utterance reset. The realtime EOU model is trained to emit <EOU>
    //    (end of utterance) / <EOB> (backchannel) and have the decoder START THE
    //    NEXT UTTERANCE FROM A FRESH STATE — exactly what NeMo's reference
    //    streaming driver does (examples/voice_agent .../nemo/streaming_asr.py
    //    NemoStreamingASRService.transcribe -> reset_state() when <EOU>/<EOB>
    //    appears in the chunk text). Without this the prediction net stays
    //    conditioned on the just-emitted <EOU> and the joint scores blank on every
    //    subsequent frame, so the stream goes silent after the first utterance
    //    (issue #13). We reset the carried RNN-T decoder state — LSTM h/c to zero
    //    and last_token back to SOS — for the next chunk. We deliberately reset
    //    ONLY the decoder, not the StreamingEncoder cache: NeMo's reset_state also
    //    drops the encoder cache, but that was verified to be a no-op for the
    //    decoded tokens (decoder-only reset == NeMo's full reset_state byte-for-
    //    byte on multi-utterance clips), so the validated streaming-encoder path is
    //    left untouched. enc_frame_ keeps running so <EOU> timestamps stay absolute
    //    in the clip, and state_.hyp keeps the full token record across utterances.
    if (acc_.last_step_had_eou()) {
        state_.state      = pred_.zero_state();
        state_.last_token = -1;     // SOS sentinel (nothing emitted yet)
        state_.have_token = false;
    }
    return emitted;
}

std::string StreamingSession::finalize() {
    // The end-of-stream tail is flushed by the caller feeding the final buffered
    // mel chunk with is_last=true (which keeps the streaming tail frames). At the
    // session level there is no further audio to process here, so finalize just
    // returns whatever newly-finalized text remains since the last take. NeMo's
    // cache-aware streaming behaves identically: the final chunk's incomplete
    // right context means a trailing <EOU> is NOT recovered, so we never
    // fabricate one — finalize emits only the carried-state tokens already
    // decoded.
    //
    // Word side: the end-of-stream has no further word-start markers, so the
    // trailing open word is now final too — regroup with flush_all so it becomes
    // available to drain_words().
    acc_.regroup_words(/*flush_all=*/true);
    return acc_.take_new_text();
}





std::vector<Word> StreamingSession::drain_words() { return acc_.drain_words(); }

std::string StreamingSession::take_new_text() { return acc_.take_new_text(); }

std::vector<EouEvent> StreamingSession::drain_events() { return acc_.drain_events(); }

void run_stream_over_pcm(
    StreamingSession& sess, const ModelLoader& ml,
    const std::vector<float>& pcm16k,
    const std::function<void(const std::string&,
                             const std::vector<EouEvent>&,
                             const std::vector<Word>&)>& on_chunk,
    const std::string& target_lang) {
    // target_lang is intentionally unused here: the session already carries its
    // resolved prompt index from construction. The parameter exists so callers
    // can route a language through a single entry point (Phase 4 C-API/CLI).
    (void)target_lang;
    // 1. Full-clip mel [n_mels, T] (feat-major inner=T), matching the offline /
    //    NeMo online_normalization=False reference (normalization over the whole
    //    clip). The streaming numerics come from the carried encoder/decoder
    //    caches, not from chunking the front end.
    MelFrontend mel_fe(ml);
    std::vector<float> mel;
    int n_mels = 0, T = 0;
    mel_fe.compute(pcm16k, mel, n_mels, T);
    if (T <= 0) return;

    const int chunk0     = sess.chunk_size_first();      // 9
    const int chunk_main = sess.chunk_size();            // 16
    const int pre_cache  = sess.pre_encode_cache_size(); // 9

    // mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        const int chunk_hi   = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        const int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        sess.feed_mel_chunk(win, win_frames, is_last);

        if (on_chunk) {
            std::string nt = sess.take_new_text();
            std::vector<EouEvent> ev = sess.drain_events();
            std::vector<Word> wd = sess.drain_words();
            if (!nt.empty() || !ev.empty() || !wd.empty()) on_chunk(nt, ev, wd);
        }

        buffer_idx += shift;
        first = false;
    }
}

} // namespace pk
