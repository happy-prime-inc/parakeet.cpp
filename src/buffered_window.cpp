#include "buffered_window.hpp"
#include "tokenizer.hpp"
#include <algorithm>

namespace pk {

bool BufferedTdtStream::supports(const ModelLoader& ml) {
    // TDT durations are what this decodes with; a model without them is either
    // CTC or plain RNN-T and belongs on a different path.
    return !ml.config().tdt_durations.empty();
}

BufferedTdtStream::BufferedTdtStream(const ModelLoader& ml,
                                     double left_secs, double chunk_secs, double right_secs)
    : ml_(ml), frontend_(ml), sub_(ml), enc_(ml), pred_(ml), joint_(ml),
      state_(tdt_decode_init(pred_)), acc_(ml) {
    const ParakeetConfig& cfg = ml.config();
    hop_         = (int)(cfg.hop_length ? cfg.hop_length : 160);
    blank_id_    = (int)cfg.blank_id;
    max_symbols_ = (int)cfg.max_symbols;
    durations_   = cfg.tdt_durations;

    // Derive the schedule from the model's own framing rather than assuming
    // 100 mel frames/s, and round each span to a whole ENCODER frame — not a
    // whole mel frame. The encoder subsamples the mel by subsampling_factor,
    // so a chunk that is not a multiple of it (1 s = 100 mel = 12.5 encoder
    // frames at x8) puts every chunk boundary between encoder frames; the
    // trim then rounds each window into overlapping or skipping its
    // neighbours while the carried decoder state assumes seamless
    // continuation. Measured on a 200 s reading before this rounding: a
    // 10/1/1 schedule deleted 162 of 369 reference words. NeMo's buffered
    // script does the same correction ("Corrected contexts (subsampled
    // encoder frames)" in its log) for the same reason.
    const double sr = (double)(cfg.sample_rate ? cfg.sample_rate : 16000);
    sub_factor_ = (int)(cfg.subsampling_factor ? cfg.subsampling_factor : 8);
    const int64_t enc_frame_samples = (int64_t)hop_ * sub_factor_;
    // Truncation, not round-half-up, because that is NeMo's convention
    // (int(secs * features_per_sec / subsampling_factor)) and parity means
    // matching its corrected schedule exactly: 1 s -> 12 encoder frames
    // (0.96 s), not 13. At least one frame per span so a small chunk cannot
    // round to zero and stall the drain loop.
    auto to_samples = [&](double secs) -> int64_t {
        int64_t frames = (int64_t)(secs * sr / (double)enc_frame_samples);
        if (secs > 0 && frames < 1) frames = 1;
        return frames * enc_frame_samples;
    };
    left_s_  = to_samples(left_secs);
    chunk_s_ = to_samples(chunk_secs);
    right_s_ = to_samples(right_secs);

    reset();
}

void BufferedTdtStream::reset() {
    state_ = tdt_decode_init(pred_);
    acc_.reset();
    audio_.clear();
    audio_origin_ = 0;
    total_seen_   = 0;
    next_chunk_   = 0;
    finished_     = false;
    tentative_.clear();
}

double BufferedTdtStream::latency_sec() const {
    const double sr = (double)(ml_.config().sample_rate ? ml_.config().sample_rate : 16000);
    return (double)(chunk_s_ + right_s_) / sr;
}

std::vector<int32_t> BufferedTdtStream::feed_pcm(const float* pcm, int n_samples, bool is_last) {
    std::vector<int32_t> emitted;
    if (n_samples < 0 || (!pcm && n_samples > 0)) return emitted;
    if (n_samples > 0) {
        audio_.insert(audio_.end(), pcm, pcm + n_samples);
        total_seen_ += n_samples;
    }
    drain_ready(is_last, emitted);
    // Before this turn's first commit, preview the opening directly. Nothing
    // can commit until a whole chunk AND its right context have arrived, so
    // without this the first words of a sentence wait ~chunk+right while every
    // later word waits ~1s — the asymmetry a reader notices at the start of
    // every sentence, and the difference between "it is working" and "is it
    // hearing me?".
    //
    // This is the one preview that needs its own encoder pass, and it is
    // affordable precisely because it only runs here: before the first commit
    // there is no left context yet, so the window is at most chunk+right.
    // Re-run at most every onset_min_secs_ of new audio, so a 100ms feed
    // cadence does not mean an encode per feed.
    if (speculate_ && !is_last && next_chunk_ == 0 && onset_min_secs_ > 0) {
        const double sr = (double)(ml_.config().sample_rate
                                   ? ml_.config().sample_rate : 16000);
        const int64_t step = (int64_t)(onset_min_secs_ * sr);
        if (total_seen_ >= step && total_seen_ - onset_last_ >= step) {
            onset_last_ = total_seen_;
            preview_span(0, total_seen_);
        }
    }
    if (is_last) finished_ = true;
    return emitted;
}

std::string BufferedTdtStream::finalize() {
    if (!finished_) {
        std::vector<int32_t> emitted;
        drain_ready(/*flush=*/true, emitted);
        finished_ = true;
    }
    // No further word-start markers can arrive, so the trailing open word is
    // final too.
    acc_.regroup_words(/*flush_all=*/true);
    return acc_.take_new_text();
}

void BufferedTdtStream::drain_ready(bool flush, std::vector<int32_t>& out) {
    for (;;) {
        // A chunk is decodable once its own audio AND its right context have
        // arrived. On flush the tail is decoded with whatever right context
        // exists, which is what makes the last words appear at end of stream.
        const bool have_full = total_seen_ >= next_chunk_ + chunk_s_ + right_s_;
        const bool have_tail = flush && total_seen_ > next_chunk_;
        if (!have_full && !have_tail) return;

        const int64_t lo        = std::max<int64_t>(0, next_chunk_ - left_s_);
        const int64_t commit_hi = std::min<int64_t>(total_seen_, next_chunk_ + chunk_s_);
        const int64_t hi        = std::min<int64_t>(total_seen_, commit_hi + right_s_);
        if (commit_hi <= next_chunk_) return;

        // The retained buffer must cover [lo, hi); it always does, because it is
        // trimmed to next_chunk_ - left_s_ at the end of each iteration.
        const size_t off_lo = (size_t)(lo - audio_origin_);
        const size_t off_hi = (size_t)(hi - audio_origin_);
        if (off_hi > audio_.size() || off_lo > off_hi) return;

        // Frontend on this window's AUDIO: per_feature statistics are the
        // window's, matching NeMo's buffered script.
        std::vector<float> window_audio(audio_.begin() + off_lo, audio_.begin() + off_hi);
        std::vector<float> mel;
        int n_mels = 0, window_T = 0;
        frontend_.compute(window_audio, mel, n_mels, window_T);
        if (window_T <= 0) return;

        std::vector<float> enc_cf;
        int d_model = 0, Tw = 0;
        enc_.forward(mel, n_mels, window_T, enc_cf, d_model, Tw);
        if (Tw <= 0 || d_model <= 0) return;

        // Encoder frames for this chunk. Every span is a whole number of
        // encoder frames (see the schedule rounding in the constructor), so
        // this is exact division rather than a length recurrence — and it is
        // what NeMo does: it slices encoder_output[:, encoder_context.left:]
        // with left already expressed in subsampled frames.
        //
        // valid_out_len is the wrong tool here despite being the encoder's own
        // authority on lengths: valid_out_len(T, 0) returns 1, not 0, because
        // its per-stage recurrence adds one after the division. Using it for
        // the chunk offset therefore skipped encoder frame 0 of the very first
        // chunk on every stream.
        const int64_t enc_frame = (int64_t)hop_ * sub_factor_;
        const int first = std::min<int>(Tw, (int)((next_chunk_ - lo) / enc_frame));
        const int last  = std::min<int>(Tw, (int)((commit_hi  - lo) / enc_frame));
        if (last <= first) { next_chunk_ = commit_hi; continue; }

        // Channels-first [d_model, Tw] -> time-major [n, d_model].
        const int n = last - first;
        std::vector<float> frames((size_t)n * d_model);
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < d_model; ++c)
                frames[(size_t)t * d_model + c] = enc_cf[(size_t)c * Tw + (first + t)];

        std::vector<TokenInfo> chunk_tokens;
        std::vector<int32_t> step =
            tdt_decode_frames(pred_, joint_, frames, n, d_model, durations_,
                              state_, blank_id_, max_symbols_, &chunk_tokens);

        // tdt_decode_frames records ABSOLUTE frames (frames_seen + t), so no
        // base offset is added here — hence the null local_frames.
        acc_.advance_frames(n);
        acc_.absorb(step, chunk_tokens, /*local_frames=*/nullptr, /*base_frame=*/0);
        acc_.regroup_words(/*flush_all=*/false);
        out.insert(out.end(), step.begin(), step.end());

        // Preview the right-context region from a COPY of the decoder state.
        // Those encoder frames are already computed; only the decode is extra,
        // and the copy means nothing here can touch the committed transcript.
        // Bounded to the frames backed by real audio, so end-of-stream padding
        // is never previewed.
        if (speculate_) {
            int tail_avail = (int)((hi - commit_hi) / enc_frame);
            if (preview_secs_ > 0) {
                const double sr = (double)(ml_.config().sample_rate
                                           ? ml_.config().sample_rate : 16000);
                const int cap = (int)(preview_secs_ * sr / (double)enc_frame);
                tail_avail = std::min(tail_avail, std::max(cap, 1));
            }
            const int tail_n = std::min(Tw - last, tail_avail);
            if (tail_n > 0) preview_frames(enc_cf, Tw, d_model, last, tail_n);
            else            tentative_.clear();
        }

        next_chunk_ = commit_hi;

        // Drop audio older than the next window's left context. This is what
        // keeps memory flat over a long stream instead of growing with it.
        const int64_t keep_from = std::max<int64_t>(0, next_chunk_ - left_s_);
        if (keep_from > audio_origin_) {
            const size_t drop = (size_t)(keep_from - audio_origin_);
            if (drop >= audio_.size()) { audio_.clear(); }
            else { audio_.erase(audio_.begin(), audio_.begin() + drop); }
            audio_origin_ = keep_from;
        }
        // Keep going: a flush can span several chunks, and stopping after the
        // first partial one silently drops the end of the stream. Termination
        // is the loop head — have_tail goes false once next_chunk_ reaches
        // total_seen_.
    }
}


void BufferedTdtStream::preview_frames(const std::vector<float>& enc_cf, int Tw,
                                       int d_model, int from, int n) {
    if (n <= 0) { tentative_.clear(); return; }
    std::vector<float> frames((size_t)n * d_model);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < d_model; ++c)
            frames[(size_t)t * d_model + c] = enc_cf[(size_t)c * Tw + (from + t)];
    // A COPY of the decoder state: the full carried linguistic history, so the
    // preview reads as a continuation rather than a cold start — and discarded
    // on return, so nothing here can reach the committed transcript.
    TdtDecodeState spec = state_;
    const std::vector<int32_t> ids =
        tdt_decode_frames(pred_, joint_, frames, n, d_model, durations_, spec,
                          blank_id_, max_symbols_);
    tentative_ = detokenize(ml_.config().tokenizer_pieces,
                            strip_special_tokens(ml_.config().tokenizer_pieces, ids));
}

void BufferedTdtStream::preview_span(int64_t lo, int64_t hi) {
    if (hi <= lo) return;
    const size_t off_lo = (size_t)(lo - audio_origin_);
    const size_t off_hi = (size_t)(hi - audio_origin_);
    if (off_hi > audio_.size() || off_lo > off_hi) return;

    std::vector<float> window_audio(audio_.begin() + off_lo, audio_.begin() + off_hi);
    std::vector<float> mel;
    int n_mels = 0, window_T = 0;
    frontend_.compute(window_audio, mel, n_mels, window_T);
    if (window_T <= 0) return;

    std::vector<float> enc_cf;
    int d_model = 0, Tw = 0;
    enc_.forward(mel, n_mels, window_T, enc_cf, d_model, Tw);
    if (Tw <= 0 || d_model <= 0) return;

    // Only the frames backed by real audio; the frontend's trailing pad frame
    // would otherwise be previewed as if it were speech.
    const int64_t enc_frame = (int64_t)hop_ * sub_factor_;
    const int valid = std::min(Tw, (int)((hi - lo) / enc_frame));
    preview_frames(enc_cf, Tw, d_model, 0, valid);
}

} // namespace pk
