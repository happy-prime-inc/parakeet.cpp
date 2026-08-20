#include "buffered_window.hpp"
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
    const int sub = (int)(cfg.subsampling_factor ? cfg.subsampling_factor : 8);
    const int64_t enc_frame_samples = (int64_t)hop_ * sub;
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

        // Encoder frames for this chunk. valid_out_len is the encoder's own
        // length recurrence (and handles causal downsampling), rather than a
        // local ceil(n/8) that silently assumes the non-causal case.
        const int mel_off    = (int)((next_chunk_ - lo) / hop_);
        const int mel_commit = (int)((commit_hi  - lo) / hop_);
        const int first = std::min(Tw, sub_.valid_out_len(window_T, mel_off));
        const int last  = std::min(Tw, sub_.valid_out_len(window_T, mel_commit));
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

} // namespace pk
