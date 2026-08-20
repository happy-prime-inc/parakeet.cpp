#pragma once
#include "encoder.hpp"
#include "joint.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "subsampling.hpp"
#include "tdt.hpp"
#include "transcript_stream.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace pk {

// Buffered streaming for OFFLINE TDT checkpoints (parakeet-tdt-0.6b-v2/v3).
//
// These models were never trained for cache-aware streaming: there is no
// per-layer conv/attention cache to carry, so the encoder cannot be resumed
// from where it stopped. NVIDIA's answer, which this reproduces, is to
// re-encode a sliding [left | chunk | right] window and decode only the chunk's
// frames once:
//
//   1. Run the offline encoder over [10 s left | 2 s chunk | 2 s right].
//   2. Keep only the encoder frames belonging to the chunk.
//   3. Decode those frames EXACTLY ONCE, carrying prediction-net state, the
//      last emitted token and any TDT duration skip into the next chunk.
//
// Because no audio is ever decoded twice, the transcript is append-only by
// construction rather than by reconciliation: there is no second reading of a
// span to disagree with the first.
//
// `TdtDecodeState::pending_skip` is what makes step 3 safe. TDT emits a token
// plus a duration, which may advance several encoder frames; when that skip
// crosses a chunk boundary the frames it consumed belong to the NEXT chunk, and
// decoding that chunk from frame zero would re-consume them.
//
// Normalization follows NeMo's buffered script: the frontend runs on each
// window's AUDIO, so per_feature statistics are the window's. Slicing a
// globally normalized mel is a different computation — it agrees on short clips
// (where every window starts at sample 0) and diverges on long ones — and it
// cannot run live, since whole-utterance statistics need the whole utterance.
//
// Shares TranscriptAccumulator with the cache-aware StreamingSession, so the
// running text, delta marker, event queue and per-word timestamps are one
// implementation rather than two.
class BufferedTdtStream {
public:
    // The 10/2/2 schedule is NVIDIA's inference-time recommendation, not model
    // metadata — there is no GGUF key for it — so it is a construction
    // parameter with those defaults. Each is rounded to a whole mel frame.
    BufferedTdtStream(const ModelLoader& ml,
                      double left_secs = 10.0,
                      double chunk_secs = 2.0,
                      double right_secs = 2.0);

    // True when the model is an offline TDT checkpoint this can drive.
    static bool supports(const ModelLoader& ml);

    void reset();

    // Feed 16 kHz mono PCM. Decodes every chunk whose right context has now
    // arrived and returns the token ids emitted by this call ("" worth of
    // tokens is normal until the first chunk + right context is buffered).
    // `is_last` flushes the tail, decoding the final partial chunk with
    // whatever right context exists.
    std::vector<int32_t> feed_pcm(const float* pcm, int n_samples, bool is_last = false);

    // Flush the tail without new audio and return the newly finalized text.
    std::string finalize();

    const std::vector<int32_t>& tokens() const { return state_.hyp; }
    const std::string& text() const { return acc_.text(); }
    std::string take_new_text() { return acc_.take_new_text(); }
    std::vector<Word> drain_words() { return acc_.drain_words(); }
    std::vector<EouEvent> drain_events() { return acc_.drain_events(); }

    // Seconds between audio arriving and its words being emitted: a chunk must
    // fill and its right context must follow. Callers that display text need
    // this; it is much larger than a cache-aware session's one-chunk lag.
    double latency_sec() const;

    // Audio samples retained right now. Bounded by left+chunk+right — the
    // buffer never grows with stream length.
    size_t buffered_samples() const { return audio_.size(); }

private:
    // Decode every chunk whose right context is available. `flush` also decodes
    // the trailing partial chunk. Appends emitted ids to `out`.
    void drain_ready(bool flush, std::vector<int32_t>& out);

    const ModelLoader& ml_;
    MelFrontend   frontend_;
    Subsampling   sub_;
    Encoder       enc_;
    PredictionNet pred_;
    Joint         joint_;
    TdtDecodeState state_;
    TranscriptAccumulator acc_;

    int hop_ = 160;
    int sub_factor_ = 8;           // encoder subsampling (mel frames per enc frame)
    int blank_id_ = 0;
    int max_symbols_ = 10;
    std::vector<int32_t> durations_;

    int64_t left_s_ = 0, chunk_s_ = 0, right_s_ = 0;  // schedule, in samples

    std::vector<float> audio_;      // retained tail, bounded by the schedule
    int64_t audio_origin_ = 0;      // absolute sample index of audio_[0]
    int64_t total_seen_ = 0;        // absolute samples fed so far
    int64_t next_chunk_ = 0;        // absolute sample index of the next chunk
    bool    finished_ = false;
};

} // namespace pk
