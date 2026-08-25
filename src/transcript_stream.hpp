#pragma once
#include "decode_types.hpp"
#include "model_loader.hpp"
#include "transcription.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace pk {

// End-of-utterance / backchannel event emitted by a streaming decoder.
//
// The model nvidia/parakeet_realtime_eou_120m-v1 emits <EOU> (id 1024) and
// <EOB> (id 1025) as regular vocab tokens to mark end-of-utterance /
// backchannel. They are surfaced as events (stripped from the running text)
// with the encoder frame on which they were emitted and a wall-clock time:
//   time_sec = encoder_frame * hop_length * subsampling_factor / sample_rate
// (the subsampled encoder frame stride in seconds), matching NeMo's
// _get_eou_predictions_from_hypotheses timestamp scaling.
struct EouEvent {
    int32_t token = 0;             // the special token id (<EOU> or <EOB>)
    bool    is_eob = false;        // true for <EOB>, false for <EOU>
    int     encoder_frame = 0;     // encoder-output frame index of the emission
    double  time_sec = 0.0;        // encoder_frame * hop * subsampling / sample_rate
};

// The transcript-side bookkeeping a streaming session does after its decoder
// has emitted tokens: running stripped text with a byte-offset delta marker,
// <EOU>/<EOB> events with absolute frame indices, and per-word timestamp
// accumulation with a finalization watermark.
//
// None of this depends on WHICH encoder produced the frames or WHICH decoder
// emitted the tokens, so it is shared by the cache-aware RNN-T path and the
// buffered TDT path rather than written twice. Extracting it is a pure
// refactor: the cache-aware transcript must stay byte-identical.
class TranscriptAccumulator {
public:
    // Resolves the per-encoder-frame stride and the <EOU>/<EOB> token ids from
    // the model itself. The ids stay -1 when the vocabulary has no such pieces
    // (every offline model), in which case no event can ever fire and the event
    // surface stays empty rather than being stubbed.
    explicit TranscriptAccumulator(const ModelLoader& ml);

    // hop_length * subsampling_factor / sample_rate — seconds per encoder frame.
    double frame_sec() const { return frame_sec_; }

    void reset();

    // Absorb one decode step's output.
    //
    // `emitted` are the token ids emitted in this step, in order.
    // `chunk_tokens` is the parallel per-token TokenInfo from the decoder.
    // `local_frames`, when non-null, holds each token's frame index RELATIVE to
    // this step, and `base_frame` is added to make it absolute — the RNN-T
    // convention. When null, TokenInfo::frame is already absolute and
    // `base_frame` is ignored — the TDT convention, where tdt_decode_frames
    // records `frames_seen + t` itself.
    //
    // Returns true if this step emitted an <EOU>/<EOB>.
    bool absorb(const std::vector<int32_t>& emitted,
                const std::vector<TokenInfo>& chunk_tokens,
                const std::vector<int32_t>* local_frames,
                int base_frame);

    // Regroup accumulated tokens into words. Mid-stream the last word is still
    // open (its text/end can change when more tokens arrive) so only words
    // before it become final; flush_all makes every word final at end-of-stream.
    void regroup_words(bool flush_all);

    // Running transcript with <EOU>/<EOB> stripped.
    const std::string& text() const { return text_; }

    // Text appended since the previous take; resets the delta marker.
    std::string take_new_text();

    // Words finalized since the previous drain.
    std::vector<Word> drain_words();

    // Move out all events collected so far.
    std::vector<EouEvent> drain_events();

    // Peek without consuming (the C-API uses the size as a watermark to
    // attribute events to one feed pass while leaving the queue intact).
    const std::vector<EouEvent>& events() const { return events_; }

    bool last_step_had_eou() const { return last_step_had_eou_; }

    // Running encoder-output frame counter, advanced by the caller as it feeds.
    int  encoder_frame() const { return enc_frame_; }
    void advance_frames(int n) { enc_frame_ += n; }

private:
    const ModelLoader& ml_;
    double frame_sec_ = 0.0;
    float  frame_sec_f_ = 0.0f;    // group_words takes a float
    int    eou_id_ = -1;
    int    eob_id_ = -1;

    int enc_frame_ = 0;

    std::vector<int32_t> non_special_;  // non-special tokens, for detokenize
    std::string text_;
    size_t text_taken_ = 0;             // byte offset of text_ already returned

    bool last_step_had_eou_ = false;
    std::vector<EouEvent> events_;

    std::vector<TokenInfo> word_tokens_;  // absolute-frame non-special tokens
    std::vector<Word> words_;             // last regrouping of word_tokens_
    size_t words_finalized_ = 0;
    size_t words_taken_ = 0;
};

} // namespace pk
