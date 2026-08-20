#include "transcript_stream.hpp"
#include "tokenizer.hpp"

namespace pk {

TranscriptAccumulator::TranscriptAccumulator(const ModelLoader& ml) : ml_(ml) {
    const ParakeetConfig& cfg = ml.config();
    // Per-encoder-frame stride in seconds: each encoder output frame spans
    // hop_length * subsampling_factor input samples.
    const double hop = (double)cfg.hop_length;
    const double sub = (double)(cfg.subsampling_factor ? cfg.subsampling_factor : 1);
    const double sr  = (double)(cfg.sample_rate ? cfg.sample_rate : 16000);
    frame_sec_   = (hop * sub) / sr;
    frame_sec_f_ = (float)frame_sec_;

    // Resolve <EOU>/<EOB> from the loaded vocab rather than hardcoding
    // 1024/1025. Offline vocabularies have neither, leaving both at -1.
    const auto& pieces = cfg.tokenizer_pieces;
    for (int i = 0; i < (int)pieces.size(); ++i) {
        if (pieces[i] == "<EOU>") eou_id_ = i;
        else if (pieces[i] == "<EOB>") eob_id_ = i;
    }
}

void TranscriptAccumulator::reset() {
    enc_frame_ = 0;
    non_special_.clear();
    text_.clear();
    text_taken_ = 0;
    last_step_had_eou_ = false;
    events_.clear();
    word_tokens_.clear();
    words_.clear();
    words_finalized_ = 0;
    words_taken_ = 0;
}

bool TranscriptAccumulator::absorb(const std::vector<int32_t>& emitted,
                                   const std::vector<TokenInfo>& chunk_tokens,
                                   const std::vector<int32_t>* local_frames,
                                   int base_frame) {
    last_step_had_eou_ = false;
    bool text_changed = false;
    for (size_t i = 0; i < emitted.size(); ++i) {
        const int32_t tok = emitted[i];
        if (tok == eou_id_ || tok == eob_id_) {
            // <EOU>/<EOB>: surface as an event, do NOT add to the text. The
            // absolute frame comes from the decoder's per-token frame index.
            EouEvent ev;
            ev.token  = tok;
            ev.is_eob = (tok == eob_id_);
            ev.encoder_frame = local_frames
                ? base_frame + (int)(*local_frames)[i]
                : (i < chunk_tokens.size() ? chunk_tokens[i].frame : enc_frame_);
            ev.time_sec = ev.encoder_frame * frame_sec_;
            events_.push_back(ev);
            last_step_had_eou_ = true;
        } else {
            non_special_.push_back(tok);
            text_changed = true;
            if (i < chunk_tokens.size()) {
                TokenInfo ti = chunk_tokens[i];
                // RNN-T reports a frame local to this step; TDT already records
                // an absolute one (frames_seen + t) inside tdt_decode_frames.
                if (local_frames) ti.frame += base_frame;
                word_tokens_.push_back(ti);
            }
        }
    }
    if (text_changed) {
        text_ = detokenize(ml_.config().tokenizer_pieces, non_special_);
    }
    return last_step_had_eou_;
}

void TranscriptAccumulator::regroup_words(bool flush_all) {
    // Re-run the validated offline grouping over the whole accumulated
    // non-special token sequence (it does the punctuation lookahead /
    // refinement exactly like the offline transcribe_with_timestamps path).
    words_ = group_words(word_tokens_, ml_.config().tokenizer_pieces, frame_sec_f_);
    if (words_.empty()) {
        words_finalized_ = 0;
    } else {
        words_finalized_ = flush_all ? words_.size() : (words_.size() - 1);
    }
    // Never "un-finalize" a word already handed out.
    if (words_finalized_ < words_taken_) words_finalized_ = words_taken_;
}

std::string TranscriptAccumulator::take_new_text() {
    if (text_taken_ >= text_.size()) return std::string();
    std::string delta = text_.substr(text_taken_);
    text_taken_ = text_.size();
    return delta;
}

std::vector<Word> TranscriptAccumulator::drain_words() {
    std::vector<Word> out;
    for (size_t i = words_taken_; i < words_finalized_ && i < words_.size(); ++i)
        out.push_back(words_[i]);
    words_taken_ = words_finalized_;
    return out;
}

std::vector<EouEvent> TranscriptAccumulator::drain_events() {
    std::vector<EouEvent> out;
    out.swap(events_);
    return out;
}

} // namespace pk
