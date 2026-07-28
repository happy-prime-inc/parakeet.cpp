#include "model.hpp"
#include "audio_io.hpp"
#include "search.hpp"
#include "tokenizer.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

// Self-consistency check for Model::transcribe_pcm_ctc_logits (the
// classroom-captions#63 logits-exposure entry point): reconstructing text from
// the exposed [T, vocab+1] log-prob matrix via the SAME ctc_greedy +
// detokenize path decode_enc_out uses internally must reproduce
// transcribe_pcm(..., kCTC)'s own greedy transcript byte-for-byte, on a real
// standalone-CTC checkpoint (parakeet-ctc-0.6b / parakeet-ctc-1.1b: decoder.*
// prefix, not the hybrid ctc_decoder.* prefix — exercises the ctc_head_tensor
// fallback path too).
//
// This is a self-consistency test (our own greedy decode vs. our own exposed
// logits, both computed here), not a NeMo parity check — that's already
// covered by test_transcribe_ctc.cpp / test_ctc.cpp.
//
// Env:
//   PARAKEET_TEST_GGUF_CTC   path to a standalone CTC GGUF (skip 77 if unset)
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; wav path is relative)
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_CTC");
    if (!gguf) {
        std::fprintf(stderr, "test_capi_ctc_logits: PARAKEET_TEST_GGUF_CTC not set; skip\n");
        return 77;
    }

    auto model = pk::Model::load(gguf);
    if (!model) {
        std::fprintf(stderr, "test_capi_ctc_logits: load failed for %s\n", gguf);
        return 1;
    }

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio) || audio.samples.empty()) {
        std::fprintf(stderr, "test_capi_ctc_logits: wav load failed\n");
        return 1;
    }

    const std::string reference = model->transcribe_pcm(audio.samples, 16000, pk::Decoder::kCTC);

    std::vector<float> logits;
    int T = 0, vocab_plus_1 = 0;
    model->transcribe_pcm_ctc_logits(audio.samples, 16000, logits, T, vocab_plus_1);

    if (T <= 0 || vocab_plus_1 <= 0 || (size_t)T * (size_t)vocab_plus_1 != logits.size()) {
        std::fprintf(stderr,
            "test_capi_ctc_logits: bad shape T=%d vocab_plus_1=%d logits.size()=%zu\n",
            T, vocab_plus_1, logits.size());
        return 1;
    }

    const int blank_id = (int)model->config().blank_id;
    std::vector<int32_t> ids = pk::ctc_greedy(logits, T, vocab_plus_1, blank_id);
    const std::string reconstructed = pk::detokenize(
        model->loader().tokenizer_pieces(),
        pk::strip_special_tokens(model->loader().tokenizer_pieces(), ids));

    std::fprintf(stderr, "test_capi_ctc_logits: reference     = %s\n", reference.c_str());
    std::fprintf(stderr, "test_capi_ctc_logits: reconstructed = %s\n", reconstructed.c_str());
    std::fprintf(stderr, "test_capi_ctc_logits: T=%d vocab_plus_1=%d blank_id=%d\n",
                 T, vocab_plus_1, blank_id);

    if (reconstructed != reference) {
        std::fprintf(stderr, "test_capi_ctc_logits: MISMATCH\n");
        return 1;
    }

    std::fprintf(stderr,
        "test_capi_ctc_logits: PASS (argmax-greedy over exposed logits reproduces "
        "the CLI's own greedy text)\n");
    return 0;
}
