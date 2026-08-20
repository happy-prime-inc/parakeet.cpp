// Incremental PCM feeding for the buffered TDT path.
//
// Two properties, both of which the window loop can silently break:
//
//   1. Block-size independence. Feeding the same audio as 100 ms blocks, as
//      awkward 1237-sample blocks, or in one call must produce the same tokens.
//      Anything that mishandles a chunk boundary — the retained-audio trim, the
//      mel window offsets, the encoder-frame trim, the TDT duration skip
//      crossing a boundary — shows up as a difference here.
//   2. Parity with NeMo, when a reference token sequence is supplied.
//
// Inputs come from the environment so ctest can skip cleanly (77) when the
// model or audio is absent:
//   PARAKEET_TEST_GGUF_06B      offline TDT checkpoint
//   PARAKEET_TEST_AUDIO         16 kHz mono WAV
//   PARAKEET_TEST_NEMO_BASELINE optional NeMo token reference (text or GGUF)
#include "audio_io.hpp"
#include "buffered_window.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static bool read_reference_tokens(const char* path, std::vector<int32_t>& out) {
    const std::string filename(path);
    if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".gguf") {
        gguf_init_params params = {};
        params.no_alloc = false;
        ggml_context* ctx = nullptr;
        params.ctx = &ctx;
        gguf_context* gguf = gguf_init_from_file(path, params);
        if (!gguf) return false;
        bool ok = false;
        for (const char* name : {"tdt_token_ids", "rnnt_token_ids"}) {
            ggml_tensor* t = ggml_get_tensor(ctx, name);
            if (!t) continue;
            const int32_t* data = (const int32_t*)t->data;
            out.assign(data, data + ggml_nelements(t));
            ok = true;
            break;
        }
        gguf_free(gguf);
        if (ctx) ggml_free(ctx);
        return ok;
    }
    std::ifstream input(path);
    int32_t token = 0;
    while (input >> token) out.push_back(token);
    return !out.empty();
}

static std::vector<int32_t> run(const pk::ModelLoader& ml, const std::vector<float>& pcm,
                                int block) {
    pk::BufferedTdtStream stream(ml);
    if (block <= 0) {
        stream.feed_pcm(pcm.data(), (int)pcm.size(), /*is_last=*/true);
    } else {
        size_t at = 0;
        while (at < pcm.size()) {
            const int n = (int)std::min((size_t)block, pcm.size() - at);
            const bool last = (at + (size_t)n) >= pcm.size();
            stream.feed_pcm(pcm.data() + at, n, last);
            at += (size_t)n;
        }
    }
    stream.finalize();
    return stream.tokens();
}

int main() {
    const char* model = std::getenv("PARAKEET_TEST_GGUF_06B");
    const char* wav   = std::getenv("PARAKEET_TEST_AUDIO");
    if (!model || !wav) return 77;

    pk::ModelLoader ml;
    if (!ml.load(model)) return 1;
    if (!pk::BufferedTdtStream::supports(ml)) {
        std::fprintf(stderr, "model has no TDT durations; not a buffered-TDT model\n");
        return 77;
    }
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) return 1;

    // One-shot is the oracle: the same schedule, no PCM boundaries to get wrong.
    const std::vector<int32_t> oracle = run(ml, audio.samples, /*block=*/0);
    std::fprintf(stderr, "one-shot tokens=%zu\n", oracle.size());
    if (oracle.empty()) { std::fprintf(stderr, "one-shot produced nothing\n"); return 1; }

    // 1600 = 100 ms, the app's capture block. 1237 and 999 are deliberately not
    // multiples of the 160-sample hop, so frames straddle feed boundaries.
    int failures = 0;
    for (int block : {1600, 1237, 999, 32000, 7}) {
        const std::vector<int32_t> got = run(ml, audio.samples, block);
        const bool same = (got == oracle);
        std::fprintf(stderr, "block=%-6d tokens=%-5zu %s\n",
                     block, got.size(), same ? "match" : "MISMATCH");
        if (!same) ++failures;
    }

    if (const char* baseline = std::getenv("PARAKEET_TEST_NEMO_BASELINE")) {
        std::vector<int32_t> reference;
        if (!read_reference_tokens(baseline, reference)) {
            std::fprintf(stderr, "could not read reference %s\n", baseline);
            return 1;
        }
        const bool same = (oracle == reference);
        std::fprintf(stderr, "nemo_tokens=%zu buffered_tokens=%zu %s\n",
                     reference.size(), oracle.size(), same ? "match" : "MISMATCH");
        if (!same) ++failures;
    }

    if (failures) { std::fprintf(stderr, "FAIL: %d check(s)\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
