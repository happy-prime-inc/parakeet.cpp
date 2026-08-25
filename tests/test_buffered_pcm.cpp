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
//   PARAKEET_STREAM_SCHEDULE    optional "left,chunk,right" seconds (default
//                               10,2,2) — the latency/accuracy sweep knob.
//                               The final transcript goes to stdout so the
//                               sweep can score it against a reference.
#include "audio_io.hpp"
#include "buffered_window.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
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

static void schedule(double& left, double& chunk, double& right) {
    left = 10.0; chunk = 2.0; right = 2.0;
    if (const char* e = std::getenv("PARAKEET_STREAM_SCHEDULE")) {
        double l = 0, c = 0, r = 0;
        if (std::sscanf(e, "%lf,%lf,%lf", &l, &c, &r) == 3 && c > 0 && r >= 0 && l >= 0) {
            left = l; chunk = c; right = r;
        } else {
            std::fprintf(stderr, "ignoring malformed PARAKEET_STREAM_SCHEDULE=%s\n", e);
        }
    }
}

static std::vector<int32_t> run(const pk::ModelLoader& ml, const std::vector<float>& pcm,
                                int block, std::string* text_out = nullptr,
                                int64_t* frames_out = nullptr) {
    double left, chunk, right;
    schedule(left, chunk, right);
    pk::BufferedTdtStream stream(ml, left, chunk, right);
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
    if (text_out) *text_out = stream.text();
    if (frames_out) *frames_out = stream.frames_decoded();
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
    double left, chunk, right;
    schedule(left, chunk, right);
    std::string text;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<int32_t> oracle = run(ml, audio.samples, /*block=*/0, &text);
    const double infer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("%s\n", text.c_str());
    std::fprintf(stderr,
        "schedule=%g,%g,%g one-shot tokens=%zu inference_ms=%.0f audio_s=%.1f\n",
        left, chunk, right, oracle.size(), infer_ms,
        audio.samples.size() / 16000.0);
    if (oracle.empty()) { std::fprintf(stderr, "one-shot produced nothing\n"); return 1; }

    // 1600 = 100 ms, the app's capture block. 1237 and 999 are deliberately not
    // multiples of the 160-sample hop, so frames straddle feed boundaries.
    // PARAKEET_TEST_ONESHOT_ONLY skips this loop: a latency/accuracy sweep
    // over a 200s file needs the transcript once, not seven decodes of it.
    int failures = 0;
    const bool oneshot_only = std::getenv("PARAKEET_TEST_ONESHOT_ONLY") != nullptr;
    for (int block : oneshot_only ? std::initializer_list<int>{}
                                  : std::initializer_list<int>{1600, 1237, 999, 32000, 7}) {
        const std::vector<int32_t> got = run(ml, audio.samples, block);
        const bool same = (got == oracle);
        std::fprintf(stderr, "block=%-6d tokens=%-5zu %s\n",
                     block, got.size(), same ? "match" : "MISMATCH");
        if (!same) ++failures;
    }

    // Every encoder frame produced from real audio must be decoded, including
    // the partial one a stream that does not end on an encoder-frame boundary
    // produces. Flooring the final chunk's endpoint silently discarded it, and
    // token parity did not notice because the fixture happens to emit nothing
    // on that frame — so assert the frame count directly rather than trusting a
    // fixture to land a word there.
    {
        const auto& cfg = ml.config();
        const int64_t hop = cfg.hop_length ? (int64_t)cfg.hop_length : 160;
        const int64_t sub = cfg.subsampling_factor ? (int64_t)cfg.subsampling_factor : 8;
        const int64_t enc_frame = hop * sub;
        const int64_t n = (int64_t)audio.samples.size();
        const int64_t expect = (n + enc_frame - 1) / enc_frame;  // ceiling
        int64_t got = 0;
        run(ml, audio.samples, /*block=*/1600, nullptr, &got);
        std::fprintf(stderr, "frames decoded=%lld expected=%lld (%lld samples)\n",
                     (long long)got, (long long)expect, (long long)n);
        if (got != expect) {
            std::fprintf(stderr, "END-OF-STREAM FRAME LOSS: %lld frame(s) never decoded\n",
                         (long long)(expect - got));
            ++failures;
        }
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
