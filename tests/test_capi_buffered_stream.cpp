// The streaming C API driving an OFFLINE TDT checkpoint (ABI v7).
//
// Checks, against the same clip fed as 100 ms blocks:
//   1. begin succeeds for a TDT model (the v6 API refused it);
//   2. feed+finalize through the C boundary produces exactly the text the
//      C++ BufferedTdtStream produces — the wrapper adds nothing and loses
//      nothing;
//   3. *eou_out stays 0 and drain_events returns 0 (TDT vocabularies have no
//      <EOU>/<EOB>; the API must not fake them);
//   4. stream_reset yields a fresh stream: the same audio again gives the
//      same text.
//
// Env (skip 77 when absent): PARAKEET_TEST_GGUF_06B, PARAKEET_TEST_AUDIO.
#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "buffered_window.hpp"
#include "model_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string capi_stream_text(parakeet_ctx* ctx, parakeet_stream* s,
                                    const std::vector<float>& pcm, bool& eou_seen) {
    std::string text;
    const int block = 1600;  // 100 ms at 16 kHz, the app's capture block
    size_t at = 0;
    while (at < pcm.size()) {
        const int n = (int)std::min((size_t)block, pcm.size() - at);
        int eou = 0;
        char* t = parakeet_capi_stream_feed(s, pcm.data() + at, n, &eou);
        if (!t) { std::fprintf(stderr, "feed failed: %s\n", parakeet_capi_last_error(ctx)); return text; }
        if (eou) eou_seen = true;
        text += t;
        parakeet_capi_free_string(t);
        at += (size_t)n;
    }
    char* tail = parakeet_capi_stream_finalize(s);
    if (!tail) { std::fprintf(stderr, "finalize failed: %s\n", parakeet_capi_last_error(ctx)); return text; }
    text += tail;
    parakeet_capi_free_string(tail);
    return text;
}

int main() {
    const char* model = std::getenv("PARAKEET_TEST_GGUF_06B");
    const char* wav   = std::getenv("PARAKEET_TEST_AUDIO");
    if (!model || !wav) return 77;

    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) return 1;

    // Reference: the C++ class, one shot.
    std::string expect;
    {
        pk::ModelLoader ml;
        if (!ml.load(model)) return 1;
        if (!pk::BufferedTdtStream::supports(ml)) return 77;
        pk::BufferedTdtStream stream(ml);
        stream.feed_pcm(audio.samples.data(), (int)audio.samples.size(), true);
        stream.finalize();
        expect = stream.text();
    }

    parakeet_ctx* ctx = parakeet_capi_load(model);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }
    parakeet_stream* s = parakeet_capi_stream_begin(ctx);
    if (!s) { std::fprintf(stderr, "begin failed: %s\n", parakeet_capi_last_error(ctx)); return 1; }

    int failures = 0;
    bool eou_seen = false;
    const std::string got = capi_stream_text(ctx, s, audio.samples, eou_seen);
    std::fprintf(stderr, "capi text: %s\n", got.c_str());
    if (got != expect) { std::fprintf(stderr, "MISMATCH vs BufferedTdtStream:\n  %s\n", expect.c_str()); ++failures; }
    if (eou_seen) { std::fprintf(stderr, "eou_out fired on a TDT model\n"); ++failures; }

    parakeet_stream_event* evs = nullptr;
    const int n_evs = parakeet_capi_stream_drain_events(s, &evs);
    if (n_evs != 0) { std::fprintf(stderr, "drain_events=%d, want 0\n", n_evs); ++failures; }
    parakeet_capi_free_events(evs);

    if (parakeet_capi_stream_reset(s) != 0) {
        std::fprintf(stderr, "reset failed: %s\n", parakeet_capi_last_error(ctx)); ++failures;
    } else {
        bool eou2 = false;
        const std::string again = capi_stream_text(ctx, s, audio.samples, eou2);
        if (again != expect) { std::fprintf(stderr, "post-reset MISMATCH: %s\n", again.c_str()); ++failures; }
    }

    parakeet_capi_stream_free(s);
    parakeet_capi_free(ctx);
    if (failures) { std::fprintf(stderr, "FAIL: %d\n", failures); return 1; }
    std::fprintf(stderr, "PASS (abi=%d)\n", parakeet_capi_abi_version());
    return 0;
}
