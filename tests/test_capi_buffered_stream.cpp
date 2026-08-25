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
//      same text AND the same previews. Checking text alone missed a real bug —
//      the onset watermark was not cleared by reset, so the second utterance
//      produced no preview until its first commit, which is exactly the wait
//      the preview exists to remove. Committed text was identical throughout.
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

// How many distinct non-empty previews the JSON feed produced, and how early
// the first one arrived (in feeds). Both must survive a reset.
struct PreviewStats {
    int distinct = 0;
    int first_feed = -1;
};

static std::string capi_stream_text(parakeet_ctx* ctx, parakeet_stream* s,
                                    const std::vector<float>& pcm, bool& eou_seen,
                                    PreviewStats* previews = nullptr) {
    std::string text;
    const int block = 1600;  // 100 ms at 16 kHz, the app's capture block
    size_t at = 0;
    int feed_index = 0;
    std::string last_preview;
    while (at < pcm.size()) {
        const int n = (int)std::min((size_t)block, pcm.size() - at);
        int eou = 0;
        char* t = nullptr;
        if (previews) {
            // "tentative" only exists in the JSON document, which is also what
            // opts this stream into speculation.
            char* doc = parakeet_capi_stream_feed_json(s, pcm.data() + at, n);
            if (!doc) { std::fprintf(stderr, "feed_json failed: %s\n", parakeet_capi_last_error(ctx)); return text; }
            const std::string json(doc);
            parakeet_capi_free_string(doc);
            const std::string key = "\"tentative\":\"";
            const size_t k = json.find(key);
            std::string preview;
            if (k != std::string::npos) {
                const size_t b = k + key.size();
                const size_t e = json.find('"', b);
                if (e != std::string::npos) preview = json.substr(b, e - b);
            }
            if (!preview.empty() && preview != last_preview) {
                ++previews->distinct;
                if (previews->first_feed < 0) previews->first_feed = feed_index;
            }
            last_preview = preview;
            // The JSON document's "text" is the same newly-final text the plain
            // entry point returns; parse it the same shallow way.
            const std::string tkey = "{\"text\":\"";
            if (json.rfind(tkey, 0) == 0) {
                const size_t e = json.find('"', tkey.size());
                if (e != std::string::npos) text += json.substr(tkey.size(), e - tkey.size());
            }
            ++feed_index;
            at += (size_t)n;
            continue;
        }
        t = parakeet_capi_stream_feed(s, pcm.data() + at, n, &eou);
        if (!t) { std::fprintf(stderr, "feed failed: %s\n", parakeet_capi_last_error(ctx)); return text; }
        if (eou) eou_seen = true;
        text += t;
        parakeet_capi_free_string(t);
        ++feed_index;
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

    // Reset is checked once, on the JSON path below, rather than twice. The
    // plain path's reset calls the same parakeet_capi_stream_reset and differs
    // only in speculation being off — and the JSON version is the stronger
    // test, since a reset that forgets the preview watermark leaves committed
    // text identical while silently withholding every preview.

    // Previews must survive a reset as well as committed text. Run the JSON
    // path twice across a reset and require the second utterance to preview as
    // early and as often as the first: a stale onset watermark leaves committed
    // text identical while silently withholding every preview until the first
    // commit.
    if (parakeet_capi_stream_reset(s) != 0) {
        std::fprintf(stderr, "reset (json) failed: %s\n", parakeet_capi_last_error(ctx)); ++failures;
    } else {
        bool ignored = false;
        PreviewStats before, after;
        const std::string first_text  = capi_stream_text(ctx, s, audio.samples, ignored, &before);
        if (parakeet_capi_stream_reset(s) != 0) {
            std::fprintf(stderr, "second reset failed: %s\n", parakeet_capi_last_error(ctx)); ++failures;
        } else {
            const std::string second_text = capi_stream_text(ctx, s, audio.samples, ignored, &after);
            std::fprintf(stderr,
                "previews before reset: %d distinct, first at feed %d\n"
                "previews after  reset: %d distinct, first at feed %d\n",
                before.distinct, before.first_feed, after.distinct, after.first_feed);
            if (before.distinct == 0) {
                std::fprintf(stderr, "no previews at all on the JSON path\n"); ++failures;
            }
            if (after.distinct != before.distinct || after.first_feed != before.first_feed) {
                std::fprintf(stderr, "PREVIEWS DID NOT SURVIVE RESET\n"); ++failures;
            }
            if (second_text != first_text) {
                std::fprintf(stderr, "json text differs across reset\n"); ++failures;
            }
        }
    }

    parakeet_capi_stream_free(s);
    parakeet_capi_free(ctx);
    if (failures) { std::fprintf(stderr, "FAIL: %d\n", failures); return 1; }
    std::fprintf(stderr, "PASS (abi=%d)\n", parakeet_capi_abi_version());
    return 0;
}
