// The speculative tail: preview the right-context region without touching
// what gets committed.
//
// Three properties:
//   1. Committed output is byte-identical with speculation on and off. The
//      preview decodes from a COPY of the decoder state; if that copy ever
//      leaked back, this is what would catch it.
//   2. The preview is non-empty in mid-stream, i.e. it actually previews
//      something rather than silently doing nothing.
//   3. The preview is mostly right: measured against what the very next chunk
//      goes on to commit. Reported, not asserted at a threshold — the point is
//      to know the number, and a hard bound here would be fitting a rule to
//      one fixture.
//
// Env (skip 77 when absent): PARAKEET_TEST_GGUF_06B, PARAKEET_TEST_AUDIO.
#include "audio_io.hpp"
#include "buffered_window.hpp"
#include "model_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        size_t j = i;
        while (j < s.size() && s[j] != ' ') ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// Words the stream has committed so far, as a flat list.
std::vector<std::string> committed(pk::BufferedTdtStream& s) {
    return split(s.text());
}

} // namespace

int main() {
    const char* model = std::getenv("PARAKEET_TEST_GGUF_06B");
    const char* wav   = std::getenv("PARAKEET_TEST_AUDIO");
    if (!model || !wav) return 77;

    pk::ModelLoader ml;
    if (!ml.load(model)) return 1;
    if (!pk::BufferedTdtStream::supports(ml)) return 77;
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) return 1;

    const int block = 1600;  // 100 ms, the app's capture block

    // --- 1. committed text must not depend on speculation ---
    //
    // Preview quality is measured in CHARACTERS against the settled
    // transcript, not in words against the next commit. A chunk boundary
    // routinely lands mid-word: the commit ends "...Moh" and the next chunk's
    // "an's" merges back into it to make "Mohan's". Word-position comparison
    // therefore drifts by a fragment at every boundary and scores correct
    // previews as wrong (it read 45% on text that is plainly almost right).
    // What the reader actually experiences is how much of the tail text stays
    // put once it settles, which is a character-level question.
    std::string with_spec, without_spec;
    int previews = 0;
    long chars_total = 0, chars_agree = 0;
    struct Claim { size_t offset; std::string text; };
    std::vector<Claim> claims;

    for (int spec = 1; spec >= 0; --spec) {
        pk::BufferedTdtStream stream(ml);
        stream.set_speculate(spec != 0);
        if (const char* d = std::getenv("PARAKEET_PREVIEW_SECS"))
            stream.set_preview_secs(std::atof(d));
        size_t at = 0;
        while (at < audio.samples.size()) {
            const int n = (int)std::min((size_t)block, audio.samples.size() - at);
            const bool last = (at + (size_t)n) >= audio.samples.size();
            if (spec) {
                // What the tail claims, and where in the transcript it claims
                // it: immediately after everything committed so far.
                // Record a claim only when it changes: the preview persists
                // across feeds that do not cross a chunk boundary, and counting
                // it once per 100 ms feed would weight one claim twenty times.
                const std::string preview = stream.tentative_text();
                if (!preview.empty()) {
                    const size_t off = stream.text().size();
                    if (claims.empty() || claims.back().text != preview
                                       || claims.back().offset != off)
                        claims.push_back({off, preview});
                }
            }
            stream.feed_pcm(audio.samples.data() + at, n, last);
            at += (size_t)n;
        }
        stream.finalize();
        (spec ? with_spec : without_spec) = stream.text();
    }

    // Score each claim against the settled transcript at the same offset.
    for (const Claim& c : claims) {
        if (c.offset >= with_spec.size()) continue;
        // detokenize emits a leading space before each word, so the settled
        // text at this offset starts with one while the claim does not.
        // Comparing without skipping it shifts every character and reports a
        // correct preview as half wrong.
        size_t off = c.offset;
        while (off < with_spec.size() && with_spec[off] == ' ') ++off;
        std::string claim = c.text;
        size_t cb = 0;
        while (cb < claim.size() && claim[cb] == ' ') ++cb;
        claim = claim.substr(cb);
        const std::string actual = with_spec.substr(off, claim.size());
        const size_t n = std::min(actual.size(), claim.size());
        if (n == 0) continue;
        ++previews;
        for (size_t k = 0; k < n; ++k) {
            ++chars_total;
            if (actual[k] == claim[k]) ++chars_agree;
        }
        if (std::getenv("PARAKEET_PREVIEW_SHOW") && previews <= 6) {
            std::fprintf(stderr, "  claimed: %s\n  settled: %s\n\n",
                         claim.c_str(), actual.c_str());
        }
    }

    int failures = 0;
    if (with_spec != without_spec) {
        std::fprintf(stderr, "COMMITTED TEXT DIFFERS with speculation on/off\n"
                             "  on : %s\n  off: %s\n",
                     with_spec.c_str(), without_spec.c_str());
        ++failures;
    } else {
        std::fprintf(stderr, "committed text identical with speculation on/off\n");
    }

    // --- 2. the preview must actually preview something ---
    if (previews == 0) {
        std::fprintf(stderr, "no previews were produced — speculation did nothing\n");
        ++failures;
    }

    // --- 3. how much of the tail survives settling, character-wise ---
    if (chars_total > 0) {
        std::fprintf(stderr, "tail survives settling: %ld/%ld chars (%.0f%%), "
                             "over %d previews\n",
                     chars_agree, chars_total,
                     100.0 * (double)chars_agree / (double)chars_total, previews);
    }

    if (failures) { std::fprintf(stderr, "FAIL: %d\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
