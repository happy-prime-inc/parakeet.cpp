#include "audio_io.hpp"
#include "encoder.hpp"
#include "joint.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "tdt.hpp"
#include "tokenizer.hpp"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/resource.h>

static int ceil_div8(int n) { return n <= 0 ? 0 : (n + 7) / 8; }

static bool read_reference_tokens(const char* path, std::vector<int32_t>& out) {
    const std::string filename(path);
    if (filename.size() < 5 || filename.substr(filename.size() - 5) != ".gguf") {
        std::ifstream input(path);
        int32_t token = 0;
        while (input >> token) out.push_back(token);
        return input.eof() && !out.empty();
    }
    ggml_context* ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, &ctx};
    gguf_context* file = gguf_init_from_file(path, params);
    if (!file) return false;
    // gen_nemo_baseline currently labels pure-TDT 0.6B checkpoints as the
    // RNNT-only path, so accept either tensor name. Both contain the NeMo
    // hypothesis y_sequence and therefore have the same token semantics.
    ggml_tensor* tensor = ggml_get_tensor(ctx, "tdt_token_ids");
    if (!tensor) tensor = ggml_get_tensor(ctx, "rnnt_token_ids");
    if (!tensor || tensor->type != GGML_TYPE_I32) {
        gguf_free(file);
        ggml_free(ctx);
        return false;
    }
    out.resize((size_t)ggml_nelements(tensor));
    std::memcpy(out.data(), tensor->data, out.size() * sizeof(int32_t));
    gguf_free(file);
    ggml_free(ctx);
    return true;
}

static int edit_distance(const std::vector<int32_t>& a,
                         const std::vector<int32_t>& b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = (int)j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = (int)i;
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        prev.swap(cur);
    }
    return prev.back();
}

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    const auto process_start = Clock::now();
    const char* model = argc > 1 ? argv[1] : std::getenv("PARAKEET_TEST_GGUF_06B");
    const char* wav = argc > 2 ? argv[2] : std::getenv("PARAKEET_TEST_AUDIO");
    if (!model || !wav) return 77;

    pk::ModelLoader ml;
    if (!ml.load(model)) return 1;
    const auto model_loaded = Clock::now();
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) return 1;
    pk::MelFrontend frontend(ml);
    std::vector<float> mel;
    int n_mels = 0, Tmel = 0;
    frontend.compute(audio.samples, mel, n_mels, Tmel);
    const auto frontend_done = Clock::now();

    // 10 s left, 2 s center, 2 s right at 100 mel frames/s. Multiples
    // of the encoder's x8 stride keep every center boundary frame-aligned.
    const int left_context = 1000;
    const int center = 200;
    const int right_context = 200;
    const int valid_mel = Tmel - 1; // final frontend frame is center-pad.

    // NeMo's buffered script runs the preprocessor on each [left|chunk|right]
    // AUDIO buffer, so per_feature normalization uses that window's statistics
    // rather than the whole utterance's. Slicing a globally-normalized mel is a
    // different computation, and on a 7.4s fixture the two happen to agree
    // because every window starts at sample 0. PARAKEET_WINDOW_NORM=1 selects
    // NeMo's scheme so both can be measured on one build.
    const bool window_norm = std::getenv("PARAKEET_WINDOW_NORM") != nullptr;
    const int hop = (int)ml.config().hop_length;

    pk::Encoder encoder(ml);
    pk::PredictionNet pred(ml);
    pk::Joint joint(ml);
    pk::TdtDecodeState state = pk::tdt_decode_init(pred);
    const auto& cfg = ml.config();

    int chunks = 0;
    int kept_frames = 0;
    double first_chunk_ms = 0.0;
    const auto inference_start = Clock::now();
    for (int pos = 0; pos < valid_mel; pos += center) {
        const int commit_hi = std::min(valid_mel, pos + center);
        const int lo = std::max(0, pos - left_context);
        const int hi = std::min(valid_mel, commit_hi + right_context);
        const int window_valid = hi - lo;
        std::vector<float> window;
        int window_T = 0;
        if (window_norm) {
            // Run the frontend on this window's audio, exactly as NeMo does.
            // compute() supplies its own centre padding, so the trailing
            // pad-frame contract falls out rather than being reconstructed.
            const size_t lo_sample = (size_t)lo * hop;
            const size_t hi_sample = std::min(audio.samples.size(), (size_t)hi * hop);
            std::vector<float> window_audio(audio.samples.begin() + lo_sample,
                                            audio.samples.begin() + hi_sample);
            int wn_mels = 0;
            frontend.compute(window_audio, window, wn_mels, window_T);
            if (wn_mels != n_mels) return 1;
        } else {
            // Slice the globally normalized mel, preserving the preprocessor's
            // one trailing pad frame contract.
            window_T = window_valid + 1;
            window.assign((size_t)n_mels * window_T, 0.0f);
            for (int m = 0; m < n_mels; ++m) {
                for (int t = 0; t < window_valid; ++t)
                    window[(size_t)m * window_T + t] =
                        mel[(size_t)m * Tmel + lo + t];
                window[(size_t)m * window_T + window_valid] =
                    mel[(size_t)m * Tmel + std::min(Tmel - 1, hi)];
            }
        }

        std::vector<float> enc_cf;
        int d_model = 0, Tw = 0;
        encoder.forward(window, n_mels, window_T, enc_cf, d_model, Tw);
        const int first = ceil_div8(pos - lo);
        const int last = std::min(Tw, ceil_div8(commit_hi - lo));
        if (last < first) return 1;
        std::vector<float> frames((size_t)(last - first) * d_model);
        for (int t = first; t < last; ++t)
            for (int c = 0; c < d_model; ++c)
                frames[(size_t)(t - first) * d_model + c] =
                    enc_cf[(size_t)c * Tw + t];
        pk::tdt_decode_frames(pred, joint, frames, last - first, d_model,
            cfg.tdt_durations, state, (int)cfg.blank_id,
            (int)cfg.max_symbols);
        kept_frames += last - first;
        ++chunks;
        if (chunks == 1) {
            first_chunk_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - inference_start).count();
        }
        const std::string text = pk::detokenize(ml.tokenizer_pieces(),
            pk::strip_special_tokens(ml.tokenizer_pieces(), state.hyp));
        std::fprintf(stderr, "chunk %d mel=[%d,%d) keep=[%d,%d) text=%s\n",
                     chunks, lo, hi, first, last, text.c_str());
    }
    const std::string text = pk::detokenize(ml.tokenizer_pieces(),
        pk::strip_special_tokens(ml.tokenizer_pieces(), state.hyp));
    const auto inference_done = Clock::now();
    std::printf("%s\n", text.c_str());
    std::fprintf(stderr, "chunks=%d kept_encoder_frames=%d tokens=%zu\n",
                 chunks, kept_frames, state.hyp.size());
    const char* baseline = argc > 3 ? argv[3] : std::getenv("PARAKEET_TEST_NEMO_BASELINE");
    if (baseline) {
        std::vector<int32_t> reference;
        if (!read_reference_tokens(baseline, reference)) {
            std::fprintf(stderr, "failed to read NeMo token baseline: %s\n", baseline);
            return 1;
        }
        const int edits = edit_distance(reference, state.hyp);
        std::fprintf(stderr,
            "nemo_tokens=%zu chunked_tokens=%zu token_edits=%d token_error_rate=%.6f\n",
            reference.size(), state.hyp.size(), edits,
            reference.empty() ? 0.0 : (double)edits / reference.size());
        std::fprintf(stderr, "chunked_token_ids=");
        for (int32_t id : state.hyp) std::fprintf(stderr, "%d,", id);
        std::fprintf(stderr, "\n");
        if (edits != 0) return 1;
    }
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    const double peak_rss_mib = usage.ru_maxrss / (1024.0 * 1024.0);
#else
    const double peak_rss_mib = usage.ru_maxrss / 1024.0;
#endif
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
        "metrics_json={\"model_load_ms\":%.3f,\"frontend_ms\":%.3f,"
        "\"first_chunk_compute_ms\":%.3f,\"inference_ms\":%.3f,"
        "\"total_ms\":%.3f,\"peak_rss_mib\":%.3f,\"audio_seconds\":%.3f,"
        "\"chunks\":%d,\"tokens\":%zu}\n",
        ms(process_start, model_loaded), ms(model_loaded, frontend_done),
        first_chunk_ms, ms(inference_start, inference_done),
        ms(process_start, inference_done), peak_rss_mib,
        audio.samples.size() / 16000.0, chunks, state.hyp.size());
    return 0;
}
