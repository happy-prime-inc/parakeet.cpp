#include "audio_io.hpp"
#include "encoder.hpp"
#include "joint.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "tdt.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const char* model = std::getenv("PARAKEET_TEST_GGUF_06B");
    const char* wav = std::getenv("PARAKEET_TEST_AUDIO");
    if (!model || !wav) return 77;

    pk::ModelLoader ml;
    if (!ml.load(model)) return 1;
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) return 1;

    pk::MelFrontend mel(ml);
    std::vector<float> feats;
    int n_mels = 0, Tmel = 0;
    mel.compute(audio.samples, feats, n_mels, Tmel);
    pk::Encoder encoder(ml);
    std::vector<float> enc_cf;
    int d_model = 0, T = 0;
    encoder.forward(feats, n_mels, Tmel, enc_cf, d_model, T);
    std::vector<float> enc((size_t)T * d_model);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < d_model; ++c)
            enc[(size_t)t * d_model + c] = enc_cf[(size_t)c * T + t];

    pk::PredictionNet pred(ml);
    pk::Joint joint(ml);
    const auto& cfg = ml.config();
    const auto whole = pk::tdt_greedy(pred, joint, enc, T, d_model,
        cfg.tdt_durations, (int)cfg.blank_id, (int)cfg.max_symbols);

    // Deliberately awkward sizes exercise durations crossing boundaries.
    const int cuts[] = {1, 2, 7, 13, 31};
    for (int chunk : cuts) {
        pk::TdtDecodeState st = pk::tdt_decode_init(pred);
        for (int lo = 0; lo < T; lo += chunk) {
            const int hi = std::min(T, lo + chunk);
            std::vector<float> part(enc.begin() + (size_t)lo * d_model,
                                    enc.begin() + (size_t)hi * d_model);
            pk::tdt_decode_frames(pred, joint, part, hi - lo, d_model,
                cfg.tdt_durations, st, (int)cfg.blank_id,
                (int)cfg.max_symbols);
        }
        if (st.hyp != whole) {
            std::fprintf(stderr, "chunk=%d mismatch: whole=%zu split=%zu\n",
                         chunk, whole.size(), st.hyp.size());
            return 1;
        }
    }
    std::fprintf(stderr, "PASS: %zu TDT tokens identical across 5 chunkings\n",
                 whole.size());
    return 0;
}
