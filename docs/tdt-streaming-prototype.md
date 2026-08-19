# Buffered streaming TDT feasibility prototype

This branch proves that the offline Parakeet TDT v2/v3 models can use NVIDIA's
left-context + chunk + right-context inference design without re-decoding or
merging text. It is intentionally a bounded native prototype, not a public API
or production real-time audio implementation.

## Algorithm

The executable computes the full mel spectrogram once and processes it with the
10-2-2 schedule recommended by NVIDIA:

1. Run the offline encoder over `[10 s left | 2 s chunk | 2 s right]`.
2. Strip the left and right encoder outputs, retaining the 25 frames belonging
   to the 2-second chunk (the model subsamples by 8).
3. Decode those retained frames exactly once.
4. Carry prediction-network LSTM state, last emitted token, accumulated token
   IDs, and any TDT duration skip crossing the boundary into the next chunk.

`TdtDecodeState::pending_skip` is essential. TDT can advance by several encoder
frames at an emission; if that duration crosses a chunk seam, decoding the next
chunk from frame zero duplicates already-consumed frames.

## Executable

Build:

```bash
cmake -S . -B build -DPARAKEET_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_tdt_context_chunks test_tdt_decode_chunking -j
```

Run v3 with the checked-in NVIDIA streaming token fixture:

```bash
build/tests/test_tdt_context_chunks \
  /path/to/parakeet-tdt-0.6b-v3.gguf \
  tests/fixtures/speech.wav \
  tests/fixtures/nemo_stream_tdt_v3_10_2_2.tokens
```

The same command accepts the v2 model and
`nemo_stream_tdt_v2_10_2_2.tokens`. The transcript is written to stdout;
incremental chunks, token parity, and a `metrics_json=...` record go to stderr.
The executable returns nonzero if the supplied reference token sequence differs.

## Reference and parity evidence

The token fixtures were captured from NVIDIA NeMo main commit
`e0a284b67db4da54f90a7873dd723aa2106a503e` using
`examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`
with CPU float32 inference and:

```text
chunk_secs=2.0
left_context_secs=10.0
right_context_secs=2.0
decoding.strategy=greedy_batch
decoding.greedy.max_symbols_per_step=10
```

On `tests/fixtures/speech.wav`:

| Model | NeMo tokens | C++ tokens | Token edits |
| --- | ---: | ---: | ---: |
| TDT v2 | 50 | 50 | 0 |
| TDT v3 | 44 | 44 | 0 |

The independent decoder-state test encodes a complete 29.2-second QA capture
once, then splits its encoder frames at sizes 1, 2, 7, 13, and 31. Every split
produces the same 139 v3 tokens as one-shot decoding. These awkward boundaries
exercise cross-chunk duration skips directly.

## Whole-file QA and performance

Fixture: `20260812T145159_001.wav`, 29.2 seconds, 15 chunks. Measurements are a
single warm local Release build on Apple Silicon CPU; model loading is included
in total time but excluded from inference time.

| Model | Inference | RTF | First chunk compute | Peak RSS | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| TDT v2 | 14.047 s | 0.481 | 325 ms | 736 MiB | 166 |
| TDT v3 | 14.152 s | 0.485 | 345 ms | 792 MiB | 140 |

Both runs were append-only across the full file: earlier tokens were never
changed or removed. Relative to the saved 69-word transcript, v2 had one
normalized word substitution (`a` -> `her`, about 1.45% WER); v3 had two
(`Dr.` -> `Doctor`, `a` -> `her`, about 2.90% WER). The latency visible to a
live user is still approximately chunk + right context = 4 seconds; the compute
time does not add another 4 seconds because it runs faster than real time.

## Deliberate prototype limitations

- The 100 mel frames/s and x8 subsampling conversion is fixed for these v2/v3
  checkpoints. A production API must derive and validate both values from model
  metadata.
- The full mel spectrogram is computed up front. A real PCM feeder needs to
  preserve the same frontend framing at chunk boundaries.
- Encoder left context is recomputed every chunk. This matches NVIDIA's buffered
  algorithm and is already faster than real time here, but it is not a
  cache-aware encoder.
- Only greedy TDT is implemented. Beam search, timestamps, language selection,
  C API ownership, reset/finalize semantics, and app wiring remain out of scope.
- The test runner lives under `tests/`; productionization should move the
  orchestration into a library session and expose it through an additive C API.

## Suggested refinement order

1. Extract the window loop into `BufferedTdtSession` with model-derived timing.
2. Add incremental PCM/mel boundary tests against the current full-mel oracle.
3. Expose begin/feed/finalize calls through the C API without changing the
   existing cache-aware RNNT streaming API.
4. Add v2/v3 token-parity CI fixtures and whole-file latency/RSS thresholds.
5. Wire the application only after the native session has reset, finalize, and
   cancellation tests.
