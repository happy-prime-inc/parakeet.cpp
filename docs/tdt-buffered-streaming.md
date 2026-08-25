# Buffered streaming for offline TDT models

Offline Parakeet TDT checkpoints (`parakeet-tdt-0.6b-v2`, `-v3`) were never
trained for cache-aware streaming: there is no per-layer conv/attention cache to
carry, so the encoder cannot be resumed from where it stopped. NVIDIA's answer,
which `pk::BufferedTdtStream` implements, is to re-encode a sliding window and
decode each chunk exactly once.

This is available through the ordinary streaming C API — `parakeet_capi_stream_*`
selects it automatically for these models — and through `pk::BufferedTdtStream`
directly.

## Algorithm

Per chunk:

1. Run the offline encoder over `[10 s left | 2 s chunk | 2 s right]`.
2. Keep only the encoder frames belonging to the chunk.
3. Decode those frames **exactly once**, carrying prediction-network state, the
   last emitted token, and any TDT duration skip into the next chunk.

Because no audio is decoded twice, the transcript is append-only by
construction: there is no second reading of a span to disagree with the first.

`TdtDecodeState::pending_skip` is what makes step 3 safe. TDT emits a token plus
a duration, which may advance several encoder frames; when that skip crosses a
chunk boundary the frames it consumed belong to the *next* chunk, and decoding
that chunk from frame zero would re-consume them.

**Normalization follows NeMo's buffered script**: the frontend runs on each
window's *audio*, so `per_feature` statistics are that window's. Slicing a
globally normalized mel is a different computation — it agrees on short clips,
where every window starts at sample 0, and diverges on long ones — and it cannot
run live at all, since whole-utterance statistics need the whole utterance.

**Every span is rounded to whole encoder frames** (`hop_length × subsampling_factor`),
by truncation, matching NeMo's `int(secs * features_per_sec / subsampling_factor)`.
A schedule whose spans are not frame-aligned puts chunk boundaries between
encoder frames, and the resulting overlap/skip deletes words: a 10/1/1 schedule
dropped 162 of 369 reference words before this rounding existed.

## Previews

Committing a word requires its chunk to fill and its right context to arrive, so
a committed word appears roughly `chunk + right` after it is spoken. Two
previews, both decoding from a **copy** of the decoder state that is then
discarded, make the newest words visible sooner without touching what is
committed:

- **Tail preview** — the right-context region, whose encoder frames are already
  computed and otherwise discarded. Costs one decoder pass, no encode.
- **Opening preview** — before a turn's first commit, when nothing can be
  committed yet. This one needs its own encoder pass, and is affordable only
  because it runs where it is cheap: before the first commit there is no left
  context, so the window is at most `chunk + right`. It stops firing once the
  turn is flowing.

Measured on a 29.2 s capture: first text at a turn's start 4.0 s → 0.4 s, and
86% of preview text survives settling unchanged (character-level against the
settled transcript). Preview text **is** revisable and belongs in a caller's
tentative tier; `text` and `words` remain append-only.

Speculation is **off by default** — a caller that cannot read `tentative_text()`
would be paying for output it can never observe. The streaming JSON entry points
enable it, since they are the only way to observe it.

## Parity

Token-exact against NeMo's
`examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`
at commit `e0a284b67db4da54f90a7873dd723aa2106a503e`, CPU float32,
`decoding.strategy=greedy_batch`, `max_symbols_per_step=10`:

| Model | Schedule | NeMo tokens | C++ tokens | Token edits |
| --- | --- | ---: | ---: | ---: |
| TDT v2 | 10/2/2 | 50 | 50 | 0 |
| TDT v3 | 10/2/2 | 44 | 44 | 0 |
| TDT v2 | 6/1/1 | — | — | text-exact |

On a 200 s human-referenced reading, v2 at 10/2/2 matches NeMo's own buffered
implementation at **WER 0.0596** with an identical sub/del/ins breakdown, and
differs from NeMo's output by 3 word substitutions in 373.

## The limit this method has

**Shortening right context deletes words rather than delaying them.** Each
chunk's frames are decoded once, so when the model declines to commit a token
for want of lookahead, nothing revisits those frames. On the 200 s reading with
v2:

| Right context | WER | Deletions |
| --- | ---: | ---: |
| 2 s | 0.0596 | 2 |
| 1 s | 0.2547 | 76 |

NeMo's own implementation behaves the same way on the same audio (right 1 s:
WER 0.1951, 55 deletions), so this is inherent to buffered TDT rather than to
this port. Left context, by contrast, is nearly free: 6/2/1 against 10/2/1
improves WER 0.2547 → 0.1220 *and* cuts RTF 0.57 → 0.35.

## Tests

| Test | Asserts |
| --- | --- |
| `test_tdt_decode_chunking` | decoder state identical across 1/2/7/13/31-frame splits |
| `test_tdt_context_chunks` | token parity vs NeMo; doubles as a benchmark harness |
| `test_buffered_pcm` | block-size independence (7…32000 samples), NeMo parity, and that no encoder frame is left undecoded at end of stream |
| `test_buffered_speculative` | committed text byte-identical with speculation on and off; reports preview survival |
| `test_capi_buffered_stream` | the C API path matches the C++ class; previews survive `stream_reset` |

All take their model and audio from the environment and skip (77) without them:
`PARAKEET_TEST_GGUF_06B`, `PARAKEET_TEST_AUDIO`, and optionally
`PARAKEET_TEST_NEMO_BASELINE`.

`PARAKEET_STREAM_SCHEDULE=left,chunk,right` overrides the schedule for
latency/accuracy sweeps; `PARAKEET_TEST_ONESHOT_ONLY=1` skips the block-size
loop when only the transcript is wanted.

## Not covered

- Beam search. Greedy only — and TDT N-best fails above a content-dependent
  40–90 s at every beam width, which is why greedy is not a limitation here.
- Timestamps are per-word via `drain_words()`; token-level timing is not exposed.
- The 10/2/2 schedule is NVIDIA's inference-time recommendation, not model
  metadata — there is no GGUF key for it — so it stays a construction parameter.
- The encoder recomputes its left context every chunk. This is NVIDIA's buffered
  algorithm and runs faster than real time here; it is not a cache-aware encoder.
