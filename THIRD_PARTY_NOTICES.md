# Third-party notices and scientific provenance

This repository is a research adaptation, not the official StreamForest trained model.

- **HERMES** — https://github.com/haowei-freesky/HERMES — MIT, Copyright (c) 2026 Haowei Zhang. `eventkv/qwen_model.py` extracts native Qwen cache-position bookkeeping and answer decoding from `inference/qwenvl_hermes_fa2.py`; HERMES-specific pruning, pseudo-query generation and budget policy are excluded. The upstream notice is retained in `licenses/HERMES.txt`.
- **OVO-Bench** — https://github.com/JoeLeelyf/OVO-Bench — MIT, Copyright (c) 2024 Junbo Niu. Official task prompts, task/annotation conventions and scoring protocol are retained. Local sampling/reply utilities originate from the existing OVO integration. See `licenses/OVO-Bench.txt`.
- **StreamForest** — https://github.com/MCG-NJU/StreamForest — Apache-2.0. The event-vector memory adapts bipartite matching / weighted merging, short-window boundary selection, adjacent-event cost management, and sampled-frame time weighting to variable-size Qwen2.5-VL features. See `licenses/StreamForest.txt`. Differences include independent persistent event KV, N/A/D scoring, text compensation, and the latest pre-action retirement gate and event-only writer.
- **Token Merging (ToMe)** — https://github.com/facebookresearch/ToMe — the bipartite merging idea is also the basis of StreamForest merging. This release retains the local EventKV implementation; it does not vendor the ToMe package.
- **Transformers / Qwen2.5-VL** — https://github.com/huggingface/transformers and https://github.com/QwenLM/Qwen2.5-VL — model projections, rotary position encoding, attention, cache and visual tower are supplied by installed Transformers. Model weights are downloaded separately and remain subject to their own license.
- **StreamingBench** — https://github.com/THUNLP-MT/StreamingBench — official task names and prompt protocol are used by the benchmark reader. No dataset or video redistribution is included.

Shared post-RoPE query capture and N/D/A signal helpers are extracted from this project's existing inference implementation; all moment-summary attention/readout branches and other method selectors have been removed. Core source hashes before import adaptation are recorded in `docs/source_provenance.json`.

All upstream notices are preserved for the reused portions. No new blanket license is asserted over third-party material; consult the individual notices. Research results in `docs/VERSIONS.md` refer to pre-existing experiments, not a fresh full-benchmark run of this release.
