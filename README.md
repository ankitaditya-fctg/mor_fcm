[![CI](https://github.com/mixture-of-recursions/mixture_of_recursions/actions/workflows/ci.yml/badge.svg)](https://github.com/mixture-of-recursions/mixture_of_recursions/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/mixture-of-recursions/mixture_of_recursions/releases/tag/v0.1.0)

# Mixture-of-Recursions (MoR)

```
Token stream ──► Router ──► depth-wise batches ──► Shared recursion block (×R) ──► LM head
                 │              │                          │
                 └─► early exit └─► KV manager ────────────┘
```

Mixture-of-Recursions (MoR) extends Transformer language models with a recursive computation graph where a single “recursion block” is reused multiple times. Tokens decide how deep to go, so easy contexts exit early while hard tokens request more capacity. The shared block keeps parameters compact, yet depth-wise batching and caching ensure the compute path stays efficient. This repository provides a concise, CPU-friendly PyTorch implementation with modular components for research and prototyping.

At each recursion step a lightweight router scores the active tokens. With **expert-choice routing** we iteratively keep the top-`p%` tokens and let the remainder exit. With **token-choice routing** every token picks its full depth once, then only those scheduled for deeper steps remain active. Two key-value cache strategies are included: recursion-wise caching stores a dedicated slot per step, while the share-first strategy reuses the first step’s cache to lower prefill latency. Both integrate with depth-wise batching to avoid wasted computation on inactive tokens. Training adds entropy regularization and optional depth targets to stabilise the router. We also ship an inference helper with greedy decoding and introspection hooks.

If you prefer an interactive tour, open [`notebooks/mor_quickstart.ipynb`](notebooks/mor_quickstart.ipynb) for a runnable walkthrough covering configuration, a few toy training iterations, and greedy decoding with per-token depth annotations.

## Quickstart

```
pip install -e .[dev]
pytest -q
```

### Train and sample

```
python mor_demo.py --router token_choice --kv_mode share_first --R 3 --target_depth 2.0
python mor_demo.py --router expert_choice --keep_ratio 0.5 --kv_mode recursion --R 4
python mor_demo.py --sharing middle_cycle --R 4 --router expert_choice --keep_ratio 0.6
```

### Quick Benchmarks

Use the lightweight `bench_infer.py` script to measure per-token throughput across router and KV cache variants.

```
python bench_infer.py \
  --router expert_choice --keep_ratio 0.5 \
  --kv_mode recursion --R 4 \
  --seq_len 1024 --batch_sizes 1,4,8 \
  --num_warmup 5 --num_iters 20 \
  --report_csv runs/bench.csv
```

Example output:

```
router=expert_choice kv_mode=recursion R=4 device=cpu tokens=64
  batch   median tok/s   p50 ms/token
      1          87.42           11.44
```

### Router Entropy and Depth Histograms

Routers now expose a temperature-scaled entropy penalty to encourage exploration. Enable it during training with the `--router_entropy_weight` flag and log per-step depth distributions with `--log_depth_hist_every`. Histograms can be persisted to CSV for later analysis and plotting.

```
python mor_demo.py \
  --router token_choice --R 4 --router_temperature 1.2 \
  --router_entropy_weight 0.02 \
  --log_depth_hist_every 100 \
  --depth_hist_path runs/depth_hist.csv \
  --deterministic
```

The helper script can convert histogram CSVs into compact figures:

```
python scripts/plot_depth_hist.py runs/depth_hist.csv runs/depth_hist.png
```

We avoid committing binary assets to the repository; generate plots locally as needed.

### Hugging Face Speed Decode

The repository also ships a standalone, weight-preserving speed adapter for Hugging Face causal LMs. It performs per-token depth routing without editing base model parameters and works with Llama, Mistral, and Qwen2/Qwen3 checkpoints on CPU or GPU. Use the `speed_decode.py` CLI to benchmark or inspect depth traces against production-scale models:

```
python speed_decode.py \
  --model meta-llama/Llama-3-8B-Instruct \
  --router expert_choice --R 4 --keep_ratio 0.6 \
  --max_new_tokens 64 --prompt "Explain mixture of recursions simply."
```

Set `--no_skip` (or `--router expert_choice --keep_ratio 1.0`) to measure parity with full-depth decoding, and enable `--return_depth_trace` to log how far each token traversed. The CLI prints throughput metrics and a depth histogram by default, and can optionally persist results with `--csv runs/depth_hist.csv`.

### Parameter Sharing

Recursive blocks now support configurable weight sharing. `--sharing none` keeps per-depth parameters, `--sharing cycle` reuses a fixed number of shards cyclically, and `--sharing middle_cycle` prioritises sharing around the centre of the recursion stack while keeping the outer layers distinct. Combine sharing with either router for rapid ablations.

## Design choices (from paper)

* Token-level recursion enables adaptive depth without increasing parameter count.
* Expert-choice and token-choice routers provide complementary trade-offs between adaptivity and stability.
* Recursion-wise caching and the share-first variant reduce KV movement while respecting per-token activity.
