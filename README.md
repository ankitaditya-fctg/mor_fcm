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
```

## Design choices (from paper)

* Token-level recursion enables adaptive depth without increasing parameter count.
* Expert-choice and token-choice routers provide complementary trade-offs between adaptivity and stability.
* Recursion-wise caching and the share-first variant reduce KV movement while respecting per-token activity.
