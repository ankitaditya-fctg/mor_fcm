# MOR Speed Adapter

The MOR speed adapter exposes a **depth-adaptive decoding** wrapper around
HuggingFace causal language models.  The wrapper does not modify model weights;
instead it runs a lightweight router that decides how many transformer blocks a
token should traverse at each generation step.

Two routing strategies are available:

* `token_choice` – every token selects a depth `d in [0, R]` for the current
  step.  Tokens that request shallow depths exit the transformer early and skip
  additional layers.
* `expert_choice` – a global `keep_ratio` defines how many tokens continue
deeper at each layer.  Only the hardest tokens (according to the router logits)
continue to the next layer; the remainder stop early.

Skipping is executed via **depth-wise batching**: we process the first
`base_layers` layers for the whole batch, compute router decisions, then peel
off tokens that have finished while the rest of the batch continues deeper.  No
dummy matrix multiplications are performed – we only execute layers for tokens
that need them.

The wrapper works for Llama/Mistral/Qwen2/Qwen3 architectures on CPU or GPU.
Setting `keep_ratio=1.0` (or using `--no_skip`) falls back to the baseline traversal and
reproduces the original outputs.

## Quickstart

```bash
pip install transformers accelerate torch --index-url https://download.pytorch.org/whl/cu121
python speed_decode.py --model meta-llama/Llama-3-8B-Instruct \
   --router expert_choice --R 4 --keep_ratio 0.6 \
   --max_new_tokens 64 --prompt "Explain mixture of recursions simply."
```

For more aggressive skipping try:

```bash
python speed_decode.py --model meta-llama/Llama-3-8B-Instruct \
   --router token_choice --R 6 --base_layers 2
```

## Quality parity

Skipping layers may change outputs.  Use `--no_skip` for full-depth traversal or
set `keep_ratio=1.0` to compare against the baseline.

## Metrics

The CLI prints tokens/second, average latency, and optional depth histograms.
Pass `--return_depth_trace` to inspect per-token depths and `--csv` to store the
histogram for later analysis.
