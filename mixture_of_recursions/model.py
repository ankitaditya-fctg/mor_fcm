"""Core Mixture-of-Recursions model implementation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .config import ModelConfig, RouterConfig, KVConfig
from .modules.embeddings import TokenPositionalEmbedding
from .recursion.recursion_block import SharedRecursionBlock
from .recursion.routers import build_router, RoutingResult
from .recursion.kv_cache import KVCacheManager


class MoRModel(nn.Module):
    """Mixture-of-Recursions language model."""

    def __init__(self, model_config: ModelConfig, router_config: RouterConfig, kv_config: Optional[KVConfig] = None) -> None:
        super().__init__()
        self.config = model_config
        self.router_config = router_config
        self.max_recursions = model_config.max_recursions
        self.embedding = TokenPositionalEmbedding(model_config.vocab_size, model_config.d_model)
        self.dropout = nn.Dropout(model_config.dropout)
        self.recursion_block = SharedRecursionBlock(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            d_ff=model_config.d_ff,
            n_layers_shared=model_config.n_layers_shared,
            dropout=model_config.dropout,
            rotary=model_config.rotary,
        )
        router, meta = build_router(router_config, model_config.d_model, model_config.max_recursions)
        self.router = router
        self.router_meta = meta
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embed.weight  # type: ignore[attr-defined]
        self.kv_manager = KVCacheManager((kv_config or KVConfig()).mode, model_config.max_recursions)
        self.final_norm = nn.LayerNorm(model_config.d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        device = input_ids.device
        batch, seq_len = input_ids.shape

        hidden_states = self.embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        attn_mask = self._causal_mask(seq_len, device)
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask[:, None, None, :]

        router_loss = torch.tensor(0.0, device=device)
        depth_map = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        routing_result: Optional[RoutingResult] = None

        if self.router_config.type == "token_choice":
            routing_result = self.router.assign_depths(hidden_states.detach())
            depth_map = routing_result.depth_map
            router_loss = router_loss + self.router.loss().to(device)
        else:
            depth_map.fill_(0)

        current_active = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        self.kv_manager.reset()

        residual_states = hidden_states
        for step in range(self.max_recursions):
            if self.router_config.type == "token_choice":
                step_mask = depth_map >= (step + 1)
            else:
                step_mask = current_active

            if not step_mask.any():
                continue

            kv_override = self.kv_manager.get(step)
            updated, kv = self.recursion_block(residual_states, attn_mask=attn_mask, kv_override=kv_override)
            updated = torch.where(step_mask.unsqueeze(-1), updated, residual_states)
            residual_states = updated
            if kv is not None:
                self.kv_manager.update(step, kv)

            if self.router_config.type == "expert_choice":
                depth_map = torch.where(step_mask, torch.full_like(depth_map, step + 1), depth_map)
                next_active = self.router.forward_step(residual_states.detach(), step_mask)
                router_loss = router_loss + self.router.loss().to(device)
                current_active = next_active
            else:
                depth_map = depth_map

        if self.router_config.type == "expert_choice":
            depth_map = torch.where(depth_map == 0, torch.ones_like(depth_map), depth_map)

        if self.router_config.target_depth is not None and self.router_config.depth_penalty > 0:
            avg_depth = depth_map.float().mean()
            target = torch.tensor(self.router_config.target_depth, device=device)
            router_loss = router_loss + self.router_config.depth_penalty * (avg_depth - target) ** 2

        hidden_states = self.final_norm(residual_states)
        logits = self.lm_head(hidden_states)
        output: Dict[str, Tensor] = {
            "logits": logits,
            "depth_map": depth_map,
            "router_loss": router_loss,
        }
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            output["loss"] = loss + router_loss
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
    ) -> Tuple[Tensor, List[int]]:
        """Greedy decoding with router decisions per generated token."""

        generated = input_ids
        token_depths: List[int] = []
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            depth = int(outputs["depth_map"][0, -1].item())
            token_depths.append(depth)
        return generated, token_depths
