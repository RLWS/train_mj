from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mjai.serving.schemas import RankedAction


@dataclass(slots=True, frozen=True)
class ResponseSelection:
    raw_index: int
    selected_index: int
    raw_action: dict[str, Any]
    selected_action: dict[str, Any]
    pass_bias: float
    no_yaku_call_penalty: float


def rank_candidates(logits: torch.Tensor, legal_actions: list[dict[str, Any]], top_k: int) -> list[RankedAction]:
    if top_k <= 0:
        return []

    probs = torch.softmax(logits, dim=0)
    limit = min(int(top_k), int(logits.shape[0]))
    scores, indices = torch.topk(logits, k=limit)
    ranked: list[RankedAction] = []
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        ranked.append(
            RankedAction(
                rank=rank,
                index=int(index),
                action=dict(legal_actions[int(index)]),
                score=float(score),
                prob=float(probs[int(index)].item()),
            )
        )
    return ranked


def select_response_action(
    logits: torch.Tensor,
    legal_actions: list[dict[str, Any]],
    *,
    pass_bias: float,
    no_yaku_call_penalty: float = 0.0,
    candidate_features: torch.Tensor | None = None,
) -> ResponseSelection:
    raw_index = int(torch.argmax(logits).item())
    adjusted_logits = _apply_no_yaku_call_penalty(
        logits,
        legal_actions,
        no_yaku_call_penalty=float(no_yaku_call_penalty),
        candidate_features=candidate_features,
    )
    pass_indices = [index for index, action in enumerate(legal_actions) if _is_pass(action)]
    non_pass_indices = [index for index, action in enumerate(legal_actions) if not _is_pass(action)]

    if not pass_indices:
        selected_index = raw_index
    else:
        best_pass_index = max(pass_indices, key=lambda index: float(adjusted_logits[index].item()))
        if not non_pass_indices:
            selected_index = best_pass_index
        else:
            best_non_pass_index = max(
                non_pass_indices,
                key=lambda index: float(adjusted_logits[index].item()),
            )
            choose_non_pass = float(adjusted_logits[best_non_pass_index].item()) - float(
                adjusted_logits[best_pass_index].item()
            ) > float(pass_bias)
            selected_index = best_non_pass_index if choose_non_pass else best_pass_index

    return ResponseSelection(
        raw_index=raw_index,
        selected_index=selected_index,
        raw_action=dict(legal_actions[raw_index]),
        selected_action=dict(legal_actions[selected_index]),
        pass_bias=float(pass_bias),
        no_yaku_call_penalty=float(no_yaku_call_penalty),
    )


def _is_pass(action: dict[str, Any]) -> bool:
    return str(action.get("kind", "")) == "pass"


def _apply_no_yaku_call_penalty(
    logits: torch.Tensor,
    legal_actions: list[dict[str, Any]],
    *,
    no_yaku_call_penalty: float,
    candidate_features: torch.Tensor | None,
) -> torch.Tensor:
    if no_yaku_call_penalty <= 0.0 or candidate_features is None:
        return logits
    adjusted = logits.clone()
    for index, action in enumerate(legal_actions):
        if not _is_no_yaku_call_candidate(action, candidate_features[index]):
            continue
        adjusted[index] = adjusted[index] - float(no_yaku_call_penalty)
    return adjusted


def _is_no_yaku_call_candidate(action: dict[str, Any], candidate_features: torch.Tensor) -> bool:
    if str(action.get("kind", "")) not in {"chi", "pon", "daiminkan"}:
        return False
    if candidate_features.numel() <= 86:
        return False
    # RESPONSE_ACTION_DIM layout: 7 kind + 37 tile + 4 source + 43 call features.
    # call feature index 38 is "opened hand and no obvious yaku route".
    return float(candidate_features[48 + 38].item()) > 0.5
