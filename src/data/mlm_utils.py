"""Masked Event Modeling utilities."""

from __future__ import annotations

import torch

from src.data.vocab import Vocabulary

MLM_IGNORE_INDEX = -100


def _build_event_token_ids(vocab: Vocabulary) -> list[int]:
    special = {vocab.pad_id, vocab.cls_id, vocab.unk_id, vocab.mask_id}
    return [i for i in range(len(vocab)) if i not in special]


def apply_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab: Vocabulary,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style 80/10/10 masking to a single sequence.

    Returns:
        mlm_input_ids: masked input token ids
        mlm_labels: original token ids at masked positions, -100 elsewhere
    """
    mlm_input_ids = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, MLM_IGNORE_INDEX)

    valid = (attention_mask == 1) & (input_ids != vocab.pad_id)
    valid[0] = False  # skip CLS

    candidate_positions = valid.nonzero(as_tuple=False).squeeze(-1)
    if candidate_positions.numel() == 0:
        return mlm_input_ids, mlm_labels

    n_mask = max(1, int(round(mask_prob * candidate_positions.numel())))
    perm = torch.randperm(candidate_positions.numel(), generator=rng)
    selected = candidate_positions[perm[:n_mask]]

    event_token_ids = _build_event_token_ids(vocab)
    event_token_tensor = torch.tensor(event_token_ids, dtype=torch.long, device=input_ids.device)

    for pos in selected.tolist():
        original_id = int(input_ids[pos].item())
        mlm_labels[pos] = original_id

        draw = torch.rand(1, generator=rng).item()
        if draw < mask_token_prob:
            mlm_input_ids[pos] = vocab.mask_id
        elif draw < mask_token_prob + random_token_prob:
            rand_idx = int(torch.randint(len(event_token_ids), (1,), generator=rng).item())
            mlm_input_ids[pos] = event_token_tensor[rand_idx]
        # else: keep original token (10%)

    return mlm_input_ids, mlm_labels
