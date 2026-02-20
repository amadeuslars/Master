"""
Unified action set for ALNS algorithms.

30 composite actions = 3 destroy ops × 5 removal sizes × 2 repair ops.

Action ordering:
    idx = d_idx * 10 + s_idx * 2 + r_idx

Decoding:
    r_idx = idx % 2
    s_idx = (idx // 2) % 5
    d_idx = idx // 10
"""

import random
from utils.operators2 import (
    random_removal, worst_removal, cluster_removal,
    greedy_insertion, regret_insertion,
)

REMOVAL_SIZES = [
    ('xs', 2, 5),
    ('sm', 5, 10),
    ('md', 10, 20),
    ('lg', 20, 30),
    ('xl', 30, 40),
]

DESTROY_OPS = [
    ('random', random_removal),
    ('worst', worst_removal),
    ('cluster', cluster_removal),
]

REPAIR_OPS = [
    ('greedy', greedy_insertion),
    ('regret', regret_insertion),
]

NUM_ACTIONS = len(DESTROY_OPS) * len(REMOVAL_SIZES) * len(REPAIR_OPS)  # 30


def build_actions():
    """
    Build 30 composite actions.

    Returns list of (sized_destroy_fn, repair_fn, label) tuples.
    The sized destroy closure draws n_remove uniformly from its fixed range.
    """
    actions = []
    for d_name, d_op in DESTROY_OPS:
        for s_name, lo, hi in REMOVAL_SIZES:
            for r_name, r_op in REPAIR_OPS:
                def sized_destroy(solution, _lo=lo, _hi=hi, _op=d_op, **kwargs):
                    n_remove = random.randint(_lo, _hi)
                    return _op(solution, n_remove, **kwargs)

                label = f"{d_name}_{s_name}_{r_name}"
                actions.append((sized_destroy, r_op, label))

    return actions


def decode_action(idx):
    """Decode a composite action index into (d_idx, s_idx, r_idx)."""
    r_idx = idx % 2
    s_idx = (idx // 2) % 5
    d_idx = idx // 10
    return d_idx, s_idx, r_idx
