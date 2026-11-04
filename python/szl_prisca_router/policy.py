"""Hierarchical cost-aware route proposal.

Levels (Soft Shield ladder, cheapest first):
  disk_ztfi → chaskallm_cpu → lean_mouth → workbench → frontier

Curriculum cost: prefer a cheaper mouth unless difficulty signals require ascent.
Returns a structured proposal with evidence state NOT_CLAIMED until paired with
a retrieval-bench MEASURED receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Mouth = Literal["disk_ztfi", "chaskallm_cpu", "lean_mouth", "workbench", "frontier"]
Difficulty = Literal["trivial", "local", "multi_hop", "adversarial"]

LADDER: tuple[Mouth, ...] = (
    "disk_ztfi",
    "chaskallm_cpu",
    "lean_mouth",
    "workbench",
    "frontier",
)

DIFFICULTY_FLOOR: dict[Difficulty, Mouth] = {
    "trivial": "disk_ztfi",
    "local": "chaskallm_cpu",
    "multi_hop": "lean_mouth",
    "adversarial": "workbench",
}


@dataclass(frozen=True)
class RouteProposal:
    mouth: Mouth
    difficulty: Difficulty
    reason: str
    evidence_state: str = "NOT_CLAIMED"
    authority: str = "PROPOSAL_ONLY"
    graph_rag_pool: str = "prisca_rag_corpus_v1"

    def to_dict(self) -> dict:
        return asdict(self)


def propose_route(difficulty: Difficulty, *, prefer_cheap: bool = True) -> RouteProposal:
    """Propose the cheapest mouth at or above the difficulty floor."""
    floor = DIFFICULTY_FLOOR[difficulty]
    idx = LADDER.index(floor)
    mouth = LADDER[idx] if prefer_cheap else LADDER[-1]
    return RouteProposal(
        mouth=mouth,
        difficulty=difficulty,
        reason=f"floor={floor}; prefer_cheap={prefer_cheap}; ladder={list(LADDER)}",
    )
