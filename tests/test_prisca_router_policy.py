import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from szl_prisca_router.policy import propose_route


def test_trivial_stays_cheap():
    p = propose_route("trivial")
    assert p.mouth == "disk_ztfi"
    assert p.authority == "PROPOSAL_ONLY"
    assert p.evidence_state == "NOT_CLAIMED"


def test_multi_hop_floor():
    assert propose_route("multi_hop").mouth == "lean_mouth"


def test_prefer_expensive_override():
    assert propose_route("trivial", prefer_cheap=False).mouth == "frontier"
