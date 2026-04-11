from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.config import SamplingConfig


def test_default_sampling_config() -> None:
    cfg = SamplingConfig()
    assert cfg.k == 3
    assert cfg.d == 8
