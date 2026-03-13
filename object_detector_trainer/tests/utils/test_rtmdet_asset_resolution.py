from __future__ import annotations

from pathlib import Path

import pytest

from object_detector_trainer.backends.rtmdet import _resolve_rtmdet_assets


def test_rtmdet_resolve_does_not_fallback_to_unrelated_checkpoint(tmp_path: Path) -> None:
    cache_dir = tmp_path / "rtmdet-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # An unrelated checkpoint exists in the shared cache (e.g. tiny),
    # but the requested variant (e.g. m) is missing. This must fail fast.
    (cache_dir / "rtmdet_tiny_8xb32-300e_coco_20220101.pth").write_bytes(b"tiny")
    (cache_dir / "rtmdet_m_8xb32-300e_coco.py").write_text("# cfg", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        _resolve_rtmdet_assets(
            config_path=None,
            checkpoint_path=None,
            config_name="rtmdet_m_8xb32-300e_coco",
            cache_dir=cache_dir,
        )
