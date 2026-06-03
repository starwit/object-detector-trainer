from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from object_detector_trainer.backends import rtmdet as core_rtmdet
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
            config_name="rtmdet_m_8xb32-300e_coco",
            cache_dir=cache_dir,
        )


def test_rtmdet_baseline_uses_portable_adjacent_config_before_cache_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"stub-weights")
    config_path = tmp_path / "model_config.py"
    config_path.write_text("# exported baseline config\n", encoding="utf-8")

    init_calls: dict[str, str] = {}

    fake_mmcv = types.ModuleType("mmcv")
    fake_mmcv_ext = types.ModuleType("mmcv._ext")
    monkeypatch.setitem(sys.modules, "mmcv", fake_mmcv)
    monkeypatch.setitem(sys.modules, "mmcv._ext", fake_mmcv_ext)

    fake_apis = types.ModuleType("mmdet.apis")

    def fake_init_detector(config: str, weights: str, device: str = "cpu"):
        init_calls["config"] = str(config)
        init_calls["weights"] = str(weights)
        init_calls["device"] = str(device)
        return object()

    fake_apis.init_detector = fake_init_detector
    fake_mmdet = types.ModuleType("mmdet")
    fake_mmdet.apis = fake_apis
    monkeypatch.setitem(sys.modules, "mmdet", fake_mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", fake_apis)

    fake_wrapper_module = types.ModuleType("object_detector_trainer.wrappers.rtmdet")

    class FakeAdapter:
        def __init__(
            self,
            detector,
            *,
            model_name: str,
            resolution: int,
            class_names: dict[int, str],
            model_variant: str | None,
            model_config_path: str,
        ) -> None:
            init_calls["adapter_model_name"] = str(model_name)
            init_calls["adapter_config_path"] = str(model_config_path)
            self.detector = detector

    fake_wrapper_module.RTMDetModelAdapter = FakeAdapter
    monkeypatch.setitem(sys.modules, "object_detector_trainer.wrappers.rtmdet", fake_wrapper_module)
    monkeypatch.setattr(core_rtmdet.torch.cuda, "is_available", lambda: False)

    core_rtmdet.load_rtmdet_baseline(
        weights_path=weights_path,
        metadata={
            "model_backend": "rtmdet",
            "model_config_path": "model_config.py",
            "rtmdet_config_name": "rtmdet_tiny_8xb32-300e_coco",
            "rtmdet_cache_dir": str(tmp_path / "missing-cache"),
            "image_size": 320,
        },
        display_name="portable-baseline",
    )

    assert init_calls["config"] == str(config_path)
    assert init_calls["weights"] == str(weights_path)
    assert init_calls["adapter_model_name"] == "portable-baseline"
    assert init_calls["adapter_config_path"] == str(config_path)


def test_rtmdet_baseline_fails_when_explicit_config_path_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"stub-weights")
    (tmp_path / "model_config.py").write_text("# adjacent config\n", encoding="utf-8")

    fake_mmcv = types.ModuleType("mmcv")
    fake_mmcv_ext = types.ModuleType("mmcv._ext")
    monkeypatch.setitem(sys.modules, "mmcv", fake_mmcv)
    monkeypatch.setitem(sys.modules, "mmcv._ext", fake_mmcv_ext)

    with pytest.raises(FileNotFoundError, match="model_config_path"):
        core_rtmdet.load_rtmdet_baseline(
            weights_path=weights_path,
            metadata={
                "model_backend": "rtmdet",
                "model_config_path": "missing_config.py",
                "rtmdet_config_name": "rtmdet_tiny_8xb32-300e_coco",
                "rtmdet_cache_dir": str(tmp_path),
                "image_size": 320,
            },
            display_name="broken-baseline",
        )
