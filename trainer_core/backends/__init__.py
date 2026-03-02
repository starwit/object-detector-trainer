from trainer_core.backends import rtmdet, rfdetr, yolo

# Backward-compatible import alias while older call sites still use `mmdet`.
mmdet = rtmdet

__all__ = ["rtmdet", "mmdet", "rfdetr", "yolo"]
