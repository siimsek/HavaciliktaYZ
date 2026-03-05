"""Competition payload schema helpers with canonical field guarantees."""

from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings
from src.competition_contract import DataContractError


class CompetitionPayloadSchema:
    CANONICAL_MOTION_FIELD = "motion_status"
    LEGACY_MOTION_FIELD = "movement_status"

    @classmethod
    def self_check(cls) -> None:
        configured = str(getattr(Settings, "MOTION_FIELD_NAME", "")).strip()
        if configured != cls.CANONICAL_MOTION_FIELD:
            raise DataContractError(
                "Invalid MOTION_FIELD_NAME config: "
                f"expected '{cls.CANONICAL_MOTION_FIELD}', got '{configured or '<empty>'}'"
            )

    @classmethod
    def normalize_motion_value(
        cls,
        obj: Dict[str, Any],
    ) -> Tuple[str, bool]:
        if cls.CANONICAL_MOTION_FIELD in obj:
            raw = obj.get(cls.CANONICAL_MOTION_FIELD, "-1")
            return cls._normalize_motion(raw), False

        if cls.LEGACY_MOTION_FIELD in obj:
            raw = obj.get(cls.LEGACY_MOTION_FIELD, "-1")
            return cls._normalize_motion(raw), True

        if Settings.MOTION_FIELD_NAME in obj:
            raw = obj.get(Settings.MOTION_FIELD_NAME, "-1")
            return cls._normalize_motion(raw), Settings.MOTION_FIELD_NAME != cls.CANONICAL_MOTION_FIELD

        return "-1", False

    @classmethod
    def normalize_object(
        cls,
        obj: Dict[str, Any],
        frame_shape: Optional[tuple] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        class_id = str(obj.get("cls", ""))
        if class_id not in {"0", "1", "2", "3"}:
            return None, False

        landing = str(obj.get("landing_status", "-1"))
        if landing not in {"-1", "0", "1"}:
            landing = "-1"

        motion, used_alias = cls.normalize_motion_value(obj)

        x1 = cls._safe_int(obj.get("top_left_x", 0))
        y1 = cls._safe_int(obj.get("top_left_y", 0))
        x2 = cls._safe_int(obj.get("bottom_right_x", 0))
        y2 = cls._safe_int(obj.get("bottom_right_y", 0))

        if frame_shape and len(frame_shape) >= 2:
            frame_h = cls._safe_int(frame_shape[0])
            frame_w = cls._safe_int(frame_shape[1])
            x1, y1, x2, y2 = cls._clamp_bbox(
                x1, y1, x2, y2, frame_w=frame_w, frame_h=frame_h
            )

        out_class = int(class_id) if Settings.PAYLOAD_CLS_AS_INT else str(class_id)
        normalized = {
            "cls": out_class,
            "landing_status": cls._format_status_value(landing),
            cls.CANONICAL_MOTION_FIELD: cls._format_status_value(motion),
            "top_left_x": int(x1),
            "top_left_y": int(y1),
            "bottom_right_x": int(x2),
            "bottom_right_y": int(y2),
            "_confidence": cls._safe_float(obj.get("_confidence", obj.get("confidence", 0.0))),
        }
        return normalized, used_alias

    @staticmethod
    def _normalize_motion(value: Any) -> str:
        v = str(value)
        if v not in {"-1", "0", "1"}:
            return "-1"
        return v

    @staticmethod
    def _format_status_value(value: str) -> Any:
        return int(value) if Settings.PAYLOAD_STATUS_AS_INT else str(value)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_bbox(
        x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))
        if x2 <= x1:
            x2 = min(frame_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_h - 1, y1 + 1)
        return x1, y1, x2, y2

    @classmethod
    def validate_top_level_payload(cls, payload: Dict[str, Any]) -> None:
        required_fields = {
            "id",
            "user",
            "frame",
            "detected_objects",
            "detected_translations",
            "detected_undefined_objects",
        }
        if not isinstance(payload, dict):
            raise DataContractError("Payload must be dict")
        if not required_fields.issubset(payload.keys()):
            missing = sorted(list(required_fields - set(payload.keys())))
            raise DataContractError(f"Payload missing fields: {missing}")

    @classmethod
    def canonicalize_objects(
        cls,
        objects: List[Dict[str, Any]],
        frame_shape: Optional[tuple] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        normalized: List[Dict[str, Any]] = []
        alias_count = 0
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            det, used_alias = cls.normalize_object(obj, frame_shape=frame_shape)
            if det is None:
                continue
            normalized.append(det)
            if used_alias:
                alias_count += 1
        return normalized, alias_count
