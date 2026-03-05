"""Payload schema and adapter for outbound competition JSON (birleşik payload_schema + payload_adapter)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings
from src.competition_contract import DataContractError


# ─── CompetitionPayloadSchema ─────────────────────────────────────────────────
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
        cls._read_status_type_profile()

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
        x1 = cls._safe_float(obj.get("top_left_x", 0))
        y1 = cls._safe_float(obj.get("top_left_y", 0))
        x2 = cls._safe_float(obj.get("bottom_right_x", 0))
        y2 = cls._safe_float(obj.get("bottom_right_y", 0))
        if frame_shape and len(frame_shape) >= 2:
            frame_h = cls._safe_int(frame_shape[0])
            frame_w = cls._safe_int(frame_shape[1])
            x1, y1, x2, y2 = cls._clamp_bbox(
                x1, y1, x2, y2, frame_w=frame_w, frame_h=frame_h
            )
        out_class = int(class_id) if Settings.PAYLOAD_CLS_AS_INT else str(class_id)
        landing_value = cls.cast_status_value(int(landing))
        motion_value = cls.cast_status_value(int(motion))
        normalized = {
            "cls": out_class,
            "landing_status": landing_value,
            cls.CANONICAL_MOTION_FIELD: motion_value,
            "top_left_x": round(x1, 2),
            "top_left_y": round(y1, 2),
            "bottom_right_x": round(x2, 2),
            "bottom_right_y": round(y2, 2),
            "_confidence": cls._safe_float(obj.get("_confidence", obj.get("confidence", 0.0))),
        }
        return normalized, used_alias

    @staticmethod
    def _normalize_motion(value: Any) -> str:
        v = str(value)
        if v not in {"-1", "0", "1"}:
            return "-1"
        return v

    @classmethod
    def cast_status_value(cls, value: int) -> Any:
        profile = cls._read_status_type_profile()
        if profile == "string":
            return str(int(value))
        return int(value)

    @staticmethod
    def _read_status_type_profile() -> str:
        profile_raw = str(
            getattr(Settings, "PAYLOAD_STATUS_TYPE_PROFILE", "int")
        ).strip().lower()
        if profile_raw in {"int", "integer"}:
            return "int"
        if profile_raw in {"string", "str"}:
            return "string"
        raise DataContractError(
            "Invalid PAYLOAD_STATUS_TYPE_PROFILE config: "
            f"expected 'int' or 'string', got '{profile_raw or '<empty>'}'"
        )

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
        x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int
    ) -> Tuple[float, float, float, float]:
        x1 = max(0.0, min(x1, float(frame_w - 1)))
        y1 = max(0.0, min(y1, float(frame_h - 1)))
        x2 = max(0.0, min(x2, float(frame_w - 1)))
        y2 = max(0.0, min(y2, float(frame_h - 1)))
        if x2 <= x1:
            x2 = min(float(frame_w - 1), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(frame_h - 1), y1 + 1.0)
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


# ─── PayloadAdapter ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PayloadProfile:
    version: str
    cls_as_int: bool
    status_type: str
    motion_field: str


class PayloadAdapter:
    """Single point for payload profile versioning and field casting."""

    _SUPPORTED = {"v1", "v1_legacy", "v2_int"}
    _CANONICAL_MOTION_FIELD = "motion_status"
    _LEGACY_MOTION_FIELD = "movement_status"

    @classmethod
    def self_check(cls) -> None:
        cls.resolve_profile()

    @classmethod
    def resolve_profile(cls, version: str | None = None) -> PayloadProfile:
        requested = (
            str(version or getattr(Settings, "PAYLOAD_ADAPTER_VERSION", "v1"))
            .strip()
            .lower()
        )
        if requested not in cls._SUPPORTED:
            raise DataContractError(
                f"Unsupported PAYLOAD_ADAPTER_VERSION='{requested}'. "
                f"Supported={sorted(cls._SUPPORTED)}"
            )
        if requested == "v1":
            return PayloadProfile(
                version="v1",
                cls_as_int=bool(getattr(Settings, "PAYLOAD_CLS_AS_INT", False)),
                status_type=str(
                    getattr(Settings, "PAYLOAD_STATUS_TYPE_PROFILE", "int")
                ).strip().lower(),
                motion_field=cls._CANONICAL_MOTION_FIELD,
            )
        if requested == "v1_legacy":
            return PayloadProfile(
                version="v1_legacy",
                cls_as_int=False,
                status_type="string",
                motion_field=cls._LEGACY_MOTION_FIELD,
            )
        return PayloadProfile(
            version="v2_int",
            cls_as_int=True,
            status_type="int",
            motion_field=cls._CANONICAL_MOTION_FIELD,
        )

    @classmethod
    def adapt_payload(
        cls,
        payload: Dict[str, Any],
        version: str | None = None,
    ) -> Dict[str, Any]:
        profile = cls.resolve_profile(version=version)
        objects_raw = payload.get("detected_objects", [])
        objects: List[Dict[str, Any]] = []
        if isinstance(objects_raw, list):
            for obj in objects_raw:
                if not isinstance(obj, dict):
                    continue
                objects.append(cls._adapt_object(obj, profile))
        adapted = {
            "id": payload.get("id"),
            "user": payload.get("user"),
            "frame": payload.get("frame"),
            "detected_objects": objects,
            "detected_translations": payload.get("detected_translations", []),
            "detected_undefined_objects": payload.get(
                "detected_undefined_objects", []
            ),
        }
        return adapted

    @classmethod
    def _adapt_object(cls, obj: Dict[str, Any], profile: PayloadProfile) -> Dict[str, Any]:
        class_value = cls._safe_int(obj.get("cls", -1), default=-1)
        out_class: Any = class_value if profile.cls_as_int else str(class_value)
        landing = cls._cast_status(obj.get("landing_status", -1), profile.status_type)
        motion_raw = obj.get(cls._CANONICAL_MOTION_FIELD, obj.get(cls._LEGACY_MOTION_FIELD, -1))
        motion = cls._cast_status(motion_raw, profile.status_type)
        adapted: Dict[str, Any] = {
            "cls": out_class,
            "landing_status": landing,
            "top_left_x": cls._safe_float(obj.get("top_left_x", 0), default=0.0),
            "top_left_y": cls._safe_float(obj.get("top_left_y", 0), default=0.0),
            "bottom_right_x": cls._safe_float(obj.get("bottom_right_x", 1), default=1.0),
            "bottom_right_y": cls._safe_float(obj.get("bottom_right_y", 1), default=1.0),
        }
        adapted[profile.motion_field] = motion
        return adapted

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _cast_status(value: Any, status_type: str) -> Any:
        safe = PayloadAdapter._safe_int(value, default=-1)
        if status_type in {"string", "str"}:
            return str(safe)
        return int(safe)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return default
