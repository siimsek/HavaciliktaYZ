"""Görev 3: Referans obje eşleştirme (ORB/SIFT). Referans görüntülerden feature çıkar, karede ara."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class ReferenceObject:
    """Referans objenin feature bilgileri."""

    def __init__(
        self,
        object_id: int,
        image: np.ndarray,
        keypoints: list,
        descriptors: Optional[np.ndarray],
        label: str = "",
        trust_score: float = 1.0,
    ) -> None:
        self.object_id = object_id
        self.image = image
        self.h, self.w = image.shape[:2]
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.label = label
        self.trust_score = max(0.0, min(1.0, float(trust_score)))


class ImageMatcher:
    """Referans obje eşleştirme (ORB/SIFT)."""

    def __init__(self) -> None:
        self.log = Logger("Task3")
        self.references: List[ReferenceObject] = []
        self._references_by_id: Dict[int, ReferenceObject] = {}
        self._reference_lifecycle: Dict[int, str] = {}
        self._reference_trust_scores: Dict[int, float] = {}
        self._quarantined_references: Dict[int, str] = {}
        self._last_load_stats: Dict[str, int] = {
            "total": 0,
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        self._frame_counter: int = 0

        method = Settings.TASK3_FEATURE_METHOD.upper()
        if method == "SIFT":
            self.detector = cv2.SIFT_create()
            self.norm_type = cv2.NORM_L2
            self.log.info("Feature method: SIFT (daha robust, yavaş)")
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.norm_type = cv2.NORM_HAMMING
            self.log.info("Feature method: ORB (hızlı, offline-uyumlu)")

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

        self.log.info(
            f"ImageMatcher initialized | "
            f"similarity_threshold={Settings.TASK3_SIMILARITY_THRESHOLD} | "
            f"fallback_threshold={Settings.TASK3_FALLBACK_THRESHOLD}"
        )

    @staticmethod
    def _normalize_object_id(raw_object_id: Any) -> Optional[int]:
        if isinstance(raw_object_id, bool) or raw_object_id is None:
            return None
        if isinstance(raw_object_id, int):
            object_id = raw_object_id
        elif isinstance(raw_object_id, float):
            if not raw_object_id.is_integer():
                return None
            object_id = int(raw_object_id)
        elif isinstance(raw_object_id, str):
            stripped = raw_object_id.strip()
            if not stripped:
                return None
            try:
                object_id = int(stripped)
            except ValueError:
                return None
        else:
            return None
        if object_id < 0:
            return None
        return object_id

    def _set_reference_trust(self, object_id: int, score: float) -> None:
        self._reference_trust_scores[object_id] = max(0.0, min(1.0, float(score)))

    def _quarantine_reference(self, object_id: int, reason: str, context: str = "") -> None:
        self._quarantined_references[object_id] = reason
        self._reference_lifecycle[object_id] = "quarantined"
        self._set_reference_trust(object_id, 0.0)
        context_suffix = f" {context}" if context else ""
        self.log.warn(
            f"event=task3_ref_quarantined reason={reason} object_id={object_id}{context_suffix}"
        )

    def load_references(self, reference_images: List[Dict[str, Any]]) -> int:
        self.references.clear()
        self._references_by_id.clear()
        self._reference_lifecycle.clear()
        self._reference_trust_scores.clear()
        self._quarantined_references.clear()
        self._last_load_stats = {
            "total": len(reference_images),
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        loaded = 0

        for idx, ref_data in enumerate(reference_images):
            if not isinstance(ref_data, dict):
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_quarantined reason=invalid_record_type object_id=unknown index={idx}"
                )
                continue

            object_id = self._normalize_object_id(ref_data.get("object_id"))
            if object_id is None:
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_quarantined reason=invalid_object_id object_id=unknown index={idx} raw_object_id={ref_data.get('object_id')}"
                )
                continue

            self._reference_lifecycle[object_id] = "received"

            if object_id in self._references_by_id:
                self._last_load_stats["duplicate"] += 1
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_duplicate_detected object_id={object_id} index={idx}"
                )
                self._quarantine_reference(
                    object_id=object_id,
                    reason="duplicate_object_id",
                    context=f"index={idx}",
                )
                continue

            label = ref_data.get("label", f"ref_{object_id}")

            if "image" in ref_data and ref_data["image"] is not None:
                image = ref_data["image"]
            elif "path" in ref_data and os.path.isfile(ref_data["path"]):
                image = cv2.imread(ref_data["path"])
                if image is None:
                    self._last_load_stats["quarantined"] += 1
                    self._quarantine_reference(object_id, "image_read_failed", context=f"index={idx} path={ref_data['path']}")
                    continue
            else:
                self._last_load_stats["quarantined"] += 1
                self._quarantine_reference(object_id, "missing_image_source", context=f"index={idx}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 4:
                self._last_load_stats["quarantined"] += 1
                self._quarantine_reference(
                    object_id,
                    "insufficient_feature",
                    context=f"index={idx} feature_count={len(keypoints) if keypoints else 0}",
                )
                continue

            self._reference_lifecycle[object_id] = "validated"
            feature_ratio = min(1.0, len(keypoints) / 150.0)
            trust_score = max(0.35, feature_ratio)
            self._set_reference_trust(object_id, trust_score)
            ref_obj = ReferenceObject(
                object_id=object_id,
                image=image,
                keypoints=keypoints,
                descriptors=descriptors,
                label=label,
                trust_score=trust_score,
            )
            self.references.append(ref_obj)
            self._references_by_id[object_id] = ref_obj
            self._reference_lifecycle[object_id] = "loaded"
            loaded += 1
            self._last_load_stats["valid"] += 1

            self.log.info(
                f"Referans #{object_id} yüklendi: "
                f"{ref_obj.w}x{ref_obj.h}px, {len(keypoints)} feature, trust={trust_score:.2f}"
            )

        self.references = list(self._references_by_id.values())
        self.log.info(
            f"event=task3_ref_validation_summary total={self._last_load_stats['total']} "
            f"valid={self._last_load_stats['valid']} duplicate={self._last_load_stats['duplicate']} "
            f"quarantined={self._last_load_stats['quarantined']}"
        )
        self.log.success(f"Toplam {loaded}/{len(reference_images)} referans obje yüklendi")
        return loaded

    def load_references_from_directory(self, directory: Optional[str] = None) -> int:
        ref_dir = directory or Settings.TASK3_REFERENCE_DIR
        if not os.path.isdir(ref_dir):
            self.log.warn(f"Referans dizini bulunamadı: {ref_dir}")
            return 0

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ref_list: List[Dict[str, Any]] = []

        files = sorted(os.listdir(ref_dir))
        for idx, fname in enumerate(files, start=1):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_exts:
                continue

            if len(ref_list) >= Settings.TASK3_MAX_REFERENCES:
                self.log.warn(
                    f"Maks referans limiti ({Settings.TASK3_MAX_REFERENCES}) aşıldı, "
                    f"fazla dosyalar atlanıyor"
                )
                break

            ref_list.append({
                "object_id": idx,
                "path": os.path.join(ref_dir, fname),
                "label": os.path.splitext(fname)[0],
            })

        return self.load_references(ref_list)

    def match(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        self._frame_counter += 1

        if not self.references:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)

        if frame_desc is None or len(frame_kp) < 4:
            return []

        results: List[Dict[str, Any]] = []

        for ref in self.references:
            current_trust = self._reference_trust_scores.get(ref.object_id, ref.trust_score)
            if ref.object_id in self._quarantined_references or current_trust < 0.30:
                continue

            bbox = self._match_reference(ref, frame_kp, frame_desc, gray.shape)
            if bbox is not None:
                if ref.object_id not in self._references_by_id:
                    continue
                self._reference_lifecycle[ref.object_id] = "matched"
                self._set_reference_trust(ref.object_id, min(1.0, current_trust + 0.02))
                results.append({
                    "object_id": ref.object_id,
                    "top_left_x": bbox[0],
                    "top_left_y": bbox[1],
                    "bottom_right_x": bbox[2],
                    "bottom_right_y": bbox[3],
                })
            else:
                updated_trust = max(0.0, current_trust - 0.01)
                self._set_reference_trust(ref.object_id, updated_trust)
                if updated_trust < 0.30:
                    self._quarantine_reference(
                        ref.object_id,
                        "low_trust_after_mismatch",
                        context=f"frame={self._frame_counter} trust={updated_trust:.2f}",
                    )

        if results:
            self.log.debug(
                f"Frame {self._frame_counter}: {len(results)} referans obje tespit edildi"
            )

        return results

    def _match_reference(
        self,
        ref: ReferenceObject,
        frame_kp: list,
        frame_desc: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        if ref.descriptors is None:
            return None

        try:
            matches = self.matcher.knnMatch(ref.descriptors, frame_desc, k=2)
        except cv2.error:
            return None

        good_matches = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        min_matches = max(4, int(len(ref.keypoints) * 0.05))
        if len(good_matches) < min_matches:
            return None

        similarity = len(good_matches) / max(1, len(ref.keypoints))

        # Periyodik olarak daha düşük eşik (fallback) denemek için
        threshold = Settings.TASK3_SIMILARITY_THRESHOLD
        if self._frame_counter % Settings.TASK3_FALLBACK_INTERVAL == 0:
            threshold = Settings.TASK3_FALLBACK_THRESHOLD

        if similarity < threshold:
            return None

        try:
            src_pts = np.float32(
                [ref.keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_kp[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            if len(np.unique(src_pts.reshape(-1, 2), axis=0)) < 4:
                return None
            if len(np.unique(dst_pts.reshape(-1, 2), axis=0)) < 4:
                return None

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or M.shape != (3, 3):
                self.log.warn("Homografi dejenere oldu, nokta bazlı bounding rect (fallback) çıkarıldı")
                pts = dst_pts.reshape(-1, 2)
            else:
                h, w = ref.h, ref.w
                corners = np.float32(
                    [[0, 0], [w, 0], [w, h], [0, h]]
                ).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, M)
                pts = transformed.reshape(-1, 2)
                if len(pts) >= 4:
                    hull = cv2.convexHull(pts.astype(np.float32))
                    if hull is None or len(hull) < 4 or not cv2.isContourConvex(hull):
                        return None

            x1 = float(max(0, pts[:, 0].min()))
            y1 = float(max(0, pts[:, 1].min()))
            x2 = float(min(frame_shape[1] if len(frame_shape) > 1 else frame_shape[0], pts[:, 0].max()))
            y2 = float(min(frame_shape[0], pts[:, 1].max()))

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 5 or bbox_h < 5:
                return None
            if bbox_w > frame_shape[1] * 0.8 or bbox_h > frame_shape[0] * 0.8:
                return None

            return (x1, y1, x2, y2)

        except (cv2.error, ValueError, IndexError):
            return None

    @property
    def reference_count(self) -> int:
        return len(self.references)

    @property
    def is_ready(self) -> bool:
        return len(self.references) > 0

    def reset(self) -> None:
        self.references.clear()
        self._references_by_id.clear()
        self._reference_lifecycle.clear()
        self._reference_trust_scores.clear()
        self._quarantined_references.clear()
        self._last_load_stats = {
            "total": 0,
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        self._frame_counter = 0
        self.log.info("ImageMatcher reset")

    @property
    def id_lifecycle_states(self) -> Dict[int, str]:
        return dict(self._reference_lifecycle)

    @property
    def last_load_stats(self) -> Dict[str, int]:
        return dict(self._last_load_stats)

    @property
    def reference_trust_scores(self) -> Dict[int, float]:
        return dict(self._reference_trust_scores)

    @property
    def quarantined_references(self) -> Dict[int, str]:
        return dict(self._quarantined_references)
