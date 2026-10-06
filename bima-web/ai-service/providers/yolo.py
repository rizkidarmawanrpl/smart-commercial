import asyncio
import base64
from typing import Dict, List, Optional

import cv2
import numpy as np

from providers.base import BaseVisionProvider
from schemas import BBox, ClassDef, DetectionItem, DetectionSchema, ModelConfigPayload
from services.notice_ocr import NOTICE_MODEL_CLASS, NOTICE_OCR_MODEL_ID, NoticeOcrResult, notice_ocr, read_notices
from services.sign_condition import STAGE2_MODEL_ID, STAGE2_MODEL_LABEL, SIGN_MODEL_CLASS, classifier, classify_signs
from services.yolo_engine import RawDetection, engine


def decode_image(image_base64: str) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Citra tidak dapat didecode (format tidak dikenali atau berkas rusak).")
    return img


def to_detection_items(
    raw: List[RawDetection],
    active_classes: List[ClassDef],
    timestamp_seconds: Optional[float],
    frame_index: Optional[int],
    conditions: Optional[Dict[int, str]] = None,
    ocr: Optional[Dict[int, NoticeOcrResult]] = None,
) -> List[DetectionItem]:
    """Memetakan keluaran model ke kelas aktif. Kelas yang tidak aktif/tidak terdaftar dibuang, bukan ditebak."""
    by_model_class = {(c.model_class or c.name): c for c in active_classes}
    items: List[DetectionItem] = []
    for i, d in enumerate(raw):
        cls = by_model_class.get(d.model_class)
        if cls is None:
            continue
        # Hasil Tahap 2 hanya dilampirkan pada kelas yang memintanya (has_condition_stage).
        state = conditions.get(i) if (conditions and cls.has_condition_stage) else None
        # Hasil OCR hanya dilampirkan pada kelas yang memintanya (has_ocr_stage).
        notice = ocr.get(i) if (ocr and cls.has_ocr_stage) else None
        text = f"Terdeteksi oleh model YOLO (confidence {d.confidence:.2f})"
        if state:
            text += f"; kondisi: {state} ({STAGE2_MODEL_LABEL})"
        if notice:
            text += f"; teks notis: {notice.label} ({NOTICE_OCR_MODEL_ID})"
        items.append(
            DetectionItem(
                class_id=cls.id,
                class_name=cls.name,
                bbox=BBox(x=d.x, y=d.y, width=d.width, height=d.height),
                condition=text,
                # Kelayakan tidak berlaku untuk YOLO; tingkat risiko dihitung di sisi web (Severity x Exposure).
                feasibility="tidak_dinilai",
                confidence=round(d.confidence, 4),
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                condition_state=state,
                condition_model=STAGE2_MODEL_ID if state else None,
                served_by=d.served_by or None,
                ocr_label=notice.label if notice else None,
                ocr_text=notice.text if notice else None,
                ocr_confidence=notice.confidence if notice else None,
                ocr_matched_roots=notice.matched_roots if notice else [],
                ocr_manual_check=notice.needs_manual_check if notice else False,
                ocr_model=NOTICE_OCR_MODEL_ID if notice else None,
            )
        )
    return items


def wants_condition_stage(active_classes: List[ClassDef]) -> bool:
    """Tahap 2 dijalankan hanya bila classifier aktif dan ada kelas rambu yang memintanya."""
    return classifier.enabled and any(c.has_condition_stage and (c.model_class or c.name) == SIGN_MODEL_CLASS for c in active_classes)


def wants_ocr_stage(active_classes: List[ClassDef]) -> bool:
    """OCR notis dijalankan hanya bila diaktifkan (NOTICE_OCR_ENABLED) dan ada kelas house_notice yang memintanya."""
    return notice_ocr.enabled and any(c.has_ocr_stage and (c.model_class or c.name) == NOTICE_MODEL_CLASS for c in active_classes)


class YoloProvider(BaseVisionProvider):
    """Deteksi objek lokal (CPU): baseline yolo11n_seed0 atau varian ablasi sesuai nama model. Kontrak provider sama dengan VLM."""

    def __init__(self, config: ModelConfigPayload):
        super().__init__(config)

    async def detect(
        self,
        image_base64: str,
        active_classes: List[ClassDef],
        timestamp_seconds: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> DetectionSchema:
        img = decode_image(image_base64)
        raw, _ = await asyncio.to_thread(engine.detect, img, None, self.config.model_name)
        conditions = await asyncio.to_thread(classify_signs, img, raw) if wants_condition_stage(active_classes) else None
        ocr = await asyncio.to_thread(read_notices, img, raw) if wants_ocr_stage(active_classes) else None
        return DetectionSchema(detections=to_detection_items(raw, active_classes, timestamp_seconds, frame_index, conditions, ocr))

    async def test_connection(self) -> bool:
        return bool(engine.status(self.config.model_name)["present"])
