"""Tahap 2 kategori house_notice: OCR teks notis (jual / sewa) per crop.

Alur (mengikuti notebook Training_Ulang_House_Notice_COLAB, Bagian 7):
  1. Stage 1 (YOLO house_notice) mendeteksi notis yang menempel pada rumah.
  2. Tiap kotak di-crop dengan margin 0,05, lalu di-upscale ke sisi pendek minimal 320 px.
  3. EasyOCR (id + en) membaca teks pada crop.
  4. Pencocokan kata kunci terhadap akar kata "jual" dan "sewa" (toleransi typo OCR: rapidfuzz 0,75).
  5. Hasil dilaporkan BINER: `sale_or_rent` atau `tidak_teridentifikasi`. Crop dengan sisi pendek <= 320 px (sebelum upscale)
     ditandai `needs_manual_check`: itu prediktor keterbacaan, bukan akurasi.

Tahap ini BUKAN model terlatih. Parameter di bawah dikunci mengikuti naskah (BAB III) dan disalin dari notebook, sehingga
tidak dibaca dari environment; environment hanya mengaktifkan tahap ini (NOTICE_OCR_ENABLED) dan menunjuk folder model OCR.

Dua perbedaan dari notebook (keduanya disengaja dan diuji di test_notice_ocr.py):
  * Pencocokan fuzzy. Notebook memakai `partial_ratio >= 0,75` pada setiap kata. Untuk kata yang LEBIH PENDEK dari akar kata,
    `partial_ratio` menjajarkan kata itu di dalam akar kata, sehingga huruf tunggal hasil derau OCR ("a", "u", "e", "s") dan
    serpihan ("se", "wa", "al") dianggap cocok "jual"/"sewa" dan notis tak berteks ikut berlabel sale_or_rent. Di sini
    `partial_ratio` hanya dipakai bila kata sama panjang atau lebih panjang dari akar kata; `ratio` tetap berlaku untuk semua kata.
  * Daftar pengecualian. "jual beli mobil bekas" pada notebook tidak pernah bisa cocok karena pengecualian diperiksa per kata;
    yang berfungsi hanya "jualan" dan "terjual", dan hanya itu yang dipertahankan.
Crop diambil dari frame tersimpan yang sama dengan yang dideteksi (resolusi sesuai FRAME_MAX_WIDTH), bukan dari foto asli
beresolusi penuh seperti pada notebook; crop kecil dilaporkan lewat `needs_manual_check`.
"""
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from config import optional_env
from services.yolo_engine import RawDetection, engine as yolo_engine

logger = logging.getLogger("ai_service.notice_ocr")

NOTICE_MODEL_CLASS = "house_notice"
NOTICE_OCR_MODEL_ID = "easyocr_id_en_keyword"
NOTICE_OCR_MODEL_LABEL = "EasyOCR (id+en) + pencocokan kata kunci jual/sewa"

# Parameter terkunci (BAB III.G), disalin dari notebook.
OCR_MARGIN = 0.05
OCR_MIN_SHORT_SIDE_PX = 320
MANUAL_CHECK_MAX_PX = 320
OCR_LANGS = ["id", "en"]
FUZZY_TOLERANCE = 0.75
KEYWORD_ROOTS = ("jual", "sewa")
EXCLUDE_WORDS = ("jualan", "terjual")

LABEL_SALE_OR_RENT = "sale_or_rent"
LABEL_UNIDENTIFIED = "tidak_teridentifikasi"


@dataclass
class NoticeOcrResult:
    label: str
    text: str
    confidence: float
    matched_roots: List[str] = field(default_factory=list)
    needs_manual_check: bool = False


def normalize_word(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def match_keywords(
    ocr_text: str,
    roots: Sequence[str] = KEYWORD_ROOTS,
    fuzzy_threshold: float = FUZZY_TOLERANCE,
    exclude_words: Sequence[str] = EXCLUDE_WORDS,
) -> Dict[str, object]:
    """Mencocokkan kata hasil OCR dengan akar kata. Pengecualian diperiksa per kata pada teks asli (sebelum spasi dibuang)."""
    from rapidfuzz import fuzz

    words = [normalize_word(w) for w in re.findall(r"[A-Za-z]+", ocr_text)]
    words = [w for w in words if w]
    excluded = [normalize_word(e) for e in exclude_words]

    matched = set()
    for w in words:
        if any(ex in w for ex in excluded):
            continue  # hanya kata ini yang dikecualikan, bukan seluruh crop ("DIJUAL ANEE" tetap cocok lewat kata "dijual")
        for root in roots:
            score = fuzz.ratio(w, root) / 100.0
            # partial_ratio hanya bermakna bila akar kata dapat "bergeser" di dalam kata (kata >= akar kata).
            partial = fuzz.partial_ratio(w, root) / 100.0 if len(w) >= len(root) else 0.0
            if score >= fuzzy_threshold or partial >= fuzzy_threshold:
                matched.add(root)
    return {"matched": bool(matched), "matched_roots": sorted(matched), "words_seen": words}


def crop_notice(image_bgr: np.ndarray, xyxy, margin: float = OCR_MARGIN, min_short_side: int = OCR_MIN_SHORT_SIDE_PX):
    """Crop dengan margin relatif lalu upscale; mengembalikan (crop, needs_manual_check) atau None bila crop kosong."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    short_side = min(ch, cw)
    needs_manual = short_side <= MANUAL_CHECK_MAX_PX
    if 0 < short_side < min_short_side:
        scale = min_short_side / short_side
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    return crop, needs_manual


def _truthy(v: Optional[str]) -> bool:
    return v is not None and v.strip().lower() in {"1", "true", "yes", "on"}


class NoticeOcr:
    """Memuat EasyOCR saat pertama dipakai. Aktif hanya bila env NOTICE_OCR_ENABLED benar."""

    def __init__(self):
        self._reader = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return _truthy(optional_env("NOTICE_OCR_ENABLED"))

    @property
    def model_dir(self) -> Optional[str]:
        d = optional_env("NOTICE_OCR_MODEL_DIR")
        if d is None:
            return None
        return d if os.path.isabs(d) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)

    def status(self) -> Dict[str, object]:
        """Memeriksa kesiapan tanpa memuat model (pustaka terpasang; berkas model ada bila folder model ditentukan)."""
        if not self.enabled:
            return {"enabled": False, "ready": False, "model": NOTICE_OCR_MODEL_ID, "detail": "nonaktif"}
        try:
            import easyocr  # noqa: F401
            import rapidfuzz  # noqa: F401
        except ImportError as e:
            return {"enabled": True, "ready": False, "model": NOTICE_OCR_MODEL_ID, "detail": f"pustaka belum terpasang ({e.name}); pip install -r requirements-ocr.txt"}
        d = self.model_dir
        if d and not (os.path.isfile(os.path.join(d, "craft_mlt_25k.pth")) and os.path.isfile(os.path.join(d, "latin_g2.pth"))):
            return {"enabled": True, "ready": True, "model": NOTICE_OCR_MODEL_ID, "detail": f"model OCR belum ada di {d}; akan diunduh saat pertama dipakai"}
        return {"enabled": True, "ready": True, "model": NOTICE_OCR_MODEL_ID, "detail": "siap"}

    def load(self) -> None:
        with self._lock:
            if self._reader is not None:
                return
            if not self.enabled:
                raise RuntimeError("Tahap 2 notis (OCR) tidak aktif (NOTICE_OCR_ENABLED).")
            try:
                import easyocr
                import rapidfuzz  # noqa: F401
            except ImportError as e:
                raise RuntimeError(f"Pustaka OCR belum terpasang ({e.name}): pip install -r requirements-ocr.txt") from e
            gpu = False
            if (yolo_engine.device or "").lower() != "cpu":
                import torch

                gpu = torch.cuda.is_available()
            kwargs = {"gpu": gpu}
            if self.model_dir:
                os.makedirs(self.model_dir, exist_ok=True)
                kwargs["model_storage_directory"] = self.model_dir
            self._reader = easyocr.Reader(OCR_LANGS, **kwargs)
            logger.info("EasyOCR dimuat (%s, gpu=%s)", "+".join(OCR_LANGS), gpu)

    def read_crop(self, crop_bgr: np.ndarray) -> NoticeOcrResult:
        self.load()
        with self._lock:
            results = self._reader.readtext(crop_bgr, detail=1)
        text = " ".join(r[1] for r in results)
        conf = float(np.mean([r[2] for r in results])) if results else 0.0
        m = match_keywords(text)
        return NoticeOcrResult(
            label=LABEL_SALE_OR_RENT if m["matched"] else LABEL_UNIDENTIFIED,
            text=text,
            confidence=round(conf, 4),
            matched_roots=list(m["matched_roots"]),  # type: ignore[arg-type]
        )


notice_ocr = NoticeOcr()


def read_notices(
    image_bgr: np.ndarray,
    detections: Sequence[RawDetection],
    ocr: NoticeOcr = notice_ocr,
) -> Dict[int, NoticeOcrResult]:
    """OCR pada tiap deteksi house_notice. Kunci hasil = indeks pada `detections`; crop kosong dilewati."""
    out: Dict[int, NoticeOcrResult] = {}
    for i, d in enumerate(detections):
        if d.model_class != NOTICE_MODEL_CLASS:
            continue
        cropped = crop_notice(image_bgr, d.xyxy)
        if cropped is None:
            logger.warning("Crop notis kosong dilewati (kotak %s).", d.xyxy)
            continue
        crop, needs_manual = cropped
        res = ocr.read_crop(crop)
        res.needs_manual_check = needs_manual
        out[i] = res
    return out
