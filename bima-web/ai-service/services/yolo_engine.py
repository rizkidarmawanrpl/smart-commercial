"""Inferensi YOLO (Ultralytics) untuk enam model kategori hasil training (RQ2).

Konfigurasi hanya dari environment (tanpa nilai bawaan di kode):
  YOLO_WEIGHTS_DIR  folder bobot (relatif terhadap ai-service/ atau absolut), mis. models/yolo11n_seed0
  YOLO_CONF         ambang confidence
  YOLO_DEVICE       opsional, mis. "cpu" (kosong = pilihan otomatis Ultralytics)
  YOLO_IMGSZ        opsional, ukuran input (kosong = bawaan model)

Berkas bobot: <kategori>-best.pt untuk kategori pavedroad, vegetation, weeds, sign, banner, house_notice.
Model dimuat saat pertama dibutuhkan; berkas yang tidak ada dilaporkan di `status()`, bukan diabaikan diam-diam.

Varian ablasi (attention): baseline tetap model `YOLO_WEIGHTS_DIR` dan dipakai bila tidak ada nama model. Varian dipilih
lewat nama model (ModelConfig.modelName dari web) dan dimuat dari folder `<YOLO_VARIANTS_DIR>/<nama>/<kategori>-best.pt`.
  YOLO_VARIANTS_DIR     opsional; bawaan: folder induk YOLO_WEIGHTS_DIR (varian berdampingan dengan baseline)
  YOLO_BASELINE_MODEL   opsional; nama model baseline; bawaan: nama folder YOLO_WEIGHTS_DIR (mis. yolo11n_seed0)
Kategori yang belum punya bobot di folder varian dilayani baseline dan dilaporkan sebagai `fallback`, tidak diam-diam.
"""
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import optional_env, optional_env_int, require_env, require_env_float

logger = logging.getLogger("ai_service.yolo")

CATEGORY_MODELS: Tuple[str, ...] = ("pavedroad", "vegetation", "weeds", "sign", "banner", "house_notice")
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RawDetection:
    model_class: str  # nama kelas keluaran model, mis. "pavedroad_pothole"
    source_model: str  # kategori model asal, mis. "pavedroad"
    confidence: float
    x: float  # kiri-atas, ternormalisasi 0-1
    y: float
    width: float
    height: float
    # Kotak dalam piksel gambar asli (x1, y1, x2, y2); dipakai Tahap 2 untuk memotong crop.
    xyxy: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Nama model yang menghasilkan kotak ini: varian, atau baseline bila kategorinya dilayani fallback.
    served_by: str = ""


@dataclass
class DetectTimings:
    per_model_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.per_model_ms.values())


# Nama varian menjadi nama folder, jadi dibatasi agar tidak bisa keluar dari YOLO_VARIANTS_DIR.
_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MISSING_CLASS_RE = re.compile(r"Can't get attribute '([^']+)'")


@dataclass
class ModelSet:
    """Model per kategori untuk satu nama model. `fallback` = kategori yang dilayani baseline karena varian belum punya bobot."""

    name: str
    is_baseline: bool
    weights_dir: str
    models: Dict[str, object] = field(default_factory=dict)
    own: List[str] = field(default_factory=list)
    fallback: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class YoloEngine:
    def __init__(self):
        self._sets: Dict[str, ModelSet] = {}
        self._lock = threading.RLock()  # RLock: memuat varian ikut memuat baseline untuk kategori fallback

    # --- konfigurasi (dibaca saat dipakai agar import modul tidak gagal) ---
    @property
    def weights_dir(self) -> str:
        d = require_env("YOLO_WEIGHTS_DIR")
        return d if os.path.isabs(d) else os.path.join(_BASE_DIR, d)

    @property
    def baseline_name(self) -> str:
        return optional_env("YOLO_BASELINE_MODEL") or os.path.basename(self.weights_dir.rstrip("/\\"))

    @property
    def variants_dir(self) -> str:
        d = optional_env("YOLO_VARIANTS_DIR")
        if d is None:
            return os.path.dirname(self.weights_dir.rstrip("/\\"))
        return d if os.path.isabs(d) else os.path.join(_BASE_DIR, d)

    @property
    def conf(self) -> float:
        return require_env_float("YOLO_CONF")

    @property
    def device(self) -> Optional[str]:
        return optional_env("YOLO_DEVICE")

    @property
    def imgsz(self) -> Optional[int]:
        return optional_env_int("YOLO_IMGSZ")

    # --- resolusi nama model ---
    def resolve(self, model_name: Optional[str]) -> Tuple[str, str, bool]:
        """(kunci, folder bobot, is_baseline). Tanpa nama, atau nama baseline -> baseline. Nama lain harus folder varian yang ada."""
        baseline = self.baseline_name
        if not model_name or model_name == baseline:
            return baseline, self.weights_dir, True
        if not _VARIANT_NAME_RE.match(model_name):
            raise RuntimeError(f"Nama model YOLO '{model_name}' tidak valid (hanya huruf, angka, '_', '-', '.').")
        folder = os.path.join(self.variants_dir, model_name)
        if not os.path.isdir(folder):
            raise RuntimeError(
                f"Varian YOLO '{model_name}' tidak ditemukan: folder {folder} tidak ada. "
                f"Varian: letakkan <kategori>-best.pt di folder itu (nama folder = nama model di menu Model AI). "
                f"Baseline: bila '{model_name}' adalah nama baseline, isi YOLO_BASELINE_MODEL={model_name} di ai-service/.env "
                f"(nama baseline saat ini '{baseline}', diambil dari nama folder YOLO_WEIGHTS_DIR)."
            )
        return model_name, folder, False

    # --- pemuatan ---
    def weight_path(self, category: str, model_name: Optional[str] = None) -> str:
        return os.path.join(self.resolve(model_name)[1], f"{category}-best.pt")

    def status(self, model_name: Optional[str] = None) -> Dict[str, object]:
        """Memeriksa keberadaan bobot tanpa memuat model. Untuk varian: `present` = bobot milik varian, `fallback` = dilayani baseline."""
        key, folder, is_baseline = self.resolve(model_name)
        present = [c for c in CATEGORY_MODELS if os.path.isfile(os.path.join(folder, f"{c}-best.pt"))]
        absent = [c for c in CATEGORY_MODELS if c not in present]
        if is_baseline:
            fallback: List[str] = []
            missing = absent
        else:
            base_dir = self.weights_dir
            fallback = [c for c in absent if os.path.isfile(os.path.join(base_dir, f"{c}-best.pt"))]
            missing = [c for c in absent if c not in fallback]
        return {
            "model_name": key,
            "is_baseline": is_baseline,
            "weights_dir": folder,
            "present": present,
            "fallback": fallback,
            "missing": missing,
            "ok": not missing,
        }

    def _load_file(self, path: str):
        from ultralytics import YOLO

        try:
            return YOLO(path)
        except AttributeError as e:
            m = _MISSING_CLASS_RE.search(str(e))
            if not m:
                raise
            raise RuntimeError(
                f"Bobot {path} memakai class '{m.group(1)}' yang belum didefinisikan di services/attention_modules.py "
                f"(class itu dibuat di notebook training). Tambahkan class dengan nama dan atribut yang sama."
            ) from e

    def load(self, model_name: Optional[str] = None) -> ModelSet:
        key, folder, is_baseline = self.resolve(model_name)
        with self._lock:
            cached = self._sets.get(key)
            if cached is not None:
                return cached
            try:
                import ultralytics  # noqa: F401
            except ImportError as e:
                raise RuntimeError("ultralytics belum terpasang di ai-service (pip install -r requirements-yolo.txt).") from e
            from services.attention_modules import register_attention_modules

            register_attention_modules()  # checkpoint ablasi menyimpan class attention di __main__
            st = self.status(model_name)
            own = {}
            for cat in st["present"]:  # type: ignore[union-attr]
                path = os.path.join(folder, f"{cat}-best.pt")
                own[cat] = self._load_file(path)
                logger.info("YOLO dimuat: %s/%s (%s)", key, cat, path)
            if not own:
                raise RuntimeError(f"Tidak ada bobot YOLO di '{folder}' ({'YOLO_WEIGHTS_DIR' if is_baseline else 'folder varian'}).")
            models: Dict[str, object] = dict(own)
            fallback: List[str] = list(st["fallback"])  # type: ignore[arg-type]
            if fallback:
                base = self.load(None)  # baseline untuk kategori yang belum punya bobot varian
                for cat in fallback:
                    models[cat] = base.models[cat]
                logger.warning("Varian %s: kategori %s dilayani baseline %s.", key, ", ".join(fallback), base.name)
            missing = list(st["missing"])  # type: ignore[arg-type]
            if missing:
                logger.warning("Bobot YOLO tidak ditemukan untuk: %s (%s)", ", ".join(missing), key)
            ms = ModelSet(name=key, is_baseline=is_baseline, weights_dir=folder, models=models, own=list(own), fallback=fallback, missing=missing)
            self._sets[key] = ms
            return ms

    @property
    def missing_categories(self) -> List[str]:
        """Kategori baseline yang tidak punya bobot (kompatibel dengan pemakaian lama)."""
        return self.missing_for(None)

    def missing_for(self, model_name: Optional[str] = None) -> List[str]:
        return list(self.load(model_name).missing)

    # --- inferensi ---
    def detect(
        self, image_bgr: np.ndarray, conf: Optional[float] = None, model_name: Optional[str] = None
    ) -> Tuple[List[RawDetection], DetectTimings]:
        """Menjalankan seluruh model kategori pada satu citra BGR. Thread-safe (satu inferensi pada satu waktu)."""
        model_set = self.load(model_name)
        threshold = self.conf if conf is None else conf
        results: List[RawDetection] = []
        timings = DetectTimings()
        kwargs = {"conf": threshold, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        with self._lock:
            for cat, model in model_set.models.items():
                served_by = self.baseline_name if cat in model_set.fallback else model_set.name
                t0 = time.perf_counter()
                r = model.predict(image_bgr, **kwargs)[0]
                timings.per_model_ms[cat] = (time.perf_counter() - t0) * 1000.0
                names = r.names
                for box, pxbox, c, s in zip(r.boxes.xyxyn.tolist(), r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    x1, y1, x2, y2 = (_clamp01(v) for v in box)
                    results.append(
                        RawDetection(
                            model_class=names[int(c)],
                            source_model=cat,
                            confidence=float(s),
                            x=x1,
                            y=y1,
                            width=max(0.0, x2 - x1),
                            height=max(0.0, y2 - y1),
                            xyxy=(float(pxbox[0]), float(pxbox[1]), float(pxbox[2]), float(pxbox[3])),
                            served_by=served_by,
                        )
                    )
        return results, timings


engine = YoloEngine()
