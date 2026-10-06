import os

os.environ.setdefault("INTERNAL_API_SECRET", "test-secret")
os.environ.setdefault("AI_SERVICE_ALLOWED_ORIGINS", "http://localhost:3000")

import numpy as np
import pytest

pytest.importorskip("rapidfuzz")

from providers.yolo import to_detection_items, wants_ocr_stage
from schemas import ClassDef
from services.notice_ocr import (
    LABEL_SALE_OR_RENT,
    LABEL_UNIDENTIFIED,
    NoticeOcr,
    NoticeOcrResult,
    crop_notice,
    match_keywords,
    read_notices,
)
from services.yolo_engine import RawDetection


# --- pencocokan kata kunci -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("DIJUAL ANEE", ["jual"]),  # kasus bug 1 di notebook: "jualan" tidak boleh mengecualikan seluruh crop
        ("DIUUAL", ["jual"]),  # typo OCR
        ("DUUAL CEPAT", ["jual"]),
        ("Dijuail", ["jual"]),
        ("DISEWAKAN", ["sewa"]),
        ("DIJUAL / DISEWAKAN 0812", ["jual", "sewa"]),
        ("DIJUALQI", ["jual"]),  # hasil OCR nyata pada citra contoh house_notice
    ],
)
def test_keywords_match_sale_or_rent_with_ocr_typos(text, expected):
    assert match_keywords(text)["matched_roots"] == expected


@pytest.mark.parametrize("text", ["HATI HATI ANJING GALAK", "Informasi", "CENTURY 21 United", "(021) 296.88.676", "", "RUMAH DIKONTRAKKAN"])
def test_keywords_do_not_match_unrelated_text(text):
    # "dikontrakkan" tidak cocok karena akar kata terkunci "jual"/"sewa" (BAB III); kasus ini sengaja didokumentasikan.
    assert match_keywords(text)["matched"] is False


def test_excluded_words_skip_only_that_word():
    assert match_keywords("JUALAN SAYUR")["matched"] is False
    assert match_keywords("TERJUAL")["matched"] is False
    assert match_keywords("JUALAN dan DIJUAL")["matched_roots"] == ["jual"]  # kata lain di crop yang sama tetap dinilai


def _notebook_match(word, roots=("jual", "sewa"), thr=0.75):
    """Rumus pencocokan notebook apa adanya (partial_ratio pada setiap kata), untuk membuktikan celahnya."""
    from rapidfuzz import fuzz

    return [r for r in roots if fuzz.ratio(word, r) / 100 >= thr or fuzz.partial_ratio(word, r) / 100 >= thr]


@pytest.mark.parametrize("noise", ["a", "u", "e", "s", "se", "wa", "al", "ju"])
def test_short_ocr_noise_is_not_a_keyword_match(noise):
    assert _notebook_match(noise)  # celah pada notebook: derau pendek dianggap cocok
    assert match_keywords(noise)["matched"] is False  # diperbaiki di sini


def test_text_of_only_noise_letters_is_unidentified():
    assert match_keywords("a u e s wa")["matched"] is False


# --- crop ----------------------------------------------------------------------------------------------------------
def test_crop_adds_margin_upscales_small_crops_and_flags_manual_check():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    crop, manual = crop_notice(img, (100, 100, 200, 160))  # 100x60 -> margin 5%/5% -> 110x66 -> sisi pendek 66 <= 320
    assert manual is True
    assert min(crop.shape[:2]) == 320 and crop.shape[1] > crop.shape[0]


def test_large_crop_is_not_upscaled_nor_flagged():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    crop, manual = crop_notice(img, (100, 100, 700, 600))  # sisi pendek 500 + margin > 320
    assert manual is False and crop.shape[:2] == (550, 660)


def test_crop_is_clamped_to_image_and_empty_crop_is_none():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    crop, _ = crop_notice(img, (-50, -50, 450, 450))
    assert crop is not None
    assert crop_notice(img, (500, 500, 600, 600)) is None


# --- read_notices (OCR palsu) --------------------------------------------------------------------------------------
class _FakeOcr(NoticeOcr):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.calls = 0

    def read_crop(self, crop):
        self.calls += 1
        m = match_keywords(self.text)
        return NoticeOcrResult(LABEL_SALE_OR_RENT if m["matched"] else LABEL_UNIDENTIFIED, self.text, 0.8, list(m["matched_roots"]))


def _raw(cls, box):
    x1, y1, x2, y2 = box
    return RawDetection(cls, cls.split("_")[0], 0.9, x1 / 800, y1 / 800, (x2 - x1) / 800, (y2 - y1) / 800, xyxy=box)


def test_read_notices_only_for_house_notice_and_keys_are_detection_indexes():
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    dets = [_raw("pavedroad_pothole", (0, 0, 100, 100)), _raw("house_notice", (100, 100, 300, 300)), _raw("sign", (400, 400, 500, 500)), _raw("house_notice", (50, 50, 450, 450))]
    fake = _FakeOcr("DIJUAL")
    out = read_notices(img, dets, ocr=fake)
    assert sorted(out) == [1, 3] and fake.calls == 2
    assert out[1].label == LABEL_SALE_OR_RENT and out[1].needs_manual_check is True  # 200 px + margin <= 320
    assert out[3].needs_manual_check is False


def _cls(name, cid, **kw):
    return ClassDef(id=cid, name=name, visual_description="v", condition_criteria="c", feasibility_criteria="f", model_class=name, **kw)


def test_ocr_result_attached_only_to_classes_with_ocr_stage():
    dets = [_raw("house_notice", (0, 0, 100, 100))]
    ocr = {0: NoticeOcrResult(LABEL_SALE_OR_RENT, "DIJUAL", 0.61, ["jual"], True)}
    on = to_detection_items(dets, [_cls("house_notice", "c1", has_ocr_stage=True)], None, 0, ocr=ocr)[0]
    assert (on.ocr_label, on.ocr_text, on.ocr_confidence, on.ocr_matched_roots, on.ocr_manual_check) == (LABEL_SALE_OR_RENT, "DIJUAL", 0.61, ["jual"], True)
    assert on.ocr_model == "easyocr_id_en_keyword" and "teks notis" in on.condition
    off = to_detection_items(dets, [_cls("house_notice", "c1")], None, 0, ocr=ocr)[0]
    assert off.ocr_label is None and off.ocr_text is None and off.ocr_model is None and off.ocr_manual_check is False
    none = to_detection_items(dets, [_cls("house_notice", "c1", has_ocr_stage=True)], None, 0)[0]
    assert none.ocr_label is None  # OCR tidak dijalankan -> tidak dipalsukan sebagai tidak_teridentifikasi


# --- aktivasi ------------------------------------------------------------------------------------------------------
def test_ocr_disabled_unless_env_true(monkeypatch):
    monkeypatch.delenv("NOTICE_OCR_ENABLED", raising=False)
    assert NoticeOcr().enabled is False and NoticeOcr().status()["detail"] == "nonaktif"
    assert wants_ocr_stage([_cls("house_notice", "c1", has_ocr_stage=True)]) is False
    for v in ("true", "1", "ON", "Yes"):
        monkeypatch.setenv("NOTICE_OCR_ENABLED", v)
        assert NoticeOcr().enabled is True
    monkeypatch.setenv("NOTICE_OCR_ENABLED", "false")
    assert NoticeOcr().enabled is False


def test_stage_requested_only_by_house_notice_class_with_flag(monkeypatch):
    monkeypatch.setenv("NOTICE_OCR_ENABLED", "true")
    assert wants_ocr_stage([_cls("house_notice", "c1", has_ocr_stage=True)]) is True
    assert wants_ocr_stage([_cls("house_notice", "c1")]) is False
    assert wants_ocr_stage([_cls("sign", "c2", has_ocr_stage=True)]) is False  # bendera pada kelas lain tidak mengaktifkan OCR


def test_loading_while_disabled_fails_loudly(monkeypatch):
    monkeypatch.delenv("NOTICE_OCR_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="NOTICE_OCR_ENABLED"):
        NoticeOcr().load()


# --- bobot/model sungguhan (otomatis dilewati bila tidak ada) ---------------------------------------------------------
WEIGHTS = os.environ.get("YOLO_WEIGHTS_DIR")
SAMPLES = os.environ.get("YOLO_SAMPLE_DIR")
OCR_DIR = os.environ.get("NOTICE_OCR_MODEL_DIR")


def _real_ready():
    try:
        import easyocr  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        return False
    if not (WEIGHTS and SAMPLES and OCR_DIR):
        return False
    base = WEIGHTS if os.path.isabs(WEIGHTS) else os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS)
    return (
        os.path.isfile(os.path.join(base, "house_notice-best.pt"))
        and os.path.isfile(os.path.join(SAMPLES, "house_notice", "house_notice.jpg"))
        and os.path.isfile(os.path.join(OCR_DIR, "craft_mlt_25k.pth"))
        and os.path.isfile(os.path.join(OCR_DIR, "latin_g2.pth"))
    )


real = pytest.mark.skipif(not _real_ready(), reason="easyocr, bobot house_notice, citra contoh, atau model OCR (NOTICE_OCR_MODEL_DIR) tidak tersedia")


@real
def test_real_endpoint_reads_notice_text_on_sample(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setenv("YOLO_CONF", "0.25")
    monkeypatch.setenv("YOLO_DEVICE", "cpu")
    monkeypatch.setenv("NOTICE_OCR_ENABLED", "true")
    sample = os.path.join(SAMPLES, "house_notice", "house_notice.jpg")

    def body(flag):
        return {
            "session_id": "s", "media_asset_id": "m", "conflict_threshold": 0.5,
            "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "url": sample}],
            "active_classes": [{"id": "c-hn", "name": "house_notice", "visual_description": "v", "condition_criteria": "c", "feasibility_criteria": "f", "has_ocr_stage": flag}],
        }

    client = TestClient(main.app)
    headers = {"X-Internal-Secret": os.environ["INTERNAL_API_SECRET"]}
    r = client.post("/api/v1/yolo/detect", json=body(True), headers=headers).json()
    assert r["success"], r
    notices = [d for d in r["detections"] if d["class_name"] == "house_notice"]
    assert notices and all(d["ocr_label"] in {LABEL_SALE_OR_RENT, LABEL_UNIDENTIFIED} and d["ocr_model"] == "easyocr_id_en_keyword" for d in notices)
    assert any(d["ocr_label"] == LABEL_SALE_OR_RENT and "jual" in d["ocr_matched_roots"] for d in notices)  # papan "DIJUAL" pada citra contoh
    assert r["metrics"]["ocr_crops"] == len(notices) and r["metrics"]["ocr_ms"] > 0 and r["metrics"]["ocr_model"] == "easyocr_id_en_keyword"

    off = client.post("/api/v1/yolo/detect", json=body(False), headers=headers).json()  # kelas tanpa bendera -> tanpa OCR
    assert off["success"]
    assert all(d["ocr_label"] is None for d in off["detections"])
    assert off["metrics"]["ocr_crops"] == 0 and off["metrics"]["ocr_model"] is None
