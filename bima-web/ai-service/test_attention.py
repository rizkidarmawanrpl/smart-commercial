import os

os.environ.setdefault("INTERNAL_API_SECRET", "test-secret")
os.environ.setdefault("AI_SERVICE_ALLOWED_ORIGINS", "http://localhost:3000")

import numpy as np
import pytest
import torch

from services.attention_modules import (
    ATTENTION_CLASSES,
    CBAM,
    CBAMAttention,
    CoordAtt,
    ECAAttention,
    SEAttention,
    register_attention_modules,
)
from services.yolo_engine import CATEGORY_MODELS, YoloEngine
from tools.install_attention_weights import plan, variant_name


# --- modul ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("module_cls", [CBAM, CBAMAttention, SEAttention, ECAAttention, CoordAtt])
@pytest.mark.parametrize("channels,hw", [(64, (80, 80)), (128, (40, 52)), (256, (20, 20))])
def test_attention_preserves_shape_and_scales_input(module_cls, channels, hw):
    m = module_cls(channels).eval()
    x = torch.randn(2, channels, *hw)
    y = m(x)
    assert y.shape == x.shape
    assert not torch.equal(y, x)  # modul benar-benar memodulasi fitur


def test_eca_kernel_follows_channel_count():
    # Pada checkpoint: 64 kanal -> kernel 3; 128 dan 256 kanal -> kernel 5.
    assert [ECAAttention(c).conv.kernel_size[0] for c in (64, 128, 256)] == [3, 5, 5]


def test_cbam_and_se_reduction_matches_checkpoints():
    assert CBAM(64).channel_attention.mlp[0].out_channels == 4  # 64/16, seperti di checkpoint banner/house_notice
    assert SEAttention(256).fc[0].out_features == 16
    assert CoordAtt(64).conv1.out_channels == 8  # reduksi minimum 8


def test_register_puts_classes_in_main_without_overwriting():
    import sys

    main = sys.modules["__main__"]
    sentinel = object()
    saved = {n: getattr(main, n) for n in ATTENTION_CLASSES if hasattr(main, n)}
    try:
        for n in ATTENTION_CLASSES:
            if hasattr(main, n):
                delattr(main, n)
        main.CBAM = sentinel  # nama yang sudah ada tidak boleh ditimpa
        register_attention_modules()
        assert main.CBAM is sentinel
        assert all(getattr(main, n) is ATTENTION_CLASSES[n] for n in ATTENTION_CLASSES if n != "CBAM")
    finally:
        for n in ATTENTION_CLASSES:
            if hasattr(main, n):
                delattr(main, n)
        for n, v in saved.items():
            setattr(main, n, v)


# --- resolusi varian (tanpa memuat model) --------------------------------------------------------------------------
def _touch(folder, cats):
    folder.mkdir(parents=True, exist_ok=True)
    for c in cats:
        (folder / f"{c}-best.pt").write_bytes(b"x")


@pytest.fixture
def layout(tmp_path, monkeypatch):
    base = tmp_path / "yolo11n_seed0"
    _touch(base, CATEGORY_MODELS)
    _touch(tmp_path / "yolo11n_attn_CBAM_seed0", ["banner", "pavedroad"])
    monkeypatch.setenv("YOLO_WEIGHTS_DIR", str(base))
    monkeypatch.delenv("YOLO_VARIANTS_DIR", raising=False)
    monkeypatch.delenv("YOLO_BASELINE_MODEL", raising=False)
    return tmp_path


def test_no_name_and_baseline_name_resolve_to_baseline(layout):
    eng = YoloEngine()
    assert eng.resolve(None)[0] == "yolo11n_seed0" and eng.resolve(None)[2] is True
    assert eng.resolve("yolo11n_seed0") == eng.resolve(None)
    assert eng.status()["ok"] is True and eng.status()["fallback"] == []


def test_variant_status_reports_own_fallback_and_missing(layout):
    st = YoloEngine().status("yolo11n_attn_CBAM_seed0")
    assert st["is_baseline"] is False
    assert st["present"] == ["pavedroad", "banner"]
    assert set(st["fallback"]) == {"vegetation", "weeds", "sign", "house_notice"}  # dilayani baseline, tidak hilang diam-diam
    assert st["missing"] == [] and st["ok"] is True


def test_variant_missing_in_both_is_reported_as_missing(layout):
    (layout / "yolo11n_seed0" / "weeds-best.pt").unlink()
    st = YoloEngine().status("yolo11n_attn_CBAM_seed0")
    assert st["missing"] == ["weeds"] and st["ok"] is False


def test_unknown_variant_fails_loudly(layout):
    with pytest.raises(RuntimeError, match="tidak ditemukan"):
        YoloEngine().status("yolo11n_attn_XYZ_seed0")


@pytest.mark.parametrize("bad", ["../yolo11n_seed0", "a/b", "..", ".hidden", "x y"])
def test_variant_name_cannot_escape_variants_dir(layout, bad):
    with pytest.raises(RuntimeError, match="tidak valid"):
        YoloEngine().resolve(bad)


def test_baseline_name_and_variants_dir_can_be_overridden(layout, monkeypatch):
    other = layout / "elsewhere"
    _touch(other / "v1", ["sign"])
    monkeypatch.setenv("YOLO_BASELINE_MODEL", "yolo11n_seed0_label")
    monkeypatch.setenv("YOLO_VARIANTS_DIR", str(other))
    eng = YoloEngine()
    assert eng.resolve("yolo11n_seed0_label")[2] is True
    assert eng.status("v1")["present"] == ["sign"]
    with pytest.raises(RuntimeError, match="tidak ditemukan"):
        eng.status("yolo11n_attn_CBAM_seed0")  # ada di folder induk lama, bukan di YOLO_VARIANTS_DIR


# --- pemasang bobot ------------------------------------------------------------------------------------------------
def test_installer_maps_training_folders_to_variant_names(tmp_path):
    for cat, mod in [("banner", "CBAM"), ("banner", "SEAttention"), ("sign", "ECAAttention"), ("weeds", "CoordAtt")]:
        d = tmp_path / cat / f"attn_{mod}_seed0_img640"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
    (tmp_path / "banner" / "notes.txt").write_text("bukan bobot")
    got = sorted(rel.replace(os.sep, "/") for _, _, rel, _ in plan([str(tmp_path)]))
    assert got == [
        "yolo11n_attn_CBAM_seed0/banner-best.pt",
        "yolo11n_attn_CoordAtt_seed0/weeds-best.pt",
        "yolo11n_attn_ECA_seed0/sign-best.pt",
        "yolo11n_attn_SE_seed0/banner-best.pt",
    ]
    assert variant_name("SEAttention", "0") == "yolo11n_attn_SE_seed0"


def test_installer_skips_unknown_category(tmp_path, capsys):
    d = tmp_path / "bogus" / "attn_CBAM_seed0_img640"
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"x")
    assert plan([str(tmp_path)]) == []
    assert "bogus" in capsys.readouterr().err


# --- bobot sungguhan (otomatis dilewati bila tidak ada) ------------------------------------------------------------
WEIGHTS = os.environ.get("YOLO_WEIGHTS_DIR")
SAMPLES = os.environ.get("YOLO_SAMPLE_DIR")  # folder berisi <kategori>/<kategori>.jpg
VARIANTS = ["yolo11n_attn_CBAM_seed0", "yolo11n_attn_SE_seed0", "yolo11n_attn_ECA_seed0", "yolo11n_attn_CoordAtt_seed0"]
ALL = list(CATEGORY_MODELS)
PARTIAL = ["banner", "house_notice", "pavedroad"]  # varian parsial buatan untuk menguji fallback ke baseline


def _variants_ready():
    if not WEIGHTS:
        return False
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        return False
    base = WEIGHTS if os.path.isabs(WEIGHTS) else os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS)
    parent = os.path.dirname(base.rstrip("/\\"))
    return all(os.path.isfile(os.path.join(parent, v, f"{c}-best.pt")) for v in VARIANTS for c in ALL) and all(
        os.path.isfile(os.path.join(base, f"{c}-best.pt")) for c in ALL
    )


@pytest.fixture
def real_env(monkeypatch):
    monkeypatch.setenv("YOLO_CONF", "0.25")
    monkeypatch.setenv("YOLO_DEVICE", "cpu")


@pytest.fixture
def partial_variants(tmp_path, monkeypatch):
    """Folder varian berisi hanya 3 kategori (tautan ke bobot nyata), agar jalur fallback tetap teruji setelah semua kategori lengkap."""
    base = WEIGHTS if os.path.isabs(WEIGHTS) else os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS)
    real_parent = os.path.dirname(base.rstrip("/\\"))
    for v in VARIANTS:
        (tmp_path / v).mkdir()
        for c in PARTIAL:
            os.symlink(os.path.join(real_parent, v, f"{c}-best.pt"), tmp_path / v / f"{c}-best.pt")
    monkeypatch.setenv("YOLO_VARIANTS_DIR", str(tmp_path))
    from services.yolo_engine import engine as shared_engine

    monkeypatch.setattr(shared_engine, "_sets", {})  # cache singleton (dipakai main.app) berkunci nama model, bukan folder
    return tmp_path


@pytest.mark.skipif(not _variants_ready(), reason="bobot baseline/varian atau ultralytics tidak tersedia")
@pytest.mark.parametrize("variant", VARIANTS)
def test_real_variant_has_all_six_categories_with_baseline_class_names(variant, real_env):
    eng = YoloEngine()
    base = eng.load(None)
    ms = eng.load(variant)
    assert sorted(ms.own) == sorted(ALL) and ms.fallback == [] and ms.missing == []
    for cat in ALL:  # nama kelas sama dengan baseline, sehingga pemetaan ke ClassDefinition tidak berubah
        assert ms.models[cat].names == base.models[cat].names
        assert ms.models[cat] is not base.models[cat]


@pytest.mark.skipif(not _variants_ready(), reason="bobot baseline/varian atau ultralytics tidak tersedia")
@pytest.mark.parametrize("variant", VARIANTS)
def test_real_partial_variant_falls_back_to_the_same_baseline_models(variant, real_env, partial_variants):
    eng = YoloEngine()
    base = eng.load(None)
    ms = eng.load(variant)
    assert sorted(ms.own) == sorted(PARTIAL)
    assert set(ms.fallback) == set(ALL) - set(PARTIAL) and ms.missing == []
    for cat in ms.fallback:  # fallback memakai objek model baseline yang sama, bukan salinan
        assert ms.models[cat] is base.models[cat]


@pytest.mark.skipif(not (_variants_ready() and SAMPLES), reason="bobot atau YOLO_SAMPLE_DIR tidak tersedia")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "category,expected",
    [("banner", "banner"), ("house_notice", "house_notice"), ("pavedroad", "pavedroad_pothole"), ("sign", "sign"), ("weeds", "weeds")],
)
def test_real_variant_detects_expected_class_on_sample(variant, category, expected, real_env):
    import cv2

    img = cv2.imread(os.path.join(SAMPLES, category, f"{category}.jpg"))
    assert img is not None
    dets, timings = YoloEngine().detect(img, model_name=variant)
    assert set(timings.per_model_ms) == set(CATEGORY_MODELS)
    assert expected in {d.model_class for d in dets if d.source_model == category}
    assert {d.served_by for d in dets} == {variant}  # semua kotak tercatat atas nama varian


@pytest.mark.skipif(not (_variants_ready() and SAMPLES), reason="bobot atau YOLO_SAMPLE_DIR tidak tersedia")
@pytest.mark.parametrize("variant", VARIANTS[:1])
def test_real_partial_variant_records_baseline_as_server_of_fallback_boxes(variant, real_env, partial_variants):
    import cv2

    img = cv2.imread(os.path.join(SAMPLES, "sign", "sign.jpg"))  # sign tidak ada di varian parsial -> baseline
    dets, _ = YoloEngine().detect(img, model_name=variant)
    sign = [d for d in dets if d.source_model == "sign"]
    assert sign and {d.served_by for d in sign} == {"yolo11n_seed0"}


@pytest.mark.skipif(not _variants_ready(), reason="bobot baseline/varian atau ultralytics tidak tersedia")
def test_real_baseline_unchanged_when_variant_also_loaded(real_env):
    eng = YoloEngine()
    img = np.full((480, 640, 3), 127, dtype=np.uint8)
    before, _ = eng.detect(img)
    eng.detect(img, model_name=VARIANTS[0])
    after, t = eng.detect(img)
    assert [(d.model_class, round(d.confidence, 4)) for d in before] == [(d.model_class, round(d.confidence, 4)) for d in after]
    assert set(t.per_model_ms) == set(CATEGORY_MODELS)


def _body(sample, model_name=None):
    body = {
        "session_id": "s", "media_asset_id": "m", "conflict_threshold": 0.5,
        "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "url": sample}],
        "active_classes": [
            {"id": "c-banner", "name": "banner", "visual_description": "v", "condition_criteria": "c", "feasibility_criteria": "f"},
        ],
    }
    if model_name is not None:
        body["model_name"] = model_name
    return body


def _client_and_headers():
    from fastapi.testclient import TestClient
    import main

    return TestClient(main.app), {"X-Internal-Secret": os.environ["INTERNAL_API_SECRET"]}


@pytest.mark.skipif(not (_variants_ready() and SAMPLES), reason="bobot atau YOLO_SAMPLE_DIR tidak tersedia")
def test_real_endpoint_reports_variant_and_keeps_baseline(real_env):
    client, headers = _client_and_headers()
    sample = os.path.join(SAMPLES, "banner", "banner.jpg")

    attn = client.post("/api/v1/yolo/detect", json=_body(sample, "yolo11n_attn_CBAM_seed0"), headers=headers).json()
    assert attn["success"], attn
    assert attn["metrics"]["model_name"] == "yolo11n_attn_CBAM_seed0" and attn["metrics"]["fallback_models"] == []
    banners = [d for d in attn["detections"] if d["class_name"] == "banner"]
    assert banners and all(d["served_by"] == "yolo11n_attn_CBAM_seed0" for d in banners)

    base = client.post("/api/v1/yolo/detect", json=_body(sample), headers=headers).json()  # tanpa model_name = baseline
    assert base["success"], base
    assert base["metrics"]["model_name"] == "yolo11n_seed0" and base["metrics"]["fallback_models"] == []

    bad = client.post("/api/v1/yolo/detect", json=_body(sample, "yolo11n_attn_NOPE_seed0"), headers=headers).json()
    assert bad["success"] is False and "tidak ditemukan" in bad["error_message"]  # galat jelas, bukan jatuh ke baseline

    conn = client.post(
        "/api/v1/test-connection",
        json={"ai_model_config": {"provider": "yolo", "model_name": "yolo11n_attn_CBAM_seed0"}},
        headers=headers,
    ).json()
    assert conn["success"] and "yolo11n_attn_CBAM_seed0" in conn["message"] and "dilayani baseline" not in conn["message"]


@pytest.mark.skipif(not (_variants_ready() and SAMPLES), reason="bobot atau YOLO_SAMPLE_DIR tidak tersedia")
def test_real_endpoint_reports_fallback_for_partial_variant(real_env, partial_variants):
    client, headers = _client_and_headers()
    sample = os.path.join(SAMPLES, "banner", "banner.jpg")
    r = client.post("/api/v1/yolo/detect", json=_body(sample, "yolo11n_attn_CBAM_seed0"), headers=headers).json()
    assert r["success"], r
    assert set(r["metrics"]["fallback_models"]) == set(ALL) - set(PARTIAL)
    conn = client.post(
        "/api/v1/test-connection",
        json={"ai_model_config": {"provider": "yolo", "model_name": "yolo11n_attn_CBAM_seed0"}},
        headers=headers,
    ).json()
    assert conn["success"] and "dilayani baseline" in conn["message"]
