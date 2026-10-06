"""Memasang bobot ablasi attention dari zip/folder hasil training ke folder varian yang dibaca ai-service.

Struktur sumber (satu zip per kategori, seperti dari Google Drive):
    banner/attn_CBAM_seed0_img640/best.pt
    banner/attn_SEAttention_seed0_img640/best.pt
    ...
Hasil (folder varian berdampingan dengan baseline yolo11n_seed0, nama folder = nama model di menu Model AI):
    ai-service/models/yolo11n_attn_CBAM_seed0/banner-best.pt
    ai-service/models/yolo11n_attn_SE_seed0/banner-best.pt

Pemakaian (dari folder ai-service):
    python tools/install_attention_weights.py ~/Downloads/banner.zip ~/Downloads/house_notice.zip ~/Downloads/pavedroad.zip
    python tools/install_attention_weights.py ~/Downloads/weeds.zip --dest models   # kategori berikutnya, tambahkan saja
Opsi --dry-run hanya menampilkan rencana. Berkas yang sudah ada ditimpa hanya dengan --overwrite.
"""
import argparse
import os
import re
import shutil
import sys
import zipfile
from typing import Iterator, List, Tuple

CATEGORIES = ("pavedroad", "vegetation", "weeds", "sign", "banner", "house_notice")
# Nama modul pada nama folder training -> label singkat di nama folder varian.
MODULE_LABEL = {"CBAM": "CBAM", "SEAttention": "SE", "ECAAttention": "ECA", "CoordAtt": "CoordAtt"}
_ENTRY = re.compile(r"(?:^|/)(?P<cat>[A-Za-z_]+)/attn_(?P<mod>[A-Za-z]+)_seed(?P<seed>\d+)_img\d+/best\.pt$")


def variant_name(module: str, seed: str) -> str:
    return f"yolo11n_attn_{MODULE_LABEL[module]}_seed{seed}"


def _entries(source: str) -> Iterator[Tuple[str, str]]:
    """(path relatif dengan '/', penanda sumber) untuk setiap best.pt di zip atau folder."""
    if os.path.isdir(source):
        for root, _, files in os.walk(source):
            for f in files:
                full = os.path.join(root, f)
                yield os.path.relpath(full, source).replace(os.sep, "/"), full
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for n in z.namelist():
                yield n, n
    else:
        raise SystemExit(f"Sumber bukan zip atau folder: {source}")


def plan(sources: List[str]) -> List[Tuple[str, str, str, str]]:
    """Daftar (sumber, penanda, tujuan relatif, ringkasan). Entri yang tidak cocok pola dilaporkan, bukan dipasang diam-diam."""
    out: List[Tuple[str, str, str, str]] = []
    for src in sources:
        found = 0
        for rel, marker in _entries(src):
            m = _ENTRY.search(rel)
            if not m:
                continue
            cat, mod, seed = m.group("cat"), m.group("mod"), m.group("seed")
            if cat not in CATEGORIES:
                print(f"  lewati {rel}: kategori '{cat}' bukan salah satu dari {', '.join(CATEGORIES)}", file=sys.stderr)
                continue
            if mod not in MODULE_LABEL:
                print(f"  lewati {rel}: modul '{mod}' belum dikenal (tambahkan di MODULE_LABEL dan services/attention_modules.py)", file=sys.stderr)
                continue
            out.append((src, marker, os.path.join(variant_name(mod, seed), f"{cat}-best.pt"), f"{cat} / {mod}"))
            found += 1
        if found == 0:
            print(f"  peringatan: tidak ada best.pt bernama attn_<modul>_seed<n>_img<n>/best.pt di {src}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="zip atau folder hasil training (satu per kategori)")
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
                    help="folder induk varian (sama dengan YOLO_VARIANTS_DIR; bawaan: ai-service/models)")
    ap.add_argument("--overwrite", action="store_true", help="timpa berkas yang sudah ada")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = plan(args.sources)
    if not items:
        print("Tidak ada bobot yang dikenali.", file=sys.stderr)
        return 1
    installed = skipped = 0
    for src, marker, rel_dest, label in items:
        dest = os.path.join(args.dest, rel_dest)
        if os.path.exists(dest) and not args.overwrite:
            print(f"  ada      {rel_dest}  ({label}) -> lewati, pakai --overwrite untuk menimpa")
            skipped += 1
            continue
        print(f"  {'rencana ' if args.dry_run else 'pasang  '} {rel_dest}  ({label})")
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isdir(src):
            shutil.copyfile(marker, dest)
        else:
            with zipfile.ZipFile(src) as z, z.open(marker) as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        installed += 1
    print(f"Selesai: {installed} dipasang, {skipped} dilewati di {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
