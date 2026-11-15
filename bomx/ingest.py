"""Load drawings into page images plus, where the file carries it, exact text.

Supported inputs
  * PDF  - digital (vector text layer), digital with vectorised (SHX) text,
           and scanned (image-only) pages, in any mix within one file
  * DXF  - native CAD text, or raster-to-vector traced drawings (OCR)
  * DWG  - converted to DXF in memory, then as DXF
  * TIFF (multi-page) / PNG / JPG / BMP / WEBP scans
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image, ImageSequence

from .models import Word

PDF_EXT = {".pdf"}
IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CAD_EXT = {".dxf", ".dwg"}
SUPPORTED_EXT = PDF_EXT | IMAGE_EXT | CAD_EXT

# Decompression bomb protection: cap at 100MP to accommodate large A0/E-size sheets safely
Image.MAX_IMAGE_PIXELS = 100_000_000


# Marker put in page notes / warnings when content is known to be incomplete
# (partial CAD conversion, OCR-only fallback, failed page ...). The backend
# uses it to tell complete from partial processing.
PARTIAL_TAG = "[partial]"

_EXT_FORMAT = {".pdf": "pdf", ".dwg": "dwg", ".dxf": "dxf", ".tif": "tif", ".tiff": "tif",
               ".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".bmp": "bmp", ".webp": "webp"}
IMAGE_FORMATS = {"png", "jpg", "tif", "bmp", "webp"}


def sniff_bytes(header: bytes) -> str:
    """Identify a drawing format from its first bytes only ("unknown" if no
    signature matches)."""
    import re
    if not header:
        return "unknown"
    # PDF signature may follow a few junk bytes (allowed by the spec: first 1 KB)
    if header.startswith(b"%PDF-") or b"%PDF-" in header[:1024]:
        return "pdf"
    # DWG: "AC10xx" version string (AC1012 .. AC1032)
    if re.match(rb"AC10\d\d", header[:6]) or header.startswith(b"MC0.0"):
        return "dwg"
    if header.startswith(b"AutoCAD Binary DXF"):
        return "dxf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if header.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "tif"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    # ASCII DXF: group code 0 followed by SECTION (optionally after 999 comments)
    text_head = header[:2048].decode("latin1", errors="ignore").lstrip("﻿")
    if re.search(r"^\s*0\s*\r?\n\s*SECTION\s*$", text_head, re.MULTILINE):
        return "dxf"
    return "unknown"


def sniff_format(path: str | Path) -> str:
    """Identify drawing file format from magic bytes; falls back to the
    extension only when the content carries no recognisable signature."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            header = f.read(2048)
    except Exception:
        header = b""
    fmt = sniff_bytes(header)
    if fmt != "unknown":
        return fmt
    return _EXT_FORMAT.get(p.suffix.lower(), "unknown")


def validate_format(path: str | Path, header: Optional[bytes] = None) -> str:
    """Return the validated format of `path` or raise ValueError for empty,
    corrupt or mismatched files (extension says one thing, content another)."""
    p = Path(path)
    if header is None:
        try:
            with open(p, "rb") as f:
                header = f.read(2048)
        except OSError as e:
            raise ValueError(f"cannot read file '{p.name}': {e}") from e
    if not header:
        raise ValueError(f"file '{p.name}' is empty")
    ext_fmt = _EXT_FORMAT.get(p.suffix.lower())
    fmt = sniff_bytes(header)
    if fmt == "unknown":
        if ext_fmt is None:
            raise ValueError(f"unsupported file type: '{p.suffix or p.name}'")
        raise ValueError(f"file '{p.name}' does not look like a valid {ext_fmt.upper()} file "
                         "(missing or corrupt file signature)")
    if ext_fmt is not None and ext_fmt != fmt:
        raise ValueError(f"file '{p.name}' has extension '{p.suffix}' but its content is "
                         f"{fmt.upper()} - rename the file or re-export it")
    return fmt


@dataclass
class RawPage:
    index: int
    image: np.ndarray  # BGR uint8
    dpi: float
    native_words: Optional[list[Word]] = None  # exact text (digital PDF / CAD)
    notes: list[str] = field(default_factory=list)
    kind: str = ""  # pdf-vector | pdf-scan | pdf-outlined-text | image | dxf | dwg (+ "-traced")
    label: str = ""  # e.g. CAD layout name
    cad_fields: dict = field(default_factory=dict)  # title-block attributes from CAD blocks
    fmt: str = ""  # validated source format: pdf | png | jpg | tif | bmp | webp | dxf | dwg
    # Hybrid pages: native words are kept but OCR is still needed, either for
    # the whole page (needs_ocr) or only for these pixel regions (x0, y0, x1, y1).
    needs_ocr: bool = False
    ocr_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    page_size: tuple[float, float] = (0.0, 0.0)  # original page size (PDF points / CAD units)
    rotation: int = 0  # source page rotation already applied to image and words (degrees)
    partial: bool = False  # content known to be incomplete


def iter_pages(path: str | Path, dpi: int = 300, max_pixels: int = 80_000_000,
               force_ocr: bool = False, native_min_words: int = 15,
               cfg: Optional[dict] = None) -> Iterator[RawPage]:
    path = Path(path)
    cfg = cfg or {}
    fmt = validate_format(path)  # content decides the parser, not the extension
    if fmt == "pdf":
        it = _iter_pdf(path, dpi, max_pixels, force_ocr, native_min_words, cfg)
    elif fmt in IMAGE_FORMATS:
        it = _iter_image(path, dpi, max_pixels)
    elif fmt in {"dxf", "dwg"}:
        from .cad import iter_cad_pages
        it = iter_cad_pages(path, dpi, max_pixels, force_ocr, native_min_words, cfg)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix or fmt}")
    for page in it:
        page.fmt = fmt
        if not page.page_size[0]:
            h, w = page.image.shape[:2]
            page.page_size = (float(w), float(h))
        if any(PARTIAL_TAG in n for n in page.notes):
            page.partial = True
        yield page


# --------------------------------------------------------------------- PDF

def _word_ok(text: str) -> bool:
    """A native PDF word is usable if it has letters/digits and no signs of a
    broken font encoding (replacement / private-use / control characters)."""
    if not any(c.isalnum() for c in text):
        return False
    bad = sum(1 for c in text if c == "�" or "" <= c <= ""
              or (ord(c) < 32 and c not in "\t"))
    return bad == 0


def _assess_pdf_text(words: list[Word], native_min_words: int, min_alpha: float) -> tuple[str, list[Word], str]:
    """Classify a page's text layer: ("native" | "hybrid" | "none", usable words, note).
    native_min_words is one input, not the sole criterion: a page with fewer
    words keeps them (hybrid) and OCR fills the rest, and a long but garbled
    text layer is not trusted."""
    if not words:
        return "none", [], ""
    good = [w for w in words if _word_ok(w.text)]
    meaningful = [w for w in words if any(c.isalnum() for c in w.text)]
    if not good:
        return "none", [], f"PDF text layer ({len(words)} objects) is unreadable - using OCR"
    quality = len(good) / max(1, len(meaningful))
    alpha = sum(c.isalpha() for w in good for c in w.text)
    tot = max(1, sum(len(w.text) for w in good))
    alpha_ratio = alpha / tot
    if quality < 0.6:
        return "hybrid", good, (f"PDF text layer partly garbled ({len(good)}/{len(meaningful)} "
                                "words readable) - keeping readable words, OCR for the rest")
    if alpha_ratio < min_alpha and len(good) < 40:
        return "hybrid", good, "PDF text layer appears degraded/noisy - keeping it and adding OCR"
    if len(good) < native_min_words:
        return "hybrid", good, (f"only {len(good)} text objects in PDF layer (text may be "
                                "vectorised) - keeping them and adding OCR")
    return "native", good, ""


def _pdf_image_regions(page, words_pt: list[Word], min_area_frac: float) -> list[tuple[float, float, float, float]]:
    """Raster images on a page with a native text layer that themselves carry
    no native text (pasted scans, embedded title blocks ...). Points, displayed
    (rotated) page coordinates."""
    import pymupdf
    page_area = abs(page.rect)
    if page_area <= 0:
        return []
    try:
        infos = page.get_image_info()
    except Exception:
        return []
    rot = page.rotation_matrix
    out = []
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        r = pymupdf.Rect(bbox) * rot
        x0, x1 = min(r.x0, r.x1), max(r.x0, r.x1)
        y0, y1 = min(r.y0, r.y1), max(r.y0, r.y1)
        if (x1 - x0) * (y1 - y0) < min_area_frac * page_area:
            continue
        inside = sum(1 for w in words_pt
                     if x0 <= (w.x0 + w.x1) / 2 <= x1 and y0 <= (w.y0 + w.y1) / 2 <= y1)
        if inside < 3:
            out.append((x0, y0, x1, y1))
    return out


def _iter_pdf(path, dpi, max_pixels, force_ocr, native_min_words, cfg=None):
    import pymupdf
    cfg = cfg or {}
    min_alpha = float(cfg.get("native_min_alpha_ratio", 0.20))
    min_region = float(cfg.get("ocr_region_min_area", 0.02))

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise ValueError(f"PDF is damaged and cannot be opened: {e}") from e
    try:
        if doc.needs_pass:
            if not doc.authenticate(""):
                raise ValueError("PDF is password protected")
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages")
        for i, page in enumerate(doc):  # each page is assessed on its own
            notes: list[str] = []
            page_size = (float(page.rect.width), float(page.rect.height))
            rotation = int(page.rotation or 0)
            if rotation:
                notes.append(f"page rotated {rotation} deg - image and text coordinates use the displayed orientation")
            try:
                words, _ = _pdf_words(page) if not force_ocr else ([], 1.0)
            except Exception as e:
                words = []
                notes.append(f"PDF text layer could not be read ({e}) - using OCR")
            mode, good, why = ("none", [], "") if force_ocr else \
                _assess_pdf_text(words, native_min_words, min_alpha)
            if why:
                notes.append(why)

            scan_dpi = _scan_resolution(page) if mode != "native" else None
            if scan_dpi is not None:
                # Image-only page: render at the scan's own resolution (no
                # resampling blur) and judge scale from the lettering, exactly
                # as for a TIFF/PNG. The PDF page size of a scan often says
                # nothing about the drawing's real size.
                render = min(float(scan_dpi), 600.0)
                img = _render_pdf(page, render, max_pixels, notes)
                rgb = img[:, :, ::-1]
                img, eff_dpi = _normalise_raster(rgb, scan_dpi, dpi, max_pixels, notes,
                                                 already_capped=True)
                yield RawPage(i, img, eff_dpi, None, notes, kind="pdf-scan",
                              page_size=page_size, rotation=rotation)
                continue

            w_in, h_in = page.rect.width / 72, page.rect.height / 72
            eff_dpi = _cap_dpi(dpi, w_in, h_in, max_pixels)
            if eff_dpi < dpi:
                notes.append(f"render DPI reduced to {eff_dpi:.0f} to respect max_pixels")
            img = _render_pdf(page, eff_dpi, max_pixels, notes)
            zoom = eff_dpi / 72
            native = [Word(w.text, w.x0 * zoom, w.y0 * zoom, w.x1 * zoom, w.y1 * zoom, 1.0)
                      for w in good]
            if mode == "native":
                regions = [(x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)
                           for x0, y0, x1, y1 in _pdf_image_regions(page, good, min_region)]
                if regions:
                    notes.append(f"{len(regions)} raster region(s) without native text - OCR applied there")
                yield RawPage(i, img, eff_dpi, native, notes,
                              kind="pdf-mixed" if regions else "pdf-vector",
                              ocr_regions=regions, page_size=page_size, rotation=rotation)
            elif mode == "hybrid":
                yield RawPage(i, img, eff_dpi, native, notes, kind="pdf-mixed", needs_ocr=True,
                              page_size=page_size, rotation=rotation)
            else:
                yield RawPage(i, img, eff_dpi, None, notes,
                              kind="pdf-outlined-text" if not force_ocr else "pdf-vector",
                              page_size=page_size, rotation=rotation)
    finally:
        doc.close()


def _pdf_words(page):
    import pymupdf
    raw = page.get_text("words")
    rot = page.rotation_matrix
    words = []
    for x0, y0, x1, y1, txt, *_ in raw:
        txt = txt.strip()
        if not txt:
            continue
        r = pymupdf.Rect(x0, y0, x1, y1) * rot  # unrotated -> displayed coords (points)
        bx0, bx1 = min(r.x0, r.x1), max(r.x0, r.x1)
        by0, by1 = min(r.y0, r.y1), max(r.y0, r.y1)
        words.append(Word(txt, bx0, by0, bx1, by1, 1.0))
    return words, 1.0


def _scan_resolution(page) -> Optional[float]:
    """If the page is essentially one raster image (a scan), return that
    image's resolution in pixels per inch of the page; else None."""
    page_area = abs(page.rect)
    if page_area <= 0:
        return None
    best_area, best_dpi = 0.0, None
    try:
        infos = page.get_image_info()
    except Exception:
        return None
    for info in infos:
        bbox = info.get("bbox")
        w, h = info.get("width") or 0, info.get("height") or 0
        if not bbox or not w or not h:
            continue
        x0, y0, x1, y1 = bbox
        bw, bh = abs(x1 - x0), abs(y1 - y0)
        area = bw * bh
        if area <= best_area or bw < 1 or bh < 1:
            continue
        # images may be rotated on the page: compare the long sides
        res = max(w, h) / (max(bw, bh) / 72)
        best_area, best_dpi = area, res
    if best_dpi is None or best_area < 0.5 * page_area:
        return None
    # A digital drawing with a small logo is not a scan; a scan with a few
    # vector annotations still is.
    try:
        n_vec = len(page.get_drawings())
    except Exception:
        n_vec = 0
    if n_vec > 200:
        return None
    return float(best_dpi)


def _render_pdf(page, dpi, max_pixels, notes):
    import pymupdf
    w_in, h_in = page.rect.width / 72, page.rect.height / 72
    d = _cap_dpi(dpi, w_in, h_in, max_pixels)
    if d < dpi - 0.5 and not any("max_pixels" in n for n in notes):
        notes.append(f"render DPI reduced to {d:.0f} to respect max_pixels")
    zoom = d / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()  # RGB -> BGR


# ------------------------------------------------------------------ images

def _iter_image(path, dpi, max_pixels):
    try:
        im = Image.open(str(path))
        im.verify()  # cheap integrity check; verify() invalidates the handle
        im.close()
    except Image.DecompressionBombError:
        raise
    except Exception as e:
        raise ValueError(f"image file '{Path(path).name}' is corrupt or truncated: {e}") from e
    with Image.open(str(path)) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            notes: list[str] = []
            info_dpi = frame.info.get("dpi")
            src_dpi = float(info_dpi[0]) if info_dpi and info_dpi[0] and info_dpi[0] > 30 else None
            w, h = frame.size
            downscaled = False
            if max_pixels and (w * h > max_pixels):
                scale = (max_pixels / (w * h)) ** 0.5
                target_w = max(1, int(w * scale))
                target_h = max(1, int(h * scale))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.ANTIALIAS if hasattr(Image, "ANTIALIAS") else None)
                frame = frame.resize((target_w, target_h), resample=resample) if resample else frame.resize((target_w, target_h))
                if src_dpi:
                    src_dpi *= scale
                downscaled = True
                notes.append("image downscaled prior to decode to respect max_pixels")
            rgb = np.asarray(frame.convert("RGB"))
            arr, eff = _normalise_raster(rgb, src_dpi, dpi, max_pixels, notes, already_capped=downscaled)
            yield RawPage(i, arr, eff, None, notes, kind="image")


def _normalise_raster(rgb: np.ndarray, src_dpi: Optional[float], dpi: int, max_pixels: int,
                      notes: list[str], already_capped: bool = False) -> tuple[np.ndarray, float]:
    """Decide the effective resolution of a raster drawing and cap its size.
    72/96 dpi (or none) is a screen value, not the drawing's scale: measure the
    lettering instead. Also catch metadata that is far off."""
    from .imaging import estimate_dpi_from_text
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    est = estimate_dpi_from_text(gray)
    if src_dpi is None or src_dpi < 150 or (est and not 0.5 <= est / src_dpi <= 2.0):
        if est:
            if src_dpi:
                notes.append(f"image resolution {src_dpi:.0f} dpi does not match the lettering "
                             f"size - using estimated {est:.0f} dpi")
            src_dpi = est
        elif src_dpi is None:
            src_dpi = float(dpi)
            notes.append(f"image has no DPI metadata - assuming {dpi}")
    h, w = rgb.shape[:2]
    if not already_capped and max_pixels and w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        src_dpi *= s
        notes.append("image downscaled to respect max_pixels")
    return np.ascontiguousarray(rgb[:, :, ::-1]), float(src_dpi)


def _cap_dpi(dpi, w_in, h_in, max_pixels):
    px = (w_in * dpi) * (h_in * dpi)
    if px <= max_pixels:
        return float(dpi)
    return float((max_pixels / (w_in * h_in)) ** 0.5)


def discover_inputs(paths: list[str], recursive: bool = True) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            it = pp.rglob("*") if recursive else pp.glob("*")
            out.extend(sorted(f for f in it if f.is_file() and f.suffix.lower() in SUPPORTED_EXT))
        elif pp.is_file():
            out.append(pp)
        else:
            raise FileNotFoundError(p)
    # de-duplicate, keep order
    seen, uniq = set(), []
    for f in out:
        k = f.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq
