"""End-to-end processing of one drawing file."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import imaging
from .dimensions import dedupe_dimensions, extract_dimensions_from_words
from .extract import (latest_revision, normalise_title_fields, parse_title_block, table_to_bom,
                      table_to_revisions)
from .ingest import PARTIAL_TAG, RawPage, iter_pages
from .layout import (Table, assign_words, complete_rows, detect_cells, find_grid_tables, find_notes,
                     find_text_tables, group_cells, split_numbered_notes, text_title_cells,
                     title_block_cells)
from .models import (Cell, Dimension, DrawingResult, FieldValue, PageResult, Region, SpecCallout,
                     Word, mean_conf, union_bbox)
from .specs import find_specs, summarise
from .textutil import fix_ocr_confusions, group_lines, split_segments

log = logging.getLogger(__name__)
_NUMERIC = re.compile(r"\d+(?:[.,]\d+)?")


def _rotate_word_frame(word: Word, rotation: int, width: int, height: int) -> Word:
    """Map a vertical OCR word into a horizontal table-analysis frame."""
    x0, y0, x1, y1 = word.bbox
    if rotation == 90:
        # cv2 ROTATE_90_CLOCKWISE: (x, y) -> (H - y, x)
        pts = [(height - y0, x0), (height - y1, x0),
               (height - y0, x1), (height - y1, x1)]
    elif rotation == 270:
        # cv2 ROTATE_90_COUNTERCLOCKWISE: (x, y) -> (y, W - x)
        pts = [(y0, width - x0), (y1, width - x0),
               (y0, width - x1), (y1, width - x1)]
    else:  # 180 degrees
        pts = [(width - x0, height - y0), (width - x1, height - y0),
               (width - x0, height - y1), (width - x1, height - y1)]
    nx0, nx1 = min(p[0] for p in pts), max(p[0] for p in pts)
    ny0, ny1 = min(p[1] for p in pts), max(p[1] for p in pts)
    return Word(word.text, nx0, ny0, nx1, ny1, word.conf)


def _map_rotated_bbox(bbox, rotation: int, width: int, height: int):
    if not bbox:
        return bbox
    x0, y0, x1, y1 = bbox
    if rotation == 90:
        # inverse of x'=H-y, y'=x
        pts = [(y0, height - x0), (y1, height - x0), (y0, height - x1), (y1, height - x1)]
    elif rotation == 270:
        # inverse of x'=y, y'=W-x
        pts = [(width - y0, x0), (width - y1, x0), (width - y0, x1), (width - y1, x1)]
    else:
        pts = [(width - x0, height - y0), (width - x1, height - y0),
               (width - x0, height - y1), (width - x1, height - y1)]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def _rotate_cell_frame(cell: Cell, rotation: int, width: int, height: int) -> Cell:
    proxy = _rotate_word_frame(Word("", *cell.bbox, 1.0), rotation, width, height)
    bb = proxy.bbox
    mapped_words = [_rotate_word_frame(word, rotation, width, height) for word in cell.words]
    return Cell(*bb, words=mapped_words, id=cell.id, parent=cell.parent)


def _restore_rotated_table(table, rotation: int, width: int, height: int) -> None:
    for col in table.columns:
        normalized_height = width if rotation in (90, 270) else height
        bb = _map_rotated_bbox((col.x0, 0, col.x1, normalized_height),
                               rotation, width, height)
        # Keep x0/x1 in the normalized frame for any downstream table logic;
        # source geometry is explicitly stored because sideways columns map to
        # source Y-ranges rather than source X-ranges.
        col.source_bbox = bb
    table.bbox = _map_rotated_bbox(table.bbox, rotation, width, height)
    table.header_bbox = _map_rotated_bbox(table.header_bbox, rotation, width, height)
    table.row_bboxes = [_map_rotated_bbox(bb, rotation, width, height) for bb in table.row_bboxes]
    for row in table.rows:
        for cell in row.values():
            cell.bbox = _map_rotated_bbox(cell.bbox, rotation, width, height)
_AS_REQ = re.compile(r"^\s*(A/?R|AS\s+REQ\w*|REF|AR)\s*\.?$", re.I)


class PageAnalysis:
    """Everything learned about one page; kept for overlays / debugging."""

    def __init__(self, raw: RawPage):
        self.raw = raw
        self.image: np.ndarray = raw.image
        self.words: list[Word] = []
        self.cells = []
        self.bom_tables: list[Table] = []
        self.rev_tables: list[Table] = []
        self.tb_cells = []
        self.notes_blocks = []
        self.specs: list[SpecCallout] = []
        self.result: Optional[PageResult] = None


def analyse_page(raw: RawPage, cfg: dict) -> PageAnalysis:
    pa = PageAnalysis(raw)
    dpi = raw.dpi
    # hybrid pages (some native words, rest needs OCR) take the OCR path and
    # merge their native words in; fully native pages may still OCR regions
    native = raw.native_words is not None and not raw.needs_ocr
    has_native = bool(raw.native_words)
    img = raw.image
    gray = imaging.to_gray(img)
    rotation, skew = 0, 0.0

    if (not native or raw.ocr_regions) and imaging.tesseract_available() is None:
        if native:  # regions only: keep the native text, report what is missing
            raw.notes.append(f"{PARTIAL_TAG} Tesseract not found - {len(raw.ocr_regions)} raster "
                             "region(s) without native text were not read")
            raw.ocr_regions = []
            raw.partial = True
        elif has_native:
            raw.notes.append(f"{PARTIAL_TAG} Tesseract not found - only the native PDF text "
                             "was used; OCR-only content is missing")
            native, raw.partial = True, True
        else:
            raise RuntimeError(
                "OCR needed but Tesseract was not found: this page has no readable text layer "
                "(scanned drawing, or CAD text exported as lines). Install Tesseract OCR "
                "(Windows: 'UB Mannheim Tesseract') or give its path with --tesseract / UI Settings. "
                "Run 'python -m bomx check' to test.")

    if not native:
        # Native word coordinates are exact for the page as rendered: rotating
        # or deskewing the image would misplace them, so only do it when the
        # page has no native words.
        if has_native:
            pass
        elif cfg.get("auto_rotate", True):
            rotation = imaging.detect_orientation(gray)
            if rotation:
                gray = imaging.rotate_quadrant(gray, rotation)
                img = imaging.rotate_quadrant(img, rotation)
        binv = imaging.binarize(gray, True, dpi)
        if cfg.get("deskew", True) and not has_native:
            skew = imaging.estimate_skew(binv, dpi)
            if abs(skew) >= 0.05:
                gray = imaging.rotate_small(gray, skew)
                img = imaging.rotate_small(img, skew)
                binv = imaging.binarize(gray, True, dpi)
    else:
        binv = imaging.binarize(gray, False, dpi)
    if raw.kind.endswith("-traced"):  # faint rules survive the trace only as dashes
        binv, specks = imaging.bridge_dashed_rules(binv, dpi)
        gray = gray.copy()
        gray[specks > 0] = 255  # rule remnants are not text: keep them from OCR
    pa.image = img

    hm, vm = imaging.line_masks(binv, dpi)
    hm, vm = imaging.close_open_table_sides(hm, vm, dpi)  # tables cut off at the sheet edge
    grid = hm | vm
    cells, frame = detect_cells(grid, dpi)

    if native:
        words = list(raw.native_words)
        if raw.ocr_regions:
            words += _ocr_regions(img, gray, grid, raw, dpi, cfg, words)
        free = assign_words(cells, words)
        clean = None
    else:
        clean = imaging.remove_lines(gray, grid)
        scanned = raw.kind in ("pdf-scan", "image") or raw.kind.endswith("-raster")
        enh = cfg.get("scan_enhance", "auto")
        if scanned and enh not in (False, "off", "none"):
            clean, why = imaging.enhance_scan(clean, dpi, force=(enh == "always"))
            if why:
                raw.notes.append(why)
        words = imaging.ocr_image(clean, dpi, cfg["ocr_lang"], psm=11, min_conf=cfg["min_word_conf"])
        if cfg.get("rotated_text_ocr", True):
            rot_words = imaging.ocr_rotated(clean, dpi, cfg["ocr_lang"], cfg["min_word_conf"],
                                            existing=words)
            if rot_words:
                raw.notes.append(f"{len(rot_words)} vertical (rotated) text line(s) read")
                words += rot_words
        if raw.native_words:
            n_boxes = [w.bbox for w in raw.native_words]
            words = list(raw.native_words) + [
                w for w in words
                if not any(_box_overlap(w.bbox, nb) > 0.4 for nb in n_boxes)
            ]
        free = assign_words(cells, words)
        if cfg.get("cell_ocr", True) and cells:
            groups0 = group_cells(cells, dpi)
            # cells holding only vertical text keep it: an upright re-read would garble it
            targets = [c for g in groups0 if len(g) >= 3 for c in g
                       if not (c.words and all(w.y1 - w.y0 > 1.2 * (w.x1 - w.x0) and len(w.text) > 1
                                               for w in c.words))]
            cell_workers = int(cfg.get("ocr_workers", 1 if cfg.get("_parallel") else 4))
            redo = imaging.ocr_cells(clean, targets, dpi, cfg["ocr_lang"], cfg["min_word_conf"],
                                     workers=cell_workers)
            for c in targets:
                new = redo.get(c.id) or []
                if new and (not c.words or mean_conf(new) >= mean_conf(c.words) - 0.05
                            or len("".join(w.text for w in new)) > len("".join(w.text for w in c.words))):
                    c.words = new
            words = free + [w for c in cells for w in c.words]
    pa.words, pa.cells = words, cells

    groups = group_cells(cells, dpi)
    used: set[int] = set()
    bom_tables, rev_tables = [], []
    for g in groups:
        bom_tables += find_grid_tables(g, cfg["bom_headers"], "bom", dpi, used, cfg["title_labels"])
    for g in groups:
        rev_tables += find_grid_tables(g, cfg["revision_headers"], "revision", dpi, used,
                                       cfg["title_labels"])
    tb_cells = title_block_cells(groups, used, cfg["title_labels"])
    tb_ids = {c.id for c in tb_cells}

    def outside(boxes):
        return [w for w in words if not any(_inside(w, b) for b in boxes)]

    # Borderless / partly ruled tables are searched for even when a ruled table
    # was found: a sheet can carry several BOMs (or a ruled revision table next
    # to a borderless BOM) anywhere on the page.
    pool = outside([t.bbox for t in bom_tables + rev_tables])
    extra_bom = [t for t in find_text_tables(pool, cfg["bom_headers"], "bom", dpi)
                 if not any(_box_overlap(t.bbox, o.bbox) > 0.3 for o in bom_tables + rev_tables)]
    if extra_bom:
        bom_tables += extra_bom
        # the text table may have been swallowed by a title-block group
        tb_cells = [c for c in tb_cells
                    if not any(_box_overlap(c.bbox, t.bbox) > 0.5 for t in extra_bom)]
    # Sideways BOM fallback: normalize vertical OCR geometry, reuse the
    # borderless table detector, then restore source-page bounding boxes.
    page_h, page_w = img.shape[:2]
    existing_boxes = [t.bbox for t in bom_tables + rev_tables]
    for rotation in (90, 180, 270):
        vertical = ([w for w in words if w.h >= 1.2 * max(w.x1 - w.x0, 1.0)]
                    if rotation in (90, 270) else list(words))
        normalized = [_rotate_word_frame(w, rotation, page_w, page_h) for w in vertical]
        normalized_cells = [_rotate_cell_frame(c, rotation, page_w, page_h) for c in cells]
        for group in group_cells(normalized_cells, dpi):
            for rotated_table in find_grid_tables(group, cfg["bom_headers"], "bom", dpi,
                                                  set(), cfg["title_labels"]):
                _restore_rotated_table(rotated_table, rotation, page_w, page_h)
                rotated_table.method = "rotated_grid"
                if not any(_box_overlap(rotated_table.bbox, old) > 0.3 for old in existing_boxes):
                    bom_tables.append(rotated_table)
                    existing_boxes.append(rotated_table.bbox)
        for rotated_table in find_text_tables(normalized, cfg["bom_headers"], "bom", dpi):
            _restore_rotated_table(rotated_table, rotation, page_w, page_h)
            rotated_table.method = "rotated_text"
            if not any(_box_overlap(rotated_table.bbox, old) > 0.3 for old in existing_boxes):
                bom_tables.append(rotated_table)
                existing_boxes.append(rotated_table.bbox)
    tb_box = [c.bbox for c in tb_cells]
    pool = outside([t.bbox for t in bom_tables + rev_tables] + tb_box)
    rev_tables += [t for t in find_text_tables(pool, cfg["revision_headers"], "revision", dpi)
                   if not any(_box_overlap(t.bbox, o.bbox) > 0.3 for o in bom_tables + rev_tables)]

    if not tb_cells:  # title-block rules too broken to form cells: read it from text layout
        tb_cells = text_title_cells(outside([t.bbox for t in bom_tables + rev_tables]),
                                    cfg["title_labels"], dpi)
    else:  # part of the title block may not have closed into cells
        tbb = union_bbox([c.bbox for c in tb_cells])
        m = dpi * 0.5
        in_cells = {id(w) for c in cells for w in c.words}
        near = [w for w in outside([t.bbox for t in bom_tables + rev_tables])
                if id(w) not in in_cells and tbb[0] - m <= w.cx <= tbb[2] + m
                and tbb[1] - m <= w.cy <= tbb[3] + m]
        if near:
            tb_cells = tb_cells + text_title_cells(near, cfg["title_labels"], dpi, min_fields=1)
    for t in bom_tables + rev_tables:  # cells whose rules were lost entirely
        complete_rows(t, free)
        if not native and cfg.get("cell_ocr", True):
            _reread_reconstructed(t, clean, dpi, cfg)
    pa.bom_tables, pa.rev_tables, pa.tb_cells = bom_tables, rev_tables, tb_cells
    if not native and clean is not None:
        for t in bom_tables:
            t.clean_image = clean
            t.dpi = dpi
            t.lang = cfg.get("ocr_lang", "eng")

    if not native:  # digit-constrained re-read of numeric columns that don't look numeric or have low confidence
        for t in bom_tables:
            if t.method != "grid":
                continue
            for row in t.rows:
                for key in ("item", "quantity", "mass"):
                    v = row.get(key)
                    if v is None or not v.text:
                        continue
                    if key != "item" and (re.search(r"\d", v.text) and v.conf >= cfg["low_conf_threshold"] or _AS_REQ.match(v.text)):
                        continue  # '4 NOS', '1.2 kg', 'A/R' - leave to the normaliser
                    if _NUMERIC.fullmatch(v.text) and v.conf >= cfg["low_conf_threshold"]:
                        continue
                    got = imaging.ocr_numeric(clean, v.bbox, dpi, cfg["ocr_lang"], orig_text=v.text)
                    if got and _NUMERIC.fullmatch(got[0]):
                        if not _NUMERIC.fullmatch(v.text) or got[1] > v.conf:
                            v.text, v.conf = got[0], max(v.conf, got[1])
                        else:
                            v.text, v.conf = got[0], max(min(v.conf, got[1]), 0.5)

    if not native:  # preserve original revision text; correct BOM fields only
        for t in bom_tables:
            for row in t.rows:
                for v in row.values():
                    if v.text:
                        v.raw_text = v.text
                        v.text = fix_ocr_confusions(v.text)
        for t in bom_tables:  # a lost leading zero: '1' among '02', '03', ...
            items = [r.get("item") for r in t.rows if r.get("item") is not None and r["item"].text]
            padded = [v for v in items if re.fullmatch(r"0\d", v.text)]
            if len(items) >= 3 and len(padded) >= len(items) - 1:
                for v in items:
                    if re.fullmatch(r"[1-9]", v.text):
                        v.text = "0" + v.text

    table_boxes = [t.bbox for t in bom_tables + rev_tables]
    tb_bbox = union_bbox([c.bbox for c in tb_cells]) if tb_cells else None
    free_words = [w for w in words if not any(_inside(w, b) for b in table_boxes)
                  and not (tb_bbox and _inside(w, tb_bbox))]
    if frame is not None:  # zone markers etc. between the outer and inner border
        free_words = [w for w in free_words if _inside(w, frame)]
    pa.notes_blocks = find_notes(free_words, dpi)
    if not native and pa.notes_blocks:
        masks = [t.bbox for t in bom_tables + rev_tables] + ([tb_bbox] if tb_bbox else [])
        pa.notes_blocks = [_reread_notes(lines, bb, clean, dpi, cfg, masks)
                           for lines, bb in pa.notes_blocks]

    regions = []
    if tb_cells:
        regions.append(Region("title_block", tb_bbox, raw.index,
                              _avg([c.conf for c in tb_cells if c.words]), tb_cells))
    for t in bom_tables:
        regions.append(Region("bom", t.bbox, raw.index, _avg([v.conf for r in t.rows for v in r.values()
                                                              if v.text])))
    for t in rev_tables:
        regions.append(Region("revision", t.bbox, raw.index,
                              _avg([v.conf for r in t.rows for v in r.values() if v.text])))
    for lines, bb in pa.notes_blocks:
        regions.append(Region("notes", bb, raw.index, _avg([l.conf for l in lines])))

    h, w = img.shape[:2]
    source_h, source_w = raw.image.shape[:2]
    source = "native" if native and not raw.ocr_regions else ("hybrid" if has_native else "ocr")
    pa.result = PageResult(index=raw.index, source=source, width=w,
                           height=h, dpi=round(dpi, 1), skew_deg=round(skew, 3),
                           rotation_deg=rotation, regions=regions, kind=raw.kind, layout=raw.label,
                           source_format=raw.fmt, status="partial" if raw.partial else "complete",
                           page_size=tuple(round(float(v), 2) for v in raw.page_size),
                           source_width=source_w, source_height=source_h)
    pa.free_words = free_words
    return pa


class _Box:
    """Minimal cell-like rectangle for imaging.ocr_cells."""

    def __init__(self, i, x0, y0, x1, y1):
        self.id, self.x0, self.y0, self.x1, self.y1 = i, x0, y0, x1, y1


def _reread_reconstructed(t: Table, clean, dpi: float, cfg: dict) -> None:
    """Values whose cell had to be reconstructed (row or column rule lost,
    or no cell at all) were read from a crop covering several rows/columns,
    or from the page-level OCR. Re-read each as its exact row x column box."""
    if t.method != "grid" or clean is None:
        return
    jobs = {}
    for ri, (row, rb) in enumerate(zip(t.rows, t.row_bboxes)):
        if not any(v.src != "cell" for v in row.values()):
            continue
        for ci, col in enumerate(t.columns):
            v = row.get(col.key)
            if v is not None and v.src == "cell" and v.text:
                continue
            jobs[len(jobs)] = (row, col.key, _Box(len(jobs), col.x0, rb[1], col.x1, rb[3]))
    if not jobs:
        return
    cell_workers = int(cfg.get("ocr_workers", 1 if cfg.get("_parallel") else 4))
    got = imaging.ocr_cells(clean, [b for *_, b in jobs.values()], dpi, cfg["ocr_lang"],
                            cfg["min_word_conf"], workers=cell_workers)
    from .layout import _cell_text, TableCellValue
    for i, (row, key, box) in jobs.items():
        ws = got.get(i) or []
        txt = _cell_text(ws)
        if not txt:
            continue
        old = row.get(key)
        conf = mean_conf(ws)
        # the exact cell box beats text pieced together from a merged crop
        if old is None or not old.text or conf >= old.conf - 0.1 or \
                (old.src != "cell" and conf >= 0.5 and len(txt) >= len(old.text)):
            row[key] = TableCellValue(txt, conf, union_bbox([w.bbox for w in ws]), "reread")


_NOTE_NUM = re.compile(r"^\s*\d{1,2}\s*[.)\]]\s")


def _reread_notes(lines, bb, clean, dpi: float, cfg: dict, masks):
    """Compare sparse and block OCR for numbering and technical-text quality."""
    from .layout import _NOTES_HEAD, _NOTES_HEAD_INLINE
    if clean is None:
        return lines, bb
    old_num = sum(1 for l in lines if _NOTE_NUM.match(l.text))
    pad = int(dpi * 0.04)
    H, W = clean.shape[:2]
    x0, y0 = max(int(bb[0]) - pad, 0), max(int(bb[1]) - pad, 0)
    x1, y1 = min(int(bb[2]) + pad, W), min(int(bb[3]) + pad, H)
    crop = clean[y0:y1, x0:x1].copy()
    for m in masks:  # keep neighbouring tables out of the block
        a0, b0 = max(int(m[0]) - x0, 0), max(int(m[1]) - y0, 0)
        a1, b1 = min(int(m[2]) - x0, crop.shape[1]), min(int(m[3]) - y0, crop.shape[0])
        if a1 > a0 and b1 > b0:
            crop[b0:b1, a0:a1] = 255
    candidates = [lines]
    for psm in (6, 11):
        ws = imaging.ocr_image(crop, dpi, cfg["ocr_lang"], psm=psm,
                               min_conf=cfg["min_word_conf"], offset=(x0, y0))
        candidate = [l for l in group_lines(ws)
                     if not (_NOTES_HEAD.match(l.text) or _NOTES_HEAD_INLINE.match(l.text))]
        if candidate:
            candidates.append(candidate)

    def quality(candidate):
        words = [w for line in candidate for w in line.words]
        if not words:
            return -1.0
        txt = " ".join(line.text for line in candidate)
        count = sum(ch.isalnum() for ch in txt)
        old_count = max(1, sum(ch.isalnum() for line in lines for ch in line.text))
        coverage = min(1.0, count / old_count)
        token_quality = sum(bool(re.search(r"[A-Za-z]{2,}|\d", w.text)) for w in words) / len(words)
        num_count = sum(1 for line in candidate if _NOTE_NUM.match(line.text))
        numbering = min(1.0, num_count / max(old_num, 1))
        return (0.68 * sum(w.conf for w in words) / len(words) +
                0.17 * coverage + 0.10 * token_quality + 0.05 * numbering)

    old_alnum = sum(ch.isalnum() for line in lines for ch in line.text)
    viable = [candidate for candidate in candidates
              if sum(1 for line in candidate if _NOTE_NUM.match(line.text)) >= old_num and
              sum(ch.isalnum() for line in candidate for ch in line.text) >= 0.85 * old_alnum]
    if not viable:
        return lines, bb
    best = max(viable, key=quality)
    return best, bb


def _extract_page_content(pa: PageAnalysis, raw: RawPage, cfg: dict, page_no: int,
                          title: dict[str, FieldValue], bom_cols: list[str],
                          res: DrawingResult, all_specs: list[SpecCallout],
                          all_dims: list[Dimension]) -> None:
    """Stage 3: Extract structured records (title block, BOM, revisions, notes, specs, dimensions)."""
    pr = pa.result

    # ---------------- title block
    # fallback text for the title block: never revision-history or BOM rows
    # (a historical 'REV A' must not become the current drawing revision)
    tbl_boxes = [t.bbox for t in pa.rev_tables + pa.bom_tables]
    page_text = "\n".join(l.text for l in group_lines(
        [w for w in pa.words if not any(_inside(w, b) for b in tbl_boxes)]))
    tb = parse_title_block(pa.tb_cells, cfg["title_labels"], raw.dpi, page_text, cfg["date_order"])
    # Unit declarations are also commonly printed just outside the title
    # block. Keep the matched word segment's bbox so the value remains locatable.
    if not (tb.get("units") and tb["units"].value):
        unit_rx = re.compile(r"\b(?:M[E3]TR[I1L]C|IMPERIAL)\s*\(\s*(?:MM|CM|M|INCH(?:ES)?|IN)\s*\)|"
                             r"\bDIMENSIONS?\s+IN\s+(?:MM|CM|M|INCH(?:ES)?|IN)\b", re.I)
        title_box = union_bbox([c.bbox for c in pa.tb_cells]) if pa.tb_cells else None
        excluded = [t.bbox for t in pa.bom_tables + pa.rev_tables]
        pad = max(raw.dpi * 0.5, 150.0)
        unit_words = [w for w in pa.words
                      if not any(_inside(w, box) for box in excluded)
                      and (title_box is None or
                           title_box[0] - pad <= w.cx <= title_box[2] + pad and
                           title_box[1] - pad <= w.cy <= title_box[3] + pad)]
        unit_candidates = []
        for line in group_lines(unit_words):
            match = unit_rx.search(line.text)
            if match:
                unit_candidates.append((match.group(0), round(line.conf, 3), line.bbox, line.text))
            for seg in split_segments(line, 2.0):
                match = unit_rx.search(seg.text)
                if match:
                    unit_candidates.append((match.group(0), round(seg.conf, 3), seg.bbox, seg.text))
        if unit_candidates:
            def unit_score(item):
                _, conf, bbox, _ = item
                inside = bool(title_box and title_box[0] <= bbox[0] and bbox[2] <= title_box[2] and
                              title_box[1] <= bbox[1] and bbox[3] <= title_box[3])
                return (1 if inside else 0, conf)
            chosen = max(unit_candidates, key=unit_score)
            values = {item[0].upper().replace(" ", "") for item in unit_candidates}
            tb["units"] = FieldValue(chosen[0], chosen[1], chosen[2], raw=chosen[3],
                                      source="page_text", page=page_no,
                                      candidates=[{"value": v, "bbox": b, "confidence": c}
                                                  for v, c, b, _ in unit_candidates],
                                      needs_review=len(values) > 1)
    # Dates physically inside a detected revision-history table are history
    # entries, not the drawing's title-block date. Preserve them as candidates
    # but require review if region selection has included the integrated box.
    for date_key in ("date", "drawn_date", "checked_date", "approved_date"):
        fv = tb.get(date_key)
        if not fv or not fv.bbox:
            continue
        x = (fv.bbox[0] + fv.bbox[2]) / 2
        y = (fv.bbox[1] + fv.bbox[3]) / 2
        if any(t.bbox[0] <= fv.bbox[0] and fv.bbox[2] <= t.bbox[2] and
               t.bbox[1] <= fv.bbox[1] and fv.bbox[3] <= t.bbox[3]
               for t in pa.rev_tables):
            fv.candidates.append(fv.to_dict())
            fv.needs_review = True
            fv.value = None
            fv.normalized = None
    _merge_cad_fields(tb, raw.cad_fields, cfg, res.warnings, page_no)
    for k, v in tb.items():
        if not v.value and not v.needs_review:
            continue
        v.page = page_no
        current = title.get(k)
        if current is None:
            title[k] = v
            continue
        previous_value = current.value
        previous_candidate = {name: getattr(current, name) for name in
                              ("value", "raw", "source", "page", "confidence", "bbox")}
        incoming_candidate = {name: getattr(v, name) for name in
                              ("value", "raw", "source", "page", "confidence", "bbox")}
        current.candidates.extend([previous_candidate, incoming_candidate])
        # A labelled cell or CAD attribute is stronger evidence than free page
        # text. Keep conflicts from independent title blocks visible for review.
        priority = {"title_block": 3, "cad_attribute": 3, "page_text": 1, "revision_table": 0}
        cur_rank = priority.get(current.source or ("title_block" if current.bbox else "page_text"), 1)
        new_rank = priority.get(v.source or ("title_block" if v.bbox else "page_text"), 1)
        if (new_rank, v.confidence) > (cur_rank, current.confidence):
            v.candidates = current.candidates + v.candidates
            v.needs_review = v.needs_review or current.needs_review
            title[k] = v
            current = v
        if v.needs_review:
            current.needs_review = True
            current.value = None
            current.normalized = None
        if previous_value and v.value and previous_value.strip().casefold() != v.value.strip().casefold():
            current.needs_review = True
            current.value = None
            current.normalized = None
            res.warnings.append(f"page {page_no}: conflicting title-block candidates for {k}; review required")

    # ---------------- BOM
    page_rows: list = []
    for ti, t in enumerate(pa.bom_tables):
        rows, warns = table_to_bom(t, page_no, cfg["low_conf_threshold"], table_index=ti,
                                   vocab=cfg["bom_headers"])
        # the same physical row read twice (overlapping table detections) is kept once
        kept, dup = [], 0
        for r in rows:
            if r.bbox and any(o.bbox and _box_overlap(r.bbox, o.bbox) > 0.6
                              and _box_overlap(o.bbox, r.bbox) > 0.6 for o in page_rows):
                dup += 1
                continue
            kept.append(r)
        if dup:
            warns.append(f"BOM table {ti + 1}: {dup} row(s) already extracted from another table - not duplicated")
        page_rows += kept
        res.bom.extend(kept)
        res.warnings.extend(f"page {page_no}: {w}" for w in warns)
        for c in t.columns:
            if c.key not in bom_cols:
                bom_cols.append(c.key)
        if t.method == "text":
            res.warnings.append(f"page {page_no}: BOM read from text alignment (no ruling lines) "
                                "- verify column assignment")

    # ---------------- revisions
    for t in pa.rev_tables:
        res.revisions.extend(table_to_revisions(t, page_no, cfg["date_order"]))

    # ---------------- notes & specs
    for lines, bb in pa.notes_blocks:
        for item in split_numbered_notes(lines):
            res.notes.append({"page": page_no, "number": item["number"], "text": item["text"],
                              "confidence": round(item["conf"], 3),
                              "bbox": [round(v, 1) for v in item["bbox"]]})
            all_specs += find_specs(item["text"], "notes", page_no, item["bbox"], item["conf"])
    for k in ("material", "finish", "heat_treatment"):
        fv = tb.get(k)
        if fv and fv.value:
            found = find_specs(fv.value, "title_block", page_no, fv.bbox, fv.confidence)
            if not found:
                cat = {"material": "material", "finish": "surface_finish",
                       "heat_treatment": "heat_treatment"}[k]
                found = [SpecCallout(cat, fv.value.upper(), fv.value, "title_block", "unrecognised",
                                     fv.confidence, fv.bbox, page_no)]
            all_specs += found
    for r in [r for r in res.bom if r.page == page_no]:
        if r.material:
            found = find_specs(r.material, "bom", page_no, r.bbox, r.confidence,
                               categories={"material", "heat_treatment", "standard"})
            if not found and len(r.material.strip()) >= 2:
                found = [SpecCallout("material", r.material.strip().upper(), r.material.strip(),
                                     "bom", "custom", r.confidence, r.bbox, page_no)]
            all_specs += found
        if r.extra.get("finish"):
            all_specs += find_specs(r.extra["finish"], "bom", page_no, r.bbox, r.confidence,
                                    categories={"surface_finish", "heat_treatment"})
    note_boxes = [bb for _, bb in pa.notes_blocks]
    loose = [w for w in pa.free_words if not any(_inside(w, b) for b in note_boxes)]
    for line in group_lines(loose):
        for seg in split_segments(line, 3.0):
            all_specs += find_specs(seg.text, "page_text", page_no, seg.bbox, seg.conf,
                                    categories={"material", "surface_finish", "heat_treatment"})

    # ---------------- dimensions
    all_dims += extract_dimensions_from_words(pa.words, page_no)


def _consolidate_and_validate(res: DrawingResult, title: dict[str, FieldValue],
                              bom_cols: list[str], all_specs: list[SpecCallout],
                              all_dims: list[Dimension], cfg: dict, t0: float) -> DrawingResult:
    """Stage 4: Consolidate multi-page findings, cross-check revisions, and build validated summary."""
    res.title_block = title
    res.bom_columns = bom_cols
    res.specs = _dedupe_specs(all_specs)
    res.dimensions = dedupe_dimensions(all_dims)
    latest = latest_revision(res.revisions)
    # The current revision is the title block's REV field; the revision table is
    # history. Its latest row is only a fallback, clearly marked as such.
    if "revision" not in title and latest:
        title["revision"] = FieldValue(latest.rev, round(min(latest.confidence, 0.85), 3), latest.bbox,
                                       raw=latest.rev, normalized=latest.rev, source="revision_table",
                                       page=latest.page, needs_review=True)
        res.warnings.append("no revision in title block - current revision inferred from latest "
                            "row of revision table")
    elif "revision" in title and latest and title["revision"].normalized and \
            latest.rev and latest.rev.upper() != str(title["revision"].normalized).upper():
        res.warnings.append(f"title block revision '{title['revision'].normalized}' differs from latest "
                            f"revision-table entry '{latest.rev}'")
    # Validation (Issue 6): evidence, completeness, duplicates, cross-checks.
    # Runs before the summary so rows flagged 'needs_review' are counted there.
    from .validate import validate_result
    res.validation = validate_result(res, cfg)
    res.summary = build_summary(res, cfg)
    v = res.validation
    res.summary["validation_errors"] = v["counts"]["error"]
    res.summary["validation_warnings"] = v["counts"]["warning"]
    res.summary["bom_completeness"] = v["completeness"]["bom"]
    res.summary["title_block_completeness"] = v["completeness"]["title_block"]
    if v["review_required"]:
        reasons = [res.summary.get("review_reasons")] if res.summary.get("review_reasons") else []
        reasons.append(f"validation: {v['counts']['error']} error(s), {v['counts']['warning']} warning(s)")
        res.summary["review_required"] = True
        res.summary["review_reasons"] = "; ".join(reasons)
    res.processing_seconds = time.time() - t0
    return res


def process_file(path: str | Path, cfg: dict, force_ocr: bool = False,
                 overlay_dir: Optional[Path] = None) -> DrawingResult:
    """Stage 5: Main orchestrator coordinating ingestion, analysis, extraction, and validation."""
    t0 = time.time()
    path = Path(path)
    imaging.find_tesseract(cfg.get("tesseract_cmd"))
    res = DrawingResult(source_file=str(path))
    title: dict[str, FieldValue] = {}
    all_specs: list[SpecCallout] = []
    all_dims: list[Dimension] = []
    bom_cols: list[str] = []

    # Stage 1: Ingestion & Iteration
    failed: list[str] = []
    n_pages = 0
    for raw in iter_pages(path, cfg["dpi"], cfg["max_pixels"], force_ocr, cfg["native_min_words"], cfg):
        n_pages += 1
        page_no = raw.index + 1
        # Stage 2: Page Layout & Detection - one bad page must not lose the others
        try:
            pa = analyse_page(raw, cfg)
        except Exception as e:
            log.exception("page %d of %s failed", page_no, path.name)
            res.warnings.extend(f"page {page_no}: {n}" for n in raw.notes)
            res.warnings.append(f"page {page_no}: {PARTIAL_TAG} page could not be processed: {e}")
            failed.append(str(e))
            h, w = raw.image.shape[:2]
            res.pages.append(PageResult(index=raw.index, source="none", width=w, height=h,
                                        dpi=round(raw.dpi, 1), kind=raw.kind, layout=raw.label,
                                        source_format=raw.fmt, status="failed",
                                        page_size=tuple(round(float(v), 2) for v in raw.page_size)))
            continue
        # notes are collected after analysis: it can add OCR / partial notes
        res.warnings.extend(f"page {page_no}: {n}" for n in raw.notes)
        pr = pa.result
        res.pages.append(pr)

        # Stage 3: Extraction
        _extract_page_content(pa, raw, cfg, page_no, title, bom_cols, res, all_specs, all_dims)

        if overlay_dir is not None:
            from .overlay import render_page
            overlay_dir.mkdir(parents=True, exist_ok=True)
            out = overlay_dir / f"{path.stem}_p{page_no}_overlay.png"
            render_page(pa, [s for s in all_specs if s.page == page_no],
                        [r for r in res.bom if r.page == page_no], out)
            pr.overlay_path = str(out)

    if n_pages and len(failed) == n_pages:
        raise RuntimeError(failed[0] if len(set(failed)) == 1 else "; ".join(dict.fromkeys(failed)))

    # Stage 4: Consolidation & Validation
    return _consolidate_and_validate(res, title, bom_cols, all_specs, all_dims, cfg, t0)


def _merge_cad_fields(tb: dict[str, FieldValue], cad_fields: dict, cfg: dict,
                      warnings: list[str], page_no: int) -> None:
    """Title-block values stored as CAD block attributes (exact, tagged)
    fill fields the layout analysis did not find. Where both exist and
    disagree, the layout value is kept and the difference reported."""
    if not cad_fields:
        return
    add: dict[str, FieldValue] = {}
    for k, d in cad_fields.items():
        val = (d.get("value") or "").strip()
        if not val:
            continue
        bb = tuple(d["bbox"]) if d.get("bbox") else None
        cad_value = FieldValue(val, 1.0, bb, raw=f"{d.get('tag')}={val}",
                               source="cad_attribute", page=page_no)
        existing = tb.get(k)
        if existing and (existing.value or existing.needs_review):
            existing.candidates.append(cad_value.to_dict())
            old = re.sub(r"[^A-Z0-9]", "", (existing.value or "").upper())
            new = re.sub(r"[^A-Z0-9]", "", val.upper())
            if existing.needs_review or (existing.value and old != new and new not in old):
                existing.needs_review = True
                existing.value = None
                existing.normalized = None
                warnings.append(f"page {page_no}: conflicting title-block and CAD candidates for {k}; review required")
            continue
        add[k] = cad_value
    normalise_title_fields(add, cfg["date_order"])
    tb.update(add)


def build_summary(res: DrawingResult, cfg: dict) -> dict:
    tb = res.title_block

    def v(k, norm=False):
        f = tb.get(k)
        if not f:
            return None
        return (f.normalized or f.value) if norm else f.value

    s = {
        "file": Path(res.source_file).name,
        "pages": len(res.pages),
        "input_type": ",".join(sorted({p.source for p in res.pages})),
        "file_format": Path(res.source_file).suffix.lstrip(".").upper() or None,
        "page_kinds": ",".join(sorted({p.kind for p in res.pages if p.kind})) or None,
        "drawing_number": v("drawing_number", True),
        "drawing_title": v("drawing_title"),
        "revision": v("revision", True),
        "drawn_by": v("drawn_by"),
        "date": v("date", True) or v("date"),
        "scale": v("scale", True) or v("scale"),
        "projection": v("projection", True) or v("projection"),
        "sheet": v("sheet"),
        "checked_by": v("checked_by"),
        "approved_by": v("approved_by"),
        "revision_source": tb["revision"].source if tb.get("revision") else None,
    }
    s["author_date"] = " / ".join(x for x in (s["drawn_by"], s["date"]) if x) or None
    if "author_date" not in tb and s["author_date"]:
        auth_values = [tb[k].confidence for k in ("drawn_by", "date")
                       if k in tb and tb[k].value]
        if auth_values:
            tb["author_date"] = FieldValue(s["author_date"], min(auth_values), None,
                                            raw=s["author_date"], source="derived")
    s.update(summarise(res.specs))
    # A note or title-block field that is explicitly labelled MATERIAL /
    # SURFACE FINISH / HEAT TREATMENT is the drawing's own callout: report it
    # as written rather than the fragments the pattern library recognises.
    for key, val in labelled_callouts(res).items():
        s[key] = val
    if tb.get("weight"):
        s["assembly_weight"] = tb["weight"].value
        s["assembly_weight_kg"] = tb["weight"].normalized
    s["bom_items"] = len(res.bom)
    qtys = [r.quantity_value for r in res.bom if isinstance(r.quantity_value, (int, float))]
    s["bom_total_quantity"] = sum(qtys) if qtys else None
    s["revisions"] = len(res.revisions)
    s["notes"] = len(res.notes)

    reasons = []
    for k in ("drawing_number", "drawing_title", "revision"):
        if not tb.get(k):
            reasons.append(f"missing {k}")
    lc = cfg["low_conf_threshold"]
    low_tb = [k for k, f in tb.items() if f.confidence < lc]
    if low_tb:
        reasons.append("low-confidence title fields: " + ", ".join(sorted(low_tb)))
    flagged = [r for r in res.bom if r.flags and set(r.flags) - {"mass_unit_assumed_kg"}]
    if flagged:
        reasons.append(f"{len(flagged)} BOM row(s) flagged")
    if not res.bom:
        reasons.append("no BOM table found")

    # Calculate text confidence independently across recognized text elements
    tb_confs = [f.confidence for f in tb.values() if f.confidence is not None]
    bom_text_confs = [r.confidence for r in res.bom if r.confidence is not None]
    notes_confs = [n["confidence"] for n in res.notes if "confidence" in n and n["confidence"] is not None]
    rev_confs = [r.confidence for r in res.revisions if r.confidence is not None]
    dim_confs = [d.confidence for d in res.dimensions if d.confidence is not None]
    text_confs = tb_confs + bom_text_confs + notes_confs + rev_confs + dim_confs

    s["text_confidence"] = round(sum(text_confs) / len(text_confs), 3) if text_confs else None

    # Structural confidence across BOM table layout
    struct_confs = [r.structural_confidence for r in res.bom]
    s["structural_confidence"] = round(sum(struct_confs) / len(struct_confs), 3) if struct_confs else 1.0

    # Overall combined mean confidence
    all_confs = text_confs + struct_confs
    s["mean_confidence"] = round(sum(all_confs) / len(all_confs), 3) if all_confs else None

    # Avoid hiding a bad row behind a high average by evaluating critical minimums
    min_row_struct_conf = min(struct_confs) if struct_confs else 1.0
    min_row_text_conf = min(bom_text_confs) if bom_text_confs else 1.0
    s["min_structural_confidence"] = round(min_row_struct_conf, 3)
    s["min_text_confidence"] = round(min(text_confs), 3) if text_confs else None

    if min_row_struct_conf < lc:
        reasons.append(f"low structural / column assignment confidence on row (min {min_row_struct_conf:.2f})")
    elif s["structural_confidence"] < lc:
        reasons.append(f"low mean structural confidence ({s['structural_confidence']:.2f})")

    if min_row_text_conf < lc:
        reasons.append(f"low text OCR confidence on row (min {min_row_text_conf:.2f})")

    s["review_required"] = bool(reasons)
    s["review_reasons"] = "; ".join(reasons) or None
    return s


_CALLOUT_LABELS = [
    ("material_callout", r"MATERIAL\s+CALL[\s-]?OUT|MATERIAL\s+SPEC(?:IFICATION)?|MATERIALS?|MAT'?L"),
    ("surface_finish", r"SURFACE\s+FINISH|SURFACE\s+TREATMENT|SURFACE\s+ROUGHNESS|FINISH|COATING"),
    ("heat_treatment", r"HEAT\s+TREATMENT|HEAT\s+TREAT(?:ED)?|H\s*/\s*T"),
]
_UOS_TAIL = re.compile(r"[\s,;]*\b(?:UNLESS\s+OTHERWISE\s+(?:SPECIFIED|STATED|NOTED)|U\.?O\.?S\.?)\b.*$", re.I)


def _callout_value(text: str, label_rx: str) -> Optional[str]:
    m = re.match(rf"^\s*(?:{label_rx})\s*[:\-=]\s*(.+)$", text or "", re.I | re.S)
    if not m:
        return None
    v = _UOS_TAIL.sub("", m.group(1))
    v = re.sub(r"\s+", " ", v).strip(" .;,")
    return v or None


def labelled_callouts(res: DrawingResult) -> dict[str, str]:
    """Explicitly labelled material / finish / heat-treatment callouts, from
    the title block first, then the notes (first labelled note wins)."""
    out: dict[str, str] = {}
    tb_map = {"material_callout": "material", "surface_finish": "finish",
              "heat_treatment": "heat_treatment"}
    for key, tbk in tb_map.items():
        f = res.title_block.get(tbk)
        if f and f.value and len(f.value.strip(" -/.")) > 1:
            out[key] = re.sub(r"\s+", " ", f.value).strip(" .;,")
    for n in res.notes:
        for key, rx in _CALLOUT_LABELS:
            if key in out:
                continue
            v = _callout_value(n["text"], rx)
            if v:
                out[key] = v
                break
    return out


def _dedupe_specs(specs: list[SpecCallout]) -> list[SpecCallout]:
    seen, out = set(), []
    for s in specs:
        k = (s.category, s.value.upper(), s.source, s.page)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _inside(w: Word, b) -> bool:
    return b[0] <= w.cx <= b[2] and b[1] <= w.cy <= b[3]


def _ocr_regions(img, gray, grid, raw: RawPage, dpi: float, cfg: dict, existing: list[Word]) -> list[Word]:
    """OCR only the page regions without native text (pasted scans etc.) and
    return words, in page pixels, that do not duplicate native words."""
    clean = imaging.remove_lines(gray, grid)
    h, w = clean.shape[:2]
    boxes = [x.bbox for x in existing]
    out: list[Word] = []
    for x0, y0, x1, y1 in raw.ocr_regions:
        ix0, iy0 = max(0, int(x0)), max(0, int(y0))
        ix1, iy1 = min(w, int(x1) + 1), min(h, int(y1) + 1)
        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            continue
        try:
            found = imaging.ocr_image(clean[iy0:iy1, ix0:ix1], dpi, cfg["ocr_lang"], psm=11,
                                      min_conf=cfg["min_word_conf"])
        except Exception as e:
            raw.notes.append(f"{PARTIAL_TAG} OCR of raster region failed: {e}")
            raw.partial = True
            continue
        for wd in found:
            wd = Word(wd.text, wd.x0 + ix0, wd.y0 + iy0, wd.x1 + ix0, wd.y1 + iy0, wd.conf)
            if any(_box_overlap(wd.bbox, b) > 0.4 for b in boxes):
                continue
            out.append(wd)
            boxes.append(wd.bbox)
    return out


def _box_overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    return ix * iy / area


def _avg(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 0.0
