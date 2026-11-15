"""Turn detected regions into structured records: title block fields, BOM
rows and revision rows, with normalisation and validation flags."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Optional

from .layout import Table
from .models import BomRow, Cell, FieldValue, Measurement, RevisionRow, union_bbox
from .textutil import clean_value, match_label_prefix

# ======================================================================
# dates
# ======================================================================
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/. ]([A-Za-z]{3,9})[-/., ]+(\d{2,4})\b"),
    re.compile(r"\b([A-Za-z]{3,9})[-/. ]+(\d{1,2})(?:st|nd|rd|th)?[-/., ]+(\d{2,4})\b"),
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b"),
]


def parse_date(text: str, order: str = "dmy") -> Optional[str]:
    """Parse common drawing date formats to ISO yyyy-mm-dd."""
    if not text:
        return None
    t = text.strip()
    for k, rx in enumerate(_DATE_PATTERNS):
        m = rx.search(t)
        if not m:
            continue
        try:
            if k == 0:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif k == 1:
                d, mo, y = int(m.group(1)), _MONTHS.get(m.group(2)[:3].upper()), int(m.group(3))
            elif k == 2:
                mo, d, y = _MONTHS.get(m.group(1)[:3].upper()), int(m.group(2)), int(m.group(3))
            else:
                a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if a > 12:
                    d, mo = a, b
                elif b > 12:
                    mo, d = a, b
                else:
                    d, mo = (a, b) if order == "dmy" else (b, a)
            if mo is None:
                continue
            if y < 100:
                y += 2000 if y < 70 else 1900
            return date(y, mo, d).isoformat()
        except (ValueError, TypeError):
            continue
    return None


def looks_like_date(text: str) -> bool:
    return parse_date(text) is not None and len(text) <= 24


# ======================================================================
# title block
# ======================================================================
_STRONG_MID = {"SCALE", "SHEET", "SHT", "SIZE", "REV", "DATE", "WEIGHT", "MASS"}
_SCALE_RX = re.compile(r"\b(\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?|NTS|N\.T\.S\.?|NONE|FULL(?:\s+SIZE)?)\b", re.I)
_PROJ_RX = re.compile(r"\b(FIRST|THIRD|1ST|3RD)[\s-]+ANGLE(?:\s+PROJECTION)?\b", re.I)
_DWG_RX = re.compile(r"\b(?:DWG|DRAWING|DRG)\.?\s*(?:NO|NUMBER|#)\.?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_./]{2,})", re.I)
_REV_RX = re.compile(r"\b(?:REV|REVISION)\b\.?\s*[:#]?\s*([A-Z0-9]{1,3})\b", re.I)
_PERSON_FIELDS = ("drawn_by", "checked_by", "approved_by")


_COMBO_PERSON_DATE = re.compile(
    r"^\s*(DRAWN|DRN|DESIGNED|CHECKED|CHKD|APPROVED|APPD)\.?(?:\s*BY)?\s*[/&,]\s*(?:SIGN\s+)?DATE\s*[:.\-]?\s*(.*)$",
    re.I | re.S)
_COMBO_FIELD = {"DRAWN": "drawn_by", "DRN": "drawn_by", "DESIGNED": "drawn_by", "CHECKED": "checked_by",
                "CHKD": "checked_by", "APPROVED": "approved_by", "APPD": "approved_by"}


def _split_labels(text: str, vocab) -> list[tuple[Optional[str], str]]:
    """Split a cell's text into (field, value) chunks. The first chunk may be
    unlabelled. Mid-text labels split only when followed by ':' or when
    they are 'strong' single-word labels like SCALE / SHEET / REV."""
    m = _COMBO_PERSON_DATE.match(text)  # "DRAWN BY / DATE: J. DOE / 2024-08-15"
    if m:
        return [(_COMBO_FIELD[m.group(1).upper()], m.group(2))]
    chunks: list[tuple[Optional[str], str]] = []
    rest = text
    f, lab, rem = match_label_prefix(rest, vocab)
    cur_field = f
    rest = rem if f else rest
    buf = []
    tokens = rest.split()
    i = 0
    while i < len(tokens):
        tail = " ".join(tokens[i:])
        if buf:  # look for a mid-text label
            f2, lab2, rem2 = match_label_prefix(tail, vocab)
            if f2:
                lab_u = lab2.strip().rstrip(":.#-").strip().upper()
                if lab2.rstrip().endswith(":") or lab_u in _STRONG_MID:
                    chunks.append((cur_field, " ".join(buf)))
                    buf = []
                    cur_field = f2
                    tokens = rem2.split()
                    i = 0
                    continue
        buf.append(tokens[i])
        i += 1
    chunks.append((cur_field, " ".join(buf)))
    return chunks


def _neighbour(c: Cell, cells: list[Cell], direction: str, tol: float) -> Optional[Cell]:
    best, best_ov = None, 0.0
    for o in cells:
        if o is c:
            continue
        if direction == "right" and abs(o.x0 - c.x1) <= tol:
            ov = min(c.y1, o.y1) - max(c.y0, o.y0)
            if ov > 0.5 * min(c.h, o.h) and ov > best_ov:
                best, best_ov = o, ov
        elif direction == "below" and abs(o.y0 - c.y1) <= tol:
            ov = min(c.x1, o.x1) - max(c.x0, o.x0)
            if ov > 0.5 * min(c.w, o.w) and ov > best_ov:
                best, best_ov = o, ov
    return best


def _spatial_neighbour(c: Cell, cells: list[Cell], label_ids: set[int], tol: float,
                       field: str = "") -> Optional[Cell]:
    """Find a neighbouring value cell using bounded spatial search when strict ruling alignment fails.

    Searches first to the right, then below:
    - Candidate is not the label cell itself and not another recognised label
    - Candidate contains words/text
    - Candidate is within bounded distance (tol * 4.5 horizontally, tol * 3.5 vertically)
    - Vertical overlap or center alignment for right neighbours; horizontal overlap or center alignment below
    - Value satisfies _valid(field, val)
    """
    max_h_tol = max(tol * 8.0, 150.0)
    overlap_tol = max(tol * 2.0, 20.0)
    c_mid_y = (c.y0 + c.y1) / 2.0

    # 1. Look right: closest value cell to the right with vertical overlap/alignment
    best_right, best_right_score = None, float("inf")
    for o in cells:
        if o is c or o.id in label_ids or not o.words:
            continue
        h_gap = o.x0 - c.x1
        if -overlap_tol <= h_gap <= max_h_tol:
            ov = min(c.y1, o.y1) - max(c.y0, o.y0)
            v_shift = abs(c_mid_y - (o.y0 + o.y1) / 2.0)
            if ov > 0.15 * min(c.h, o.h) or v_shift <= max(c.h, o.h) * 1.5:
                txt = clean_value(o.text)
                if not txt:
                    continue
                if field and not _valid(field, txt):
                    continue
                score = max(0.0, h_gap) + v_shift * 1.5
                if score < best_right_score:
                    best_right, best_right_score = o, score

    if best_right is not None:
        return best_right

    # 2. Look below: closest value cell below with horizontal overlap/alignment
    max_v_tol = max(tol * 6.0, 100.0)
    c_mid_x = (c.x0 + c.x1) / 2.0
    best_below, best_below_score = None, float("inf")
    for o in cells:
        if o is c or o.id in label_ids or not o.words:
            continue
        v_gap = o.y0 - c.y1
        if -overlap_tol <= v_gap <= max_v_tol:
            ov = min(c.x1, o.x1) - max(c.x0, o.x0)
            h_shift = abs(c_mid_x - (o.x0 + o.x1) / 2.0)
            if ov > 0.15 * min(c.w, o.w) or h_shift <= max(c.w, o.w) * 1.5:
                txt = clean_value(o.text)
                if not txt:
                    continue
                if field and not _valid(field, txt):
                    continue
                score = max(0.0, v_gap) + h_shift * 1.5
                if score < best_below_score:
                    best_below, best_below_score = o, score

    return best_below


def parse_title_block(cells: list[Cell], vocab: dict, dpi: float, page_text: str,
                      date_order: str = "dmy") -> dict[str, FieldValue]:
    tol = dpi * 0.035
    cands: dict[str, list[tuple[FieldValue, Cell]]] = defaultdict(list)
    labelled: dict[int, list] = {}
    for c in cells:
        if c.words:
            labelled[c.id] = _split_labels(c.text, vocab)
    label_ids = {cid for cid, ch in labelled.items() if any(f for f, _ in ch)}

    # rank of a candidate: how directly the value is tied to its label
    # 0 = same cell ('SCALE 1:2'), 1 = right neighbour, 2 = below, 3 = spatial guess
    rank: dict[int, int] = {}
    for c in cells:
        for f, val in labelled.get(c.id, []):
            if not f:
                continue
            val = clean_value(val)
            src_cell, r = c, 0
            if not val or not _valid(f, val) or _is_label_only(val, vocab):
                nb = None
                for r_try, d in ((1, "right"), (2, "below")):
                    n = _neighbour(c, cells, d, tol)
                    if n is not None and n.words and n.id not in label_ids:
                        val_candidate = clean_value(n.text)
                        if val_candidate and _valid(f, val_candidate) and \
                                not _is_label_only(val_candidate, vocab):
                            nb, r = n, r_try
                            break
                if nb is None:
                    nb, r = _spatial_neighbour(c, cells, label_ids, tol, f), 3
                if nb is None:
                    continue
                val, src_cell = clean_value(nb.text), nb
                if _is_label_only(val, vocab):
                    continue
            fv = FieldValue(val, round(src_cell.conf * (1.0, 0.97, 0.93, 0.8)[r], 3), src_cell.bbox,
                            raw=val, source="title_block")
            rank[id(fv)] = r
            cands[f].append((fv, c))

    # Each value cell belongs to one label: fields whose value is tied most
    # directly (same cell, then right, below, spatial) claim it first.
    out: dict[str, FieldValue] = {}
    claimed: set = set()
    order = sorted(cands, key=lambda f: min((rank[id(x[0])] for x in cands[f]), default=9))
    for f in order:
        lst = [x for x in cands[f] if _valid(f, x[0].value)]
        lst = [x for x in lst if rank[id(x[0])] == 0 or x[0].bbox not in claimed]
        if not lst:
            continue
        if f == "drawing_title":
            pick = min(lst, key=lambda x: (rank[id(x[0])] > 1, -len(x[0].value or "")))[0]
        else:
            pick = min(lst, key=lambda x: (rank[id(x[0])], -x[0].confidence))[0]
        pick.candidates = [candidate.to_dict() for candidate, _ in lst]
        unique_values = {str(candidate.value).strip().casefold() for candidate, _ in lst if candidate.value}
        if len(unique_values) > 1:
            pick.needs_review = True
        out[f] = pick
        if rank[id(pick)] > 0:
            claimed.add(pick.bbox)

    # person rows like | DRAWN | J. DOE | 15/08/2024 |
    for pf in _PERSON_FIELDS:
        if pf in out:
            v = out[pf]
            # value may contain the date: "J. DOE 2024-08-15"
            d = parse_date(v.value, date_order)
            if d:
                name = re.sub(r"\d{1,4}[-/. ]\w{1,9}[-/., ]+\d{2,4}", "", v.value).strip(" ,/")
                out[pf] = FieldValue(name or v.value, v.confidence, v.bbox, raw=v.raw)
                dm = re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/. ]\w{1,9}[-/., ]+\d{2,4}", v.value)
                out.setdefault(pf.replace("_by", "_date"), FieldValue(dm.group(0) if dm else d, v.confidence,
                                                                       v.bbox, raw=v.raw, normalized=d))
                continue
            src = next((c for c in cells if c.bbox == v.bbox), None)
            if src is not None:
                n = _neighbour(src, cells, "right", tol)
                if n is None or not (n.words and looks_like_date(n.text)):
                    n = _spatial_neighbour(src, cells, label_ids, tol, "date")
                if n is not None and n.words and looks_like_date(n.text):
                    key = pf.replace("_by", "_date")
                    out.setdefault(key, FieldValue(clean_value(n.text), round(n.conf, 3), n.bbox,
                                                   raw=n.text))

    # no labelled title: the largest unlabelled lettering in the block is the title
    if "drawing_title" not in out:
        used_boxes = {fv.bbox for fv in out.values()}
        best, best_h = None, 0.0
        for c in cells:
            if not c.words or c.id in label_ids or c.bbox in used_boxes:
                continue
            txt = clean_value(c.text)
            if len(re.findall(r"[A-Za-z]{2,}", txt)) < 2 or looks_like_date(txt):
                continue
            hgt = sorted(w.h for w in c.words)[len(c.words) // 2]
            if hgt > best_h:
                best, best_h = c, hgt
        if best is not None:
            big = [w for w in best.words if w.h >= 0.7 * best_h]  # drop small label text
            from .textutil import group_lines
            txt = clean_value(" ".join(l.text for l in group_lines(big)))
            out["drawing_title"] = FieldValue(txt, round(best.conf * 0.8, 3), best.bbox,
                                              raw=best.text)

    # resolve the generic DATE: drawn date > date label nearest DRAWN > first
    if "drawn_date" in out and "date" not in out:
        out["date"] = out["drawn_date"]
    elif len(cands.get("date", [])) > 1 and "drawn_by" in out:
        db = out["drawn_by"].bbox
        best = min((x for x in cands["date"] if _valid("date", x[0].value)),
                   key=lambda x: _dist(x[0].bbox, db), default=None)
        if best:
            out["date"] = best[0]

    # ---- fallbacks from free page text
    if "drawing_number" not in out and page_text:
        m = _DWG_RX.search(page_text)
        if m:
            out["drawing_number"] = FieldValue(m.group(1), 0.6, None, raw=m.group(0))
    # page_text excludes revision / BOM tables (see pipeline), so this cannot
    # pick up a historical revision-table entry
    if "revision" not in out and page_text:
        m = _REV_RX.search(page_text)
        if m:
            val = m.group(1).strip()
            if _valid("revision", val):
                out["revision"] = FieldValue(val, 0.6, None, raw=m.group(0), normalized=val,
                                             source="page_text")
    if "scale" not in out and page_text:
        m = _SCALE_LABEL_RX.search(page_text)
        if m:
            out["scale"] = FieldValue(m.group(1).strip(), 0.7, None, raw=m.group(0), source="page_text")
    if "projection" not in out or not _PROJ_RX.search(out["projection"].value or ""):
        m = _PROJ_RX.search(page_text)
        if m:
            out["projection"] = FieldValue(m.group(0).upper(), 0.9, None, raw=m.group(0),
                                           source="page_text")
        elif "projection" in out:
            out["projection"].confidence = min(out["projection"].confidence, 0.5)

    for fv in out.values():
        if fv.source is None:
            fv.source = "title_block" if fv.bbox else "page_text"
    normalise_title_fields(out, date_order)
    return out


def normalise_title_fields(out: dict[str, FieldValue], date_order: str = "dmy") -> None:
    """Fill FieldValue.normalized (drawing numbers, dates, scale, projection,
    revision, weight) in place."""
    for k, fv in out.items():
        if fv.value is None:
            continue
        if k in ("drawing_number", "part_number"):
            fv.normalized = re.sub(r"\s*([-_/.])\s*", r"\1", fv.value).replace(" ", "")
        elif k.endswith("date") or k == "date":
            fv.normalized = fv.normalized or parse_date(fv.value, date_order)
        elif k == "scale":
            m = _SCALE_RX.search(fv.value)
            if m:
                fv.normalized = re.sub(r"\s+", "", m.group(1)).replace("/", ":").upper()
            elif re.fullmatch(r"\d{2,3}", fv.value.strip()):  # OCR dropped the colon: 15 -> 1:5
                d = fv.value.strip()
                fv.normalized = f"{d[0]}:{d[1:]}"
                fv.confidence = min(fv.confidence, 0.5)
        elif k == "projection":
            u = fv.value.upper()
            fv.normalized = ("THIRD ANGLE" if ("THIRD" in u or "3RD" in u)
                             else "FIRST ANGLE" if ("FIRST" in u or "1ST" in u) else None)
        elif k == "revision":
            fv.normalized = re.sub(r"^REV(?:ISION)?\.?\s*[:#]?\s*", "", fv.value.upper()).strip()
        elif k == "weight":
            normalized, measurement, flags = parse_mass_measurement(fv.value)
            fv.normalized = normalized
            if flags or measurement is None or normalized is None:
                fv.needs_review = True
                fv.candidates.append({"raw": fv.value, "measurement": measurement.to_dict() if measurement else None,
                                      "flags": flags, "page": fv.page})


_SCALE_LABEL_RX = re.compile(r"\bSCALE\b\s*[:=]?\s*(\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?|NTS|N\.T\.S\.?|FULL(?:\s+SIZE)?)",
                             re.I)


def _is_label_only(v: str, vocab: dict) -> bool:
    """True when `v` is just one or more title-block labels with no value
    ('SCALE', 'DRAWN BY / DATE') - an adjacent label, not a value."""
    parts = _split_labels(v, vocab)
    return bool(parts) and all(f and not clean_value(val) for f, val in parts)


def _valid(f: str, v: Optional[str]) -> bool:
    if not v:
        return False
    if f == "drawing_number":
        return bool(re.search(r"\d", v)) and len(v) <= 40 and not looks_like_date(v) \
            and not _SCALE_RX.fullmatch(v.strip())
    if f == "drawing_title":
        return bool(re.search(r"[A-Za-z]{2}", v)) and not looks_like_date(v) \
            and not _SCALE_RX.fullmatch(v.strip()) and len(v) <= 120
    if f == "projection":
        return bool(_PROJ_RX.search(v)) or bool(re.search(r"\b(FIRST|THIRD|1ST|3RD)\b", v, re.I))
    if f == "revision":
        v_clean = re.sub(r"\s+", " ", v).strip()
        if len(v_clean) > 25:
            return False
        if re.fullmatch(r"REV(?:ISION)?\.?", v_clean, re.I):
            return False
        if re.search(r"\b(?:NOTE|GENERAL|SHEET|TOLERANCE|UNLESS|ZONE|TITLE)\b", v_clean, re.I):
            return False
        v_val = re.sub(r"^REV(?:ISION)?\.?\s*[:#]?\s*", "", v_clean, flags=re.I).strip()
        if not v_val or len(v_val) > 8 or bool(re.search(r"[^A-Za-z0-9\-./]", v_val)):
            return False
        if v_val.lower() in {"is", "no", "to", "of", "the", "in", "on", "at", "by", "revis", "history", "rev", "drg", "dwg", "all", "and", "for"}:
            return False
        if looks_like_date(v_val):
            return False
        return True
    if f in ("date",) or f.endswith("_date"):
        return parse_date(v) is not None
    if f == "scale":
        return bool(_SCALE_RX.search(v)) or bool(re.fullmatch(r"\d{2,3}", v.strip()))
    if f in _PERSON_FIELDS:
        # a name (may carry a date: 'J. DOE 15/08/2024' - split later), never a bare date / number
        return len(v) <= 60 and bool(re.search(r"[A-Za-z]{2}", v)) and not _SCALE_RX.fullmatch(v.strip())
    return True


def _dist(a, b) -> float:
    if a is None or b is None:
        return 1e12
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return (ax - bx) ** 2 + (ay - by) ** 2 * 4  # same row matters more


# ======================================================================
# BOM
# ======================================================================
_NUM_PATTERN = r"(?:\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
_QTY_RX = re.compile(rf"^\s*({_NUM_PATTERN})\s*(NOS?|PCS?|PC|EA|OFF|SETS?|M|MM|MTR|KG|L|LTR)?\.?\s*$", re.I)
_AR_RX = re.compile(r"^\s*(A/?R|AS\s+REQ(?:UIRED|D)?|AS\s+REQUIRED|REF)\s*\.?$", re.I)
_MASS_RX = re.compile(rf"({_NUM_PATTERN})\s*(KGS?|G|GMS?|GRAMS?|LBS?|POUNDS?|OZ|OUNCES?|T|TONNES?|MG)?\b", re.I)


def _parse_num_str(s: str, is_quantity: bool = False) -> Optional[float]:
    """Parse numeric strings supporting grouped thousands separators, decimal points,
    decimal commas, and mixed formats across engineering drawing locales.
    """
    s = s.strip()
    if not s:
        return None
    # Normalize spaces between digits (e.g. ISO 80000 '1 234.50' -> '1234.50')
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    # 1. Both comma and dot present: determine decimal by the last occurring separator
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # European style: 1.234,50 or 1.234.567,89
            s = s.replace(".", "").replace(",", ".")
        else:
            # Standard/US style: 1,234.50 or 1,234,567.89
            s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    # 2. Multiple occurrences of the same separator (unambiguously thousands)
    if s.count(",") > 1:
        # e.g. 1,234,567
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    if s.count(".") > 1:
        # e.g. 1.234.567
        s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return None

    # 3. Single comma present
    if "," in s:
        parts = s.split(",")
        if len(parts) == 2:
            # If digits after comma != 3 (e.g. 1,5 or 0,75 or 12,50 or 1,2345),
            # it is unambiguously a decimal comma:
            if len(parts[1]) != 3:
                s = s.replace(",", ".")
            else:
                # Exactly 3 digits after comma (e.g. 1,234 or 10,000):
                # For quantities (component counts) or numbers with >= 2 leading digits (10,000)
                # or ending in '000', treat as thousands separator.
                if is_quantity or parts[1] == "000" or len(parts[0]) >= 2:
                    s = parts[0] + parts[1]
                else:
                    # In standard engineering BOM tables, 1,234 is thousands (1234)
                    s = parts[0] + parts[1]
        try:
            return float(s)
        except ValueError:
            return None

    # 4. Standard integer or float
    try:
        return float(s)
    except ValueError:
        return None


def ocr_digits(s: str) -> str:
    """Fix letter/digit confusions in values that should be numeric."""
    return s.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8",
                                      "{": "1", "}": "1", "[": "1", "]": "1", "!": "1", "i": "1"}))


def parse_quantity_measurement(text: str, conf: float = 1.0) -> tuple[Optional[float], Optional[Measurement], list[str]]:
    """Parse quantity string into nominal value, canonical Measurement, and validation flags."""
    if not text:
        return None, None, ["missing_quantity"]
    raw = text.strip()
    if _AR_RX.match(raw):
        meas = Measurement(
            raw=raw,
            nominal=None,
            unit=None,
            canonical_value=None,
            canonical_unit="count",
            measurement_type="quantity",
            confidence=conf,
        )
        return None, meas, ["quantity_as_required"]

    m = _QTY_RX.match(raw)
    if not m:
        return None, None, ["quantity_not_numeric"]

    v = _parse_num_str(m.group(1), is_quantity=True)
    if v is None:
        return None, None, ["quantity_not_numeric"]

    unit_str = m.group(2)
    nominal = int(v) if v.is_integer() else v

    canonical_val = float(nominal)
    canonical_unit = "count"
    if unit_str:
        u_upper = unit_str.upper()
        if u_upper in ("M", "MTR"):
            canonical_unit = "m"
        elif u_upper == "MM":
            canonical_unit = "m"
            canonical_val = canonical_val / 1000.0
        elif u_upper.startswith("KG"):
            canonical_unit = "kg"
        elif u_upper in ("L", "LTR"):
            canonical_unit = "l"
        else:
            canonical_unit = "count"

    meas = Measurement(
        raw=raw,
        nominal=float(nominal),
        unit=unit_str,
        canonical_value=canonical_val,
        canonical_unit=canonical_unit,
        measurement_type="quantity",
        confidence=conf,
    )
    return nominal, meas, []


def parse_mass_measurement(text: str, default_unit: Optional[str] = None, conf: float = 1.0) -> tuple[Optional[float], Optional[Measurement], list[str]]:
    """Parse mass string into kg value, canonical Measurement, and validation flags."""
    if not text:
        return None, None, []
    raw = text.strip()
    m = _MASS_RX.search(raw)
    if not m:
        flags = ["unsupported_mass_unit"] if re.search(r"[A-Za-z]{2,}", raw) else ["mass_not_numeric"]
        return None, None, flags

    v = _parse_num_str(m.group(1), is_quantity=False)
    if v is None:
        flags = ["unsupported_mass_unit"] if re.search(r"[A-Za-z]{2,}", raw) else ["mass_not_numeric"]
        return None, None, flags

    matched_unit = m.group(2)
    suffix = raw[m.end(1):].strip()
    if not matched_unit and re.search(r"[A-Za-z]", suffix):
        meas = Measurement(raw=raw, nominal=v, unit=suffix, canonical_value=None,
                           canonical_unit=None, measurement_type="mass", confidence=conf)
        return None, meas, ["unsupported_mass_unit"]
    unit_str = matched_unit or default_unit
    flags = []
    if not unit_str:
        meas = Measurement(raw=raw, nominal=v, unit=None, canonical_value=None,
                           canonical_unit=None, measurement_type="mass", confidence=conf)
        return None, meas, ["mass_unit_assumed_kg"]
    u = unit_str.upper()

    if u.startswith("KG"):
        canonical_val = v
        canonical_unit = "kg"
    elif u in ("G", "GM", "GMS", "GRAM", "GRAMS"):
        canonical_val = v / 1000.0
        canonical_unit = "kg"
    elif u.startswith("LB") or u.startswith("POUND"):
        canonical_val = v * 0.45359237
        canonical_unit = "kg"
    elif u.startswith("OZ") or u.startswith("OUNCE"):
        canonical_val = v * 0.0283495
        canonical_unit = "kg"
    elif u == "MG":
        canonical_val = v / 1_000_000.0
        canonical_unit = "kg"
    elif u in ("T", "TONNE", "TONNES"):
        canonical_val = v * 1000.0
        canonical_unit = "kg"
    else:
        meas = Measurement(raw=raw, nominal=v, unit=unit_str, canonical_value=None,
                           canonical_unit=None, measurement_type="mass", confidence=conf)
        return None, meas, ["unsupported_mass_unit"]

    meas = Measurement(
        raw=raw,
        nominal=v,
        unit=unit_str,
        canonical_value=round(canonical_val, 6),
        canonical_unit=canonical_unit,
        measurement_type="mass",
        confidence=conf,
    )
    return canonical_val, meas, flags


def to_kg(text: str, default_unit: Optional[str] = None) -> Optional[float]:
    val, _, _ = parse_mass_measurement(text, default_unit=default_unit)
    return val


def _merged_fill(t: Table) -> dict[int, list[str]]:
    """A cell merged vertically over 3+ rows (same MATERIAL for several items)
    is cut into one piece per row with the text in only one of them. Give its
    value to every row it spans. Returns {row index: [filled columns]}.
    Two-row spans are left alone: there a lost row rule is just as likely."""
    filled: dict[int, list[str]] = {}
    spans: dict[tuple, list[int]] = {}
    for i, vals in enumerate(t.rows):
        for k, v in vals.items():
            g = getattr(v, "group", -1)
            if g >= 0:
                spans.setdefault((k, g), []).append(i)
    for (k, _), idx in spans.items():
        if len(set(idx)) < 3:
            continue
        with_text = [i for i in idx if t.rows[i][k].text.strip()]
        if len(with_text) != 1:
            continue
        src = t.rows[with_text[0]][k]
        for i in idx:
            if i != with_text[0] and not t.rows[i][k].text.strip():
                v = t.rows[i][k]
                t.rows[i][k] = type(v)(src.text, src.conf, src.bbox, "merged",
                                       struct_conf=min(getattr(src, "struct_conf", 1.0), 0.85),
                                       group=getattr(src, "group", -1))
                filled.setdefault(i, []).append(k)
    return filled


def _non_data_row(vals: dict, vocab: Optional[dict]) -> Optional[str]:
    """A repeated header line or a caption ('BILL OF MATERIALS') read as a
    row: structure, not data. Returns the reason, or None for a data row."""
    from .layout import _CAPTION, match_header
    texts = [v.text.strip() for v in vals.values() if v.text and v.text.strip()]
    if not texts:
        return "empty"
    if len(texts) == 1 and _CAPTION.search(texts[0]) and len(texts[0]) <= 40:
        return "caption"
    if vocab and len(texts) >= 2:
        hits = sum(1 for x in texts if match_header(x, vocab)[0])
        if hits >= max(2, int(0.75 * len(texts) + 0.5)):
            return "repeated header"
    return None


def table_to_bom(t: Table, page: int, low_conf: float, table_index: int = 0,
                 vocab: Optional[dict] = None) -> tuple[list[BomRow], list[str]]:
    std = ("item", "part_number", "description", "quantity", "material", "mass", "remarks")
    mass_col = next((c for c in t.columns if c.key == "mass"), None)
    mass_unit = mass_col.unit if mass_col else None
    mass_is_total = bool(mass_col and re.search(r"TOTAL", mass_col.header, re.I))
    rows: list[BomRow] = []
    warnings: list[str] = []
    merged = _merged_fill(t)
    skipped: dict[str, int] = {}
    # item-numbered table: a line with no item / part number / quantity and
    # only text columns is a wrapped description (own ruled line), not a part
    n_items = sum(1 for v in t.rows if v.get("item") is not None and v["item"].text.strip())
    item_driven = n_items >= max(2, 0.5 * len(t.rows))
    text_keys = ("description", "material", "remarks", "size", "supplier", "finish")
    n_joined = 0
    # Wrapped lines sit *below* their row on the sheet. Bottom-header BOMs are
    # stored nearest-the-header first (bottom-up), so walk them top-down here
    # (and restore the order afterwards) for the join to hit the right row.
    bottom_up = getattr(t, "header_position", "top") == "bottom"
    order = range(len(t.rows) - 1, -1, -1) if bottom_up else range(len(t.rows))
    for ri in order:
        vals, bb = t.rows[ri], t.row_bboxes[ri]
        why = _non_data_row(vals, vocab)
        if why:
            if why != "empty":
                skipped[why] = skipped.get(why, 0) + 1
            continue
        own = [k for k, v in vals.items() if v.text and v.text.strip() and k not in merged.get(ri, [])]
        continuation_candidate = bool(item_driven and rows and own and \
                not any(k.split("_")[0] in ("item", "part", "quantity", "mass") for k in own) and \
                all(k.split("_")[0] in text_keys for k in own))
        # Keep identifier-less text rows separate. They may be wrapped
        # descriptions, but without geometric proof they remain reviewable rows.
        r = BomRow(page=page, bbox=bb, table=table_index, row_index=ri)
        if continuation_candidate:
            r.flags.append("possible_continuation")
        confs = []
        struct_confs = []
        for k, v in vals.items():
            txt = clean_value(v.text) or None
            if txt:
                confs.append(v.conf)
                struct_confs.append(getattr(v, "struct_conf", 1.0))
                r.field_confidence[k] = round(v.conf, 3)
                if v.bbox and len(v.bbox) == 4 and v.bbox[3] > v.bbox[1]:
                    r.field_bbox[k] = [round(float(x), 1) for x in v.bbox]
            if k in std:
                setattr(r, k, txt)
            elif txt:
                r.extra[k] = txt
        for k in merged.get(ri, []):
            r.flags.append(f"merged_cell_value:{k}")
        # ---- normalise
        if r.item:
            fixed = ocr_digits(r.item) if re.fullmatch(r"[\dOolI|{}\[\]!i]+\.?", r.item) else r.item
            r.item = fixed.rstrip(".")
        for k in ("description", "material", "remarks"):  # 'Stee!' -> 'Steel'
            v = getattr(r, k)
            if v and "!" in v:
                setattr(r, k, re.sub(r"(?<=[A-Za-z])!(?=[A-Za-z]|\s|$)", "l", v))
        for k in ("part_number", "material"):  # '$'/'§' are OCR misreads of S in codes
            v = getattr(r, k)
            if v and ("$" in v or "§" in v):
                setattr(r, k, v.replace("$", "S").replace("§", "S"))
        if r.part_number:
            r.part_number = re.sub(r"\s*([-_/.])\s*", r"\1", r.part_number)
        
        row_conf = round(min(confs) if confs else 1.0, 3)
        if r.quantity:
            q = r.quantity
            qty_val, qty_meas, q_flags = parse_quantity_measurement(q, conf=row_conf)
            if qty_meas is None and parse_quantity_measurement(ocr_digits(q))[1] is not None:
                r.quantity = ocr_digits(q)
                qty_val, qty_meas, q_flags = parse_quantity_measurement(r.quantity, conf=row_conf)
            r.quantity_value = qty_val
            r.quantity_measurement = qty_meas
            r.flags.extend(q_flags)
        else:
            r.flags.append("missing_quantity")
        if r.mass:
            raw_mass = r.mass
            mass_to_parse = raw_mass
            is_suspicious = False
            # Check for colon/semicolon/apostrophe separator confusion (e.g. '2:10')
            if re.search(r"(?<=\d)\s*[:;']\s*(?=\d)", raw_mass):
                is_suspicious = True
            else:
                m_num = re.search(r"(\d+(?:[.,]\d+)?)", raw_mass)
                if m_num:
                    ns = m_num.group(1)
                    if "." not in ns and "," not in ns:
                        try:
                            nv = float(ns)
                            u = (mass_unit or "").upper()
                            # Suspicious integer mass in kg: e.g. 210 kg, 150 kg (often misread decimal like 2.10, 1.50)
                            if (u.startswith("KG") or not u) and nv >= 10 and (ns.endswith("0") or len(ns) >= 3):
                                is_suspicious = True
                        except ValueError:
                            pass

            re_ocr_resolved = False
            clean_img = getattr(t, "clean_image", None)
            mass_cell = vals.get("mass")
            if is_suspicious and clean_img is not None and mass_cell and getattr(mass_cell, "bbox", None):
                try:
                    from .imaging import ocr_numeric
                    got = ocr_numeric(clean_img, mass_cell.bbox, dpi=getattr(t, "dpi", 300.0), lang=getattr(t, "lang", "eng"),
                                      orig_text=raw_mass)
                    if got and got[0]:
                        num_txt, num_conf = got
                        orig_digits = re.sub(r"\D", "", ocr_digits(raw_mass))
                        got_digits = re.sub(r"\D", "", ocr_digits(num_txt))
                        # Compare both OCR results:
                        # Must have decimal point, good confidence, and digit agreement
                        if ("." in num_txt or "," in num_txt) and num_conf >= 0.85 and got_digits == orig_digits:
                            norm_num = num_txt.replace(",", ".")
                            m_u = re.search(r"([A-Za-z]+)", raw_mass)
                            u_str = f" {m_u.group(1)}" if m_u else ""
                            mass_to_parse = f"{norm_num}{u_str}"
                            re_ocr_resolved = True
                        elif "." in num_txt or "," in num_txt:
                            # Found decimal but low confidence or digit disagreement -> ambiguous candidate
                            norm_num = num_txt.replace(",", ".")
                            m_u = re.search(r"([A-Za-z]+)", raw_mass)
                            u_str = f" {m_u.group(1)}" if m_u else ""
                            mass_to_parse = f"{norm_num}{u_str}"
                            r.flags.append("mass_decimal_ambiguous")
                        else:
                            # Both OCR passes confirmed integer digits or no dot found -> ambiguous
                            r.flags.append("mass_decimal_ambiguous")
                except Exception:
                    r.flags.append("mass_decimal_ambiguous")
            elif is_suspicious:
                # If separator had colon or apostrophe, parse candidate with dot but flag as ambiguous
                if re.search(r"(?<=\d)\s*[:;']\s*(?=\d)", mass_to_parse):
                    mass_to_parse = re.sub(r"(?<=\d)\s*[:;']\s*(?=\d)", ".", mass_to_parse)
                r.flags.append("mass_decimal_ambiguous")

            mass_val, mass_meas, m_flags = parse_mass_measurement(mass_to_parse, default_unit=mass_unit, conf=row_conf)
            # Crucial: preserve original OCR text in r.mass
            r.mass = raw_mass
            if "mass_decimal_ambiguous" in r.flags:
                # Ambiguity is authoritative: do not populate confirmed mass_kg or mass_measurement
                r.mass_kg = None
                r.mass_measurement = None
                r.mass_verified = False
                if mass_val is not None:
                    r.mass_candidate = round(mass_val, 4)
                    r.extra["mass_candidate"] = str(round(mass_val, 4))
            else:
                if mass_meas:
                    mass_meas.raw = raw_mass
                r.mass_kg = mass_val
                r.mass_measurement = mass_meas
                # Parsing yields a candidate; approval requires human review.
                r.mass_verified = False
                r.mass_candidate = None
            r.flags.extend(m_flags)
        if mass_col:
            r.extra["mass_basis"] = "total" if mass_is_total else "unit"
        if not r.part_number and not r.description:
            r.flags.append("missing_part_number")
        elif not r.part_number:
            r.flags.append("missing_part_number")
        r.confidence = round(min(confs) if confs else 0.0, 3)
        r.structural_confidence = round(min(struct_confs) if struct_confs else 1.0, 3)
        if r.confidence < low_conf:
            r.flags.append("low_confidence")
        if r.structural_confidence < low_conf:
            r.flags.append("uncertain_column_assignment")
        rows.append(r)
    if bottom_up:
        rows.reverse()  # back to nearest-the-header first

    for why, n in skipped.items():
        warnings.append(f"BOM table {table_index + 1}: {n} {why} line(s) inside the table not read as rows")
    if n_joined:
        warnings.append(f"BOM table {table_index + 1}: {n_joined} wrapped line(s) joined to the row above")
    nums = []
    for r in rows:
        if r.item and r.item.isdigit():
            nums.append(int(r.item))
    if nums and len(nums) == len(rows):
        rows.sort(key=lambda r: int(r.item))
        seen = set()
        for r in rows:
            if int(r.item) in seen:
                r.flags.append("duplicate_item")
            seen.add(int(r.item))
        exp = set(range(min(nums), max(nums) + 1))
        missing = sorted(exp - set(nums))
        if missing:
            warnings.append(f"BOM item numbers missing from sequence: {missing}")
    return rows, warnings


def table_to_revisions(t: Table, page: int, date_order: str) -> list[RevisionRow]:
    out = []
    keys = [c.key for c in t.columns]
    desc_alias = None
    if "description" not in keys:
        # header not recognised (or misspelled, e.g. 'Revisition'): the description
        # is the unnamed column holding the longest free text
        cands = [k for k in keys if k not in ("rev", "date", "approved", "zone", "ecn")]
        best, best_len = None, 0.0
        for k in cands:
            texts = [r.get(k).text for r in t.rows if r.get(k) is not None and r.get(k).text]
            if not texts or any(looks_like_date(x) for x in texts):
                continue
            avg = sum(len(x) for x in texts) / len(texts)
            if avg > best_len and avg >= 8:
                best, best_len = k, avg
        desc_alias = best
    for vals, bb in zip(t.rows, t.row_bboxes):
        r = RevisionRow(page=page, bbox=bb)
        confs = []
        for k, v in vals.items():
            txt = clean_value(v.text) or None
            if txt:
                confs.append(v.conf)
            if k == desc_alias:
                r.description = txt
            elif k in ("rev", "description", "date", "approved"):
                setattr(r, k, txt)
            elif txt:
                r.extra[k] = txt
        if r.date:
            iso = parse_date(r.date, date_order)
            if iso:
                r.extra["date_iso"] = iso

        # A revision entry must have an actual revision identifier.
        # Do not export title-block author/date text as a revision row.
        if not r.rev or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9._/-]{0,7}",
            r.rev.strip(),
            re.I,
        ):
            continue
        # A revision identifier alone is not enough to establish a useful row.
        if not any((r.description, r.date, r.approved, r.extra.get("ecn"))):
            continue

        r.confidence = round(min(confs) if confs else 0.0, 3)
        out.append(r)
    return out


def latest_revision(revs: list[RevisionRow]) -> Optional[RevisionRow]:
    revs = [r for r in revs if r.rev]
    if not revs:
        return None
    dated = [r for r in revs if r.extra.get("date_iso")]
    if dated:
        return max(dated, key=lambda r: (r.extra["date_iso"], _revision_key(r.rev)))
    return max(revs, key=lambda r: _revision_key(r.rev))


def _revision_key(value: Optional[str]) -> tuple:
    """Natural revision ordering: A10 follows A9; numeric revisions sort numerically."""
    text = (value or "").strip().upper()
    parts = re.findall(r"\d+|[A-Z]+", text)
    if not parts:
        return (0, text)
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts)
