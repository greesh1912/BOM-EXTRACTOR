"""Validation of an extraction result (Issue 6): checks every value against
its own evidence (format, confidence, source), finds missing and duplicate
information, and cross-checks BOM, title block, revision history and notes.

Produces a structured report stored in ``DrawingResult.validation``:

    {"issues": [{severity, category, section, field, page, row, message, value}],
     "counts": {"error": n, "warning": n, "info": n},
     "review_required": bool,
     "completeness": {"bom": 0..1, "title_block": 0..1},
     "review_fields": {"bom": {row: [fields]}, "title_block": [fields]}}

severity: error   - value is very likely wrong / contradicts other data
          warning - uncertain, verify manually
          info    - noteworthy, no action needed
Rows / fields with error or warning issues get the flag ``needs_review``.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from typing import Any, Optional

from .models import DrawingResult

DEFAULTS = {
    "mandatory_bom_fields": ["item", "part_number", "description", "quantity"],
    "mandatory_title_fields": ["drawing_number", "drawing_title", "revision", "date"],
    "review_conf": 0.75,  # value/row confidence below this -> manual review
    "part_number_pattern": None,  # optional company regex, e.g. "^[A-Z]{2}-\\d{5}$"
    "max_quantity": 10000,
    "mass_tolerance": 0.10,  # title-block weight vs BOM total (relative)
}

_COUNT_UNITS = {None, "", "EA", "PC", "PCS", "NO", "NOS", "OFF", "SET", "SETS"}
_OCR_JUNK = re.compile(r"[|!§¦{}\[\]<>~`]")
_PN_TOKEN = re.compile(r"\b(?=[A-Z0-9\-_/.]*\d)(?=[A-Z0-9\-_/.]*[A-Z])[A-Z0-9][A-Z0-9\-_/.]{4,}\b", re.I)
_NOTE_ITEM = re.compile(r"\b(?:ITEMS?|POS(?:ITION)?|FIND\s+NO)\.?\s*(?:NO\.?\s*)?(\d{1,3}(?:\s*(?:,|&|AND)\s*\d{1,3})*)",
                        re.I)


class _Report:
    def __init__(self):
        self.issues: list[dict] = []

    def add(self, severity: str, category: str, section: str, message: str, *,
            field: Optional[str] = None, page: Optional[int] = None, row: Optional[int] = None,
            value: Any = None) -> None:
        self.issues.append({"severity": severity, "category": category, "section": section,
                            "field": field, "page": page, "row": row, "message": message,
                            "value": value})


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[\s\-_./]+", "", (s or "").upper())


def validate_result(res: DrawingResult, cfg: dict) -> dict:
    vcfg = {**DEFAULTS, **(cfg.get("validation") or {})}
    rep = _Report()
    review_conf = float(vcfg["review_conf"])

    _check_bom_rows(res, vcfg, rep, review_conf)
    _check_bom_duplicates(res, rep)
    _check_missing_rows(res, rep)
    _check_title(res, vcfg, rep, review_conf)
    _cross_checks(res, vcfg, rep)

    # ---- aggregate
    counts = {"error": 0, "warning": 0, "info": 0}
    review_rows: dict[int, set] = defaultdict(set)
    review_tb: set = set()
    for i in rep.issues:
        counts[i["severity"]] += 1
        if i["severity"] == "info":
            continue
        if i["section"] == "bom" and i["row"] is not None:
            review_rows[i["row"]].add(i["field"] or "row")
        elif i["section"] == "title_block" and i["field"]:
            review_tb.add(i["field"])
    for ri in review_rows:
        if 0 <= ri < len(res.bom) and "needs_review" not in res.bom[ri].flags:
            res.bom[ri].flags.append("needs_review")

    mand_bom = [f for f in vcfg["mandatory_bom_fields"] if f in res.bom_columns]
    cells = len(res.bom) * len(mand_bom)
    filled = sum(1 for r in res.bom for f in mand_bom if getattr(r, f, None))
    mand_tb = vcfg["mandatory_title_fields"]
    tb_filled = sum(1 for f in mand_tb if res.title_block.get(f) and res.title_block[f].value)

    return {
        "issues": rep.issues,
        "counts": counts,
        "review_required": counts["error"] > 0 or counts["warning"] > 0,
        "completeness": {
            "bom": round(filled / cells, 3) if cells else None,
            "title_block": round(tb_filled / len(mand_tb), 3) if mand_tb else None,
        },
        "review_fields": {"bom": {str(k): sorted(v) for k, v in review_rows.items()},
                          "title_block": sorted(review_tb)},
    }


# ---------------------------------------------------------------- BOM rows

def _check_bom_rows(res: DrawingResult, vcfg: dict, rep: _Report, review_conf: float) -> None:
    from .specs import find_specs
    cols = set(res.bom_columns)
    for f in vcfg["mandatory_bom_fields"]:
        if res.bom and f not in cols:
            rep.add("info", "missing_column", "bom", f"BOM has no '{f}' column", field=f)
    pn_rx = re.compile(vcfg["part_number_pattern"]) if vcfg.get("part_number_pattern") else None
    descs = [len(r.description) for r in res.bom if r.description]
    med_desc = statistics.median(descs) if descs else 0
    all_pns = {_norm(r.part_number) for r in res.bom if r.part_number}

    for i, r in enumerate(res.bom):
        loc = {"page": r.page, "row": i}
        for f in vcfg["mandatory_bom_fields"]:
            if f in cols and not getattr(r, f, None):
                rep.add("warning", "missing_value", "bom", f"row {r.item or i + 1}: '{f}' is empty",
                        field=f, **loc)
        # quantity
        if r.quantity:
            unit = (r.quantity_measurement.unit if r.quantity_measurement else None) or ""
            q = r.quantity_value
            if q is None:
                if not re.fullmatch(r"\s*(A/?R|AS\s+REQ.*|REF)\s*\.?", r.quantity, re.I):
                    rep.add("error", "invalid_value", "bom", f"quantity '{r.quantity}' is not a number",
                            field="quantity", value=r.quantity, **loc)
            elif q <= 0:
                rep.add("error", "invalid_value", "bom", f"quantity {q:g} must be positive",
                        field="quantity", value=r.quantity, **loc)
            else:
                if unit.upper() in _COUNT_UNITS and abs(q - round(q)) > 1e-6:
                    rep.add("warning", "invalid_value", "bom",
                            f"fractional count quantity {q:g} (decimal point misread?)",
                            field="quantity", value=r.quantity, **loc)
                if q > float(vcfg["max_quantity"]):
                    rep.add("warning", "implausible_value", "bom", f"unusually large quantity {q:g}",
                            field="quantity", value=r.quantity, **loc)
        # part number
        if r.part_number:
            pn = r.part_number
            if pn_rx and not pn_rx.search(pn):
                rep.add("warning", "format_mismatch", "bom", f"part number '{pn}' does not match the "
                        "configured part-number pattern", field="part_number", value=pn, **loc)
            if _OCR_JUNK.search(pn) or not re.search(r"[A-Za-z0-9]", pn) or len(pn) > 40:
                rep.add("warning", "suspicious_value", "bom", f"part number '{pn}' contains "
                        "OCR-suspicious characters", field="part_number", value=pn, **loc)
        # description
        if r.description:
            d = r.description
            if not re.search(r"[A-Za-z]{2}", d):
                rep.add("warning", "suspicious_value", "bom", f"description '{d}' has no words",
                        field="description", value=d, **loc)
            # unrelated rows merged into one: very long text holding other rows' part numbers
            foreign = {_norm(t) for t in _PN_TOKEN.findall(d)} & (all_pns - {_norm(r.part_number)})
            if foreign or (med_desc and len(d) > max(80, 2.5 * med_desc)):
                rep.add("warning", "possible_merge", "bom",
                        "description is unusually long" + (" and contains other rows' part numbers"
                                                           if foreign else "") +
                        " - rows may have been merged", field="description", value=d, **loc)
        if "wrapped_text_joined" in r.flags and med_desc and r.description and \
                len(r.description) > 2 * max(med_desc, 20):
            rep.add("warning", "possible_merge", "bom", "joined wrapped line made the description "
                    "much longer than others - check it belongs to this row", field="description", **loc)
        # material
        if r.material and not find_specs(r.material, "bom", categories={"material"}):
            rep.add("info", "unrecognised_value", "bom", f"material '{r.material}' not in the "
                    "material library", field="material", value=r.material, **loc)
        # confidence
        if r.confidence < review_conf:
            rep.add("warning", "low_confidence", "bom", f"low OCR confidence ({r.confidence:.2f})",
                    field=None, value=r.confidence, **loc)
        if r.structural_confidence < review_conf:
            rep.add("warning", "low_confidence", "bom", "uncertain column assignment "
                    f"({r.structural_confidence:.2f})", value=r.structural_confidence, **loc)
        if "mass_decimal_ambiguous" in r.flags:
            rep.add("warning", "ambiguous_value", "bom", f"mass '{r.mass}' has an ambiguous decimal "
                    "separator", field="mass", value=r.mass, **loc)
        for flag, message in (("unsupported_mass_unit", "unsupported mass unit"),
                              ("mass_unit_assumed_kg", "mass unit is missing or assumed"),
                              ("malformed_mass_value", "mass contains trailing or malformed text"),
                              ("uncertain_column_assignment", "one or more columns have uncertain meaning"),
                              ("possible_continuation", "text row may continue the preceding row")):
            if flag in r.flags:
                rep.add("warning", "ambiguous_value", "bom", message,
                        field="mass" if "mass" in flag else None,
                        value=r.mass if "mass" in flag else None, **loc)
        if r.mass_kg is not None and not r.mass_verified:
            rep.add("warning", "unverified_value", "bom", "parsed mass has not been reviewed",
                    field="mass", value=r.mass, **loc)


def _check_bom_duplicates(res: DrawingResult, rep: _Report) -> None:
    """Repeated parts under different item numbers are legitimate; the same
    item number for different parts, or identical rows, are not."""
    by_item: dict[tuple, list[int]] = defaultdict(list)
    by_pn: dict[str, list[int]] = defaultdict(list)
    by_all: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(res.bom):
        if r.item:
            by_item[(r.page, r.table, r.item.strip().upper())].append(i)
        if r.part_number:
            by_pn[_norm(r.part_number)].append(i)
        by_all[(r.page, r.table, _norm(r.item), _norm(r.part_number), _norm(r.description),
                _norm(r.quantity))].append(i)
    for (page, _, item), idx in by_item.items():
        if len(idx) > 1:
            pns = {_norm(res.bom[i].part_number) or _norm(res.bom[i].description) for i in idx}
            sev = "error" if len(pns) > 1 else "warning"
            for i in idx:
                rep.add(sev, "duplicate", "bom", f"item number {item} appears {len(idx)} times"
                        + (" for different parts" if len(pns) > 1 else ""), field="item",
                        page=page, row=i, value=item)
    for key, idx in by_all.items():
        if len(idx) > 1 and any(key[2:]):
            for i in idx[1:]:
                rep.add("warning", "duplicate", "bom", "identical to another row - possible duplicate "
                        "extraction", page=res.bom[i].page, row=i)
    for pn, idx in by_pn.items():
        if len(idx) < 2:
            continue
        descs = {_norm(res.bom[i].description) for i in idx if res.bom[i].description}
        if len(descs) > 1:
            for i in idx:
                rep.add("warning", "inconsistent", "bom", f"part number {res.bom[i].part_number} has "
                        "different descriptions in different rows", field="part_number",
                        page=res.bom[i].page, row=i)
        elif len({res.bom[i].item for i in idx}) == len(idx):
            rep.add("info", "repeated_part", "bom", f"part {res.bom[idx[0]].part_number} listed under "
                    f"{len(idx)} item numbers (legitimate repeat?)", field="part_number",
                    page=res.bom[idx[0]].page, row=idx[0])


def _check_missing_rows(res: DrawingResult, rep: _Report) -> None:
    tables: dict[tuple, list] = defaultdict(list)
    for i, r in enumerate(res.bom):
        tables[(r.page, r.table)].append(r)
    for (page, t), rows in tables.items():
        nums = sorted({int(r.item) for r in rows if r.item and r.item.strip().isdigit()})
        if len(nums) < 2:
            continue
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        if missing:
            rep.add("warning", "missing_row", "bom", f"BOM table {t + 1}: item number(s) "
                    f"{missing[:15]}{' ...' if len(missing) > 15 else ''} missing from the sequence "
                    "- row(s) may not have been read", page=page, value=missing)
        if nums[0] > 1:
            rep.add("info", "missing_row", "bom", f"BOM table {t + 1} starts at item {nums[0]}",
                    page=page)


# ---------------------------------------------------------------- title block

def _check_title(res: DrawingResult, vcfg: dict, rep: _Report, review_conf: float) -> None:
    tb = res.title_block
    for f in vcfg["mandatory_title_fields"]:
        if not (tb.get(f) and tb[f].value):
            rep.add("warning", "missing_value", "title_block", f"title block '{f}' not found", field=f)
    for f, fv in tb.items():
        if f == "author_date":
            continue
        if getattr(fv, "needs_review", False):
            rep.add("warning", "ambiguous_value", "title_block",
                    f"'{f}' has conflicting or ambiguous candidates", field=f,
                    page=getattr(fv, "page", None), value=fv.value)
        if not fv.value:
            continue
        if fv.confidence < review_conf:
            rep.add("warning", "low_confidence", "title_block", f"'{f}' read with low confidence "
                    f"({fv.confidence:.2f})", field=f, value=fv.value)
        src = getattr(fv, "source", None)
        if src in ("page_text", "revision_table"):
            rep.add("warning" if f in vcfg["mandatory_title_fields"] else "info", "weak_source",
                    "title_block", f"'{f}' was not read from a labelled title-block cell "
                    f"(source: {src})", field=f, value=fv.value)
        if (f == "date" or f.endswith("_date")) and fv.normalized is None:
            rep.add("warning", "invalid_value", "title_block", f"'{f}' value '{fv.value}' is not a "
                    "valid date", field=f, value=fv.value)
        if f == "scale" and fv.normalized is None:
            rep.add("warning", "invalid_value", "title_block", f"scale '{fv.value}' not recognised",
                    field=f, value=fv.value)
        if f == "projection" and fv.normalized is None:
            rep.add("warning", "invalid_value", "title_block", f"projection '{fv.value}' is neither "
                    "first nor third angle", field=f, value=fv.value)
        if f == "drawing_number" and _OCR_JUNK.search(fv.value):
            rep.add("warning", "suspicious_value", "title_block", f"drawing number '{fv.value}' "
                    "contains OCR-suspicious characters", field=f, value=fv.value)


# ---------------------------------------------------------------- cross checks

def _cross_checks(res: DrawingResult, vcfg: dict, rep: _Report) -> None:
    from .extract import latest_revision
    tb = res.title_block
    # title revision vs revision history
    latest = latest_revision(res.revisions)
    rev = tb.get("revision")
    if rev and latest and latest.rev and getattr(rev, "source", None) != "revision_table":
        cur = str(rev.normalized or rev.value or "").upper()
        if cur and cur != latest.rev.upper():
            known = {(r.rev or "").upper() for r in res.revisions}
            sev = "warning" if cur in known else "error"
            rep.add(sev, "cross_check", "cross_check",
                    f"title block revision '{cur}' differs from latest revision-table entry "
                    f"'{latest.rev}'" + ("" if cur in known else " and is not in the history"),
                    field="revision", value=cur)
    seen: dict[str, int] = {}
    for revision_idx, r in enumerate(res.revisions):
        if "missing_or_unreadable_revision_identifier" in getattr(r, "flags", []):
            rep.add("warning", "ambiguous_value", "revision",
                    "revision row was preserved but its revision identifier could not be read",
                    page=r.page, row=revision_idx,
                    value=r.extra.get("raw_revision_candidate") or r.description)
        if r.rev:
            seen[r.rev.upper()] = seen.get(r.rev.upper(), 0) + 1
    for k, n in seen.items():
        if n > 1:
            rep.add("warning", "duplicate", "revision", f"revision '{k}' appears {n} times in the "
                    "revision table", value=k)

    # title-block weight vs BOM total
    w = tb.get("weight")
    if w and isinstance(w.normalized, (int, float)) and res.bom and \
            all(r.mass_kg is not None for r in res.bom):
        total = sum(r.mass_kg * (1 if r.extra.get("mass_basis") == "total"
                                 else (r.quantity_value or 1)) for r in res.bom)
        if total > 0 and abs(total - w.normalized) / max(total, w.normalized) > float(vcfg["mass_tolerance"]):
            rep.add("warning", "cross_check", "cross_check", f"title-block weight {w.normalized:g} kg "
                    f"differs from BOM total {total:.3g} kg", field="weight", value=w.normalized)

    # single-part drawing: title-block material vs BOM material
    m = tb.get("material")
    if m and m.value and len(res.bom) == 1 and res.bom[0].material and \
            _norm(m.value) != _norm(res.bom[0].material):
        rep.add("warning", "cross_check", "cross_check", f"title-block material '{m.value}' differs "
                f"from BOM material '{res.bom[0].material}'", field="material", value=m.value)

    # notes that reference BOM items
    items = {r.item.strip().upper() for r in res.bom if r.item}
    if items:
        for n in res.notes:
            for grp in _NOTE_ITEM.findall(n.get("text") or ""):
                for num in re.findall(r"\d{1,3}", grp):
                    if num not in items and num.lstrip("0") not in items:
                        rep.add("warning", "cross_check", "cross_check",
                                f"note references item {num}, which is not in the BOM",
                                page=n.get("page"), value=num)

    # BOM part numbers vs drawing number: an assembly should not list itself
    dn = tb.get("drawing_number")
    if dn and dn.value:
        for i, r in enumerate(res.bom):
            if r.part_number and _norm(r.part_number) == _norm(dn.value) and len(res.bom) > 1:
                rep.add("info", "cross_check", "cross_check", f"BOM row {r.item or i + 1} has the "
                        "drawing's own number as part number", page=r.page, row=i)
