"""Extraction accuracy against manually verified reference drawings (Issue 6.6).

Reference file (JSON, one per drawing, same stem as the drawing):

    {"title_block": {"drawing_number": "AB-1001", "revision": "C", "scale": "1:2", ...},
     "bom": [{"item": "1", "part_number": "PN-100", "description": "BASE PLATE",
              "quantity": "2", "material": "S355"}, ...]}

Prediction: an extraction result JSON (``DrawingResult.to_dict()``, as written
by ``bomx extract``) or the drawing itself (it is then processed).

Reports per drawing and overall:
  * BOM rows: matched / missed (omissions) / extra (spurious), precision, recall, F1
  * BOM fields: accuracy per column over matched rows, with every wrong value
  * title block: accuracy per field, wrong and missing fields
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

BOM_FIELDS = ["item", "part_number", "description", "quantity", "material", "mass",
              "size", "supplier", "finish", "remarks"]


def _norm(field: str, v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):  # FieldValue dict
        v = v.get("normalized") or v.get("value")
    s = str(v).strip().upper()
    if field in ("part_number", "drawing_number", "item", "revision"):
        return re.sub(r"[\s\-_./]+", "", s).lstrip("0") or s
    if field in ("quantity", "mass"):
        m = re.search(r"\d+(?:[.,]\d+)?", s)
        if m:
            try:
                return f"{float(m.group(0).replace(',', '.')):g}"
            except ValueError:
                pass
    if field == "scale":
        return s.replace(" ", "").replace("/", ":")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _row_value(row: dict, f: str) -> Any:
    v = row.get(f)
    if v in (None, "") and isinstance(row.get("extra"), dict):
        v = row["extra"].get(f)
    return v


def _match_rows(pred: list[dict], ref: list[dict]) -> list[tuple[int, int]]:
    """Greedy one-to-one matching: item number, then part number, then
    description similarity (>= 0.8)."""
    pairs, used_p, used_r = [], set(), set()
    for key, thr in (("item", None), ("part_number", None), ("description", 0.8)):
        for ri, r in enumerate(ref):
            if ri in used_r:
                continue
            rv = _norm(key, _row_value(r, key))
            if not rv:
                continue
            best, best_s = None, 0.0
            for pi, p in enumerate(pred):
                if pi in used_p:
                    continue
                pv = _norm(key, _row_value(p, key))
                s = (1.0 if pv == rv else 0.0) if thr is None else _similar(pv, rv)
                if s > best_s:
                    best, best_s = pi, s
            if best is not None and best_s >= (thr or 1.0):
                pairs.append((best, ri))
                used_p.add(best)
                used_r.add(ri)
    return pairs


def compare(pred: dict, ref: dict) -> dict:
    """Compare one prediction with its reference."""
    p_bom, r_bom = pred.get("bom") or [], ref.get("bom") or []
    pairs = _match_rows(p_bom, r_bom)
    matched_r = {ri for _, ri in pairs}
    matched_p = {pi for pi, _ in pairs}
    fields = [f for f in BOM_FIELDS if any(_row_value(r, f) not in (None, "") for r in r_bom)]
    f_ok: dict[str, int] = defaultdict(int)
    f_tot: dict[str, int] = defaultdict(int)
    errors: list[dict] = []
    for pi, ri in pairs:
        for f in fields:
            rv = _row_value(r_bom[ri], f)
            if rv in (None, ""):
                continue
            f_tot[f] += 1
            pv = _row_value(p_bom[pi], f)
            if _norm(f, pv) == _norm(f, rv):
                f_ok[f] += 1
            else:
                errors.append({"section": "bom", "row": r_bom[ri].get("item") or ri + 1, "field": f,
                               "expected": rv, "extracted": pv,
                               "kind": "missing" if pv in (None, "") else "wrong"})
    omissions = [r_bom[i] for i in range(len(r_bom)) if i not in matched_r]
    extras = [p_bom[i] for i in range(len(p_bom)) if i not in matched_p]
    tp = len(pairs)
    prec = tp / len(p_bom) if p_bom else (1.0 if not r_bom else 0.0)
    rec = tp / len(r_bom) if r_bom else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    p_tb, r_tb = pred.get("title_block") or {}, ref.get("title_block") or {}
    tb_ok, tb_res = 0, {}
    for f, rv in r_tb.items():
        if rv in (None, ""):
            continue
        pv = p_tb.get(f)
        ok = _norm(f, pv) == _norm(f, rv)
        tb_ok += ok
        tb_res[f] = ok
        if not ok:
            pval = pv.get("value") if isinstance(pv, dict) else pv
            errors.append({"section": "title_block", "field": f, "expected": rv, "extracted": pval,
                           "kind": "missing" if pval in (None, "") else "wrong"})
    cell_tot = sum(f_tot.values())
    return {
        "bom": {"reference_rows": len(r_bom), "extracted_rows": len(p_bom), "matched": tp,
                "missed": len(omissions), "extra": len(extras),
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                "field_accuracy": {f: round(f_ok[f] / f_tot[f], 4) for f in f_tot},
                "cell_accuracy": round(sum(f_ok.values()) / cell_tot, 4) if cell_tot else None,
                "_cells": (sum(f_ok.values()), cell_tot),
                "missed_rows": omissions, "extra_rows": extras},
        "title_block": {"fields": len(tb_res), "correct": tb_ok,
                        "accuracy": round(tb_ok / len(tb_res), 4) if tb_res else None,
                        "per_field": tb_res},
        "errors": errors,
    }


def _load_pred(path: Path, cfg: Optional[dict]) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    from .config import DEFAULT_CONFIG
    from .pipeline import process_file
    import copy
    return process_file(path, cfg or copy.deepcopy(DEFAULT_CONFIG)).to_dict()


def evaluate(pred_path: str | Path, ref_path: str | Path, cfg: Optional[dict] = None) -> dict:
    """Evaluate one pair (files) or a set (directories: references *.json are
    paired with predictions/drawings of the same stem)."""
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    if ref_path.is_dir():
        refs = sorted(ref_path.glob("*.json"))
        preds = {p.stem.replace("_result", ""): p for p in Path(pred_path).iterdir() if p.is_file()}
        pairs = [(preds.get(r.stem), r) for r in refs]
    else:
        pairs = [(pred_path, ref_path)]
    per, missing = {}, []
    for p, r in pairs:
        if p is None:
            missing.append(r.stem)
            continue
        per[r.stem] = compare(_load_pred(p, cfg), json.loads(r.read_text(encoding="utf-8")))
    # micro-averaged totals
    tp = sum(d["bom"]["matched"] for d in per.values())
    n_p = sum(d["bom"]["extracted_rows"] for d in per.values())
    n_r = sum(d["bom"]["reference_rows"] for d in per.values())
    c_ok = sum(d["bom"]["_cells"][0] for d in per.values())
    c_tot = sum(d["bom"]["_cells"][1] for d in per.values())
    t_ok = sum(d["title_block"]["correct"] for d in per.values())
    t_tot = sum(d["title_block"]["fields"] for d in per.values())
    prec = tp / n_p if n_p else None
    rec = tp / n_r if n_r else None
    for d in per.values():
        d["bom"].pop("_cells", None)
    return {
        "drawings": len(per),
        "unpaired_references": missing,
        "overall": {
            "bom_row_precision": round(prec, 4) if prec is not None else None,
            "bom_row_recall": round(rec, 4) if rec is not None else None,
            "bom_row_f1": round(2 * prec * rec / (prec + rec), 4) if prec and rec else None,
            "bom_cell_accuracy": round(c_ok / c_tot, 4) if c_tot else None,
            "title_block_accuracy": round(t_ok / t_tot, 4) if t_tot else None,
            "rows_missed": n_r - tp, "rows_extra": n_p - tp,
            "errors": sum(len(d["errors"]) for d in per.values()),
        },
        "per_drawing": per,
    }
