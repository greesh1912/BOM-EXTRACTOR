"""Export service for BOMX extraction results (Excel, CSV, JSON)."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from ..storage import EXPORTS_DIR, storage

# Styling constants matching BOMX enterprise aesthetic
HDR_FILL = PatternFill("solid", fgColor="172B4D")  # Dark navy
HDR_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
ROW_BAND = PatternFill("solid", fgColor="F8FAFC")
BORDER_THIN = Side(border_style="thin", color="E4EAF2")
CELL_BORDER = Border(left=BORDER_THIN, right=BORDER_THIN, top=BORDER_THIN, bottom=BORDER_THIN)

_BOM_STD = [("item", "Item"), ("part_number", "Part Number"), ("description", "Description"),
            ("quantity", "Quantity"), ("material", "Material Grade"), ("mass", "Mass / Weight"),
            ("remarks", "Remarks")]
_BOM_EXTRA_LABELS = {"size": "Size", "supplier": "Supplier", "finish": "Finish",
                     "mass_basis": "Mass Basis (Unit / Total)"}
_BOM_INTERNAL_EXTRA = {"mass_candidate"}
_VALIDATION_HEADERS = ["Severity", "Section", "Field", "Page", "Row", "Message", "Review Status", "Original Value"]


def _validation_rows(res: Dict[str, Any]) -> List[list]:
    issues = ((res.get("validation") or {}).get("issues") or [])
    return [[i.get("severity", ""), i.get("section", ""), i.get("field", ""),
             i.get("page", ""), i.get("row", ""), i.get("message", ""),
             "Needs Review" if i.get("severity") in ("error", "warning") else "Informational",
             json.dumps(i.get("value"), ensure_ascii=False) if i.get("value") is not None else ""]
            for i in issues]


def _bom_table(res: Dict[str, Any], pct: bool) -> tuple[List[str], List[list]]:
    """Header + rows for the BOM sheet / CSV: standard columns, every other
    column the drawing's BOM tables actually have (size, supplier, finish,
    unrecognised headers ...), and page / table so multiple BOMs stay apart."""
    rows = res.get("bom", []) or []
    std_keys = {k for k, _ in _BOM_STD}
    extras: List[str] = []
    for k in list(res.get("bom_columns") or []) + [k for r in rows for k in (r.get("extra") or {})]:
        base = k
        if base not in std_keys and base not in _BOM_INTERNAL_EXTRA and base not in extras:
            extras.append(base)
    headers = ["Page", "Table"] + [h for _, h in _BOM_STD] + \
        [_BOM_EXTRA_LABELS.get(k, k.replace("_", " ").title()) for k in extras] + ["Flags", "Confidence"]
    out = []
    for r in rows:
        extra = r.get("extra") or {}
        conf = r.get("confidence")
        conf = 1.0 if conf is None else conf
        out.append([r.get("page", ""), (r.get("table") or 0) + 1]
                   + [r.get(k) or "" for k, _ in _BOM_STD]
                   + [extra.get(k, "") for k in extras]
                   + ["; ".join(r.get("flags") or []),
                      f"{round(conf * 100, 1)}%" if pct else conf])
    return headers, out


def generate_excel_export(drawing_id: str) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res = storage.get_result(drawing_id)
    meta = storage.get_metadata(drawing_id)
    if not res:
        raise ValueError(f"No result found for drawing {drawing_id}")

    out_path = EXPORTS_DIR / f"{meta.filename if meta else drawing_id}_extracted.xlsx"
    wb = Workbook()

    # 1. Sheet: Summary
    ws_sum = wb.active
    ws_sum.title = "Summary"
    ws_sum.append(["Drawing Information Summary", ""])
    ws_sum.cell(1, 1).font = Font(name="Arial", size=13, bold=True, color="172B4D")

    summary_rows = [
        ("Source File", meta.filename if meta else res.get("source_file")),
        ("Drawing Number", res.get("summary", {}).get("drawing_number") or res.get("title_block", {}).get("drawing_number", {}).get("value")),
        ("Drawing Title", res.get("summary", {}).get("drawing_title") or res.get("title_block", {}).get("drawing_title", {}).get("value")),
        ("Revision", res.get("summary", {}).get("revision") or res.get("title_block", {}).get("revision", {}).get("value")),
        ("Status", "Verified" if meta and meta.verified else ("Needs Review" if res.get("summary", {}).get("review_required") else "Extracted")),
        ("Mean OCR Confidence", f"{round((res.get('summary', {}).get('mean_confidence') or 0) * 100, 1)}%"),
        ("Structural Confidence", f"{round((res.get('summary', {}).get('structural_confidence') or 0) * 100, 1)}%"),
        ("Total BOM Items", len(res.get("bom", []))),
        ("Total Pages", len(res.get("pages", [1]))),
        ("Processing Time (s)", f"{res.get('processing_seconds', 0)}s"),
        ("Review Reasons", res.get("summary", {}).get("review_reasons") or "None"),
    ]

    for k, v in summary_rows:
        ws_sum.append([k, str(v) if v is not None else ""])

    for r in range(2, len(summary_rows) + 3):
        ws_sum.cell(r, 1).font = Font(name="Arial", size=10, bold=True, color="172B4D")
        ws_sum.cell(r, 1).fill = PatternFill("solid", fgColor="F1F5F9")
        ws_sum.cell(r, 1).border = CELL_BORDER
        ws_sum.cell(r, 2).border = CELL_BORDER

    ws_sum.column_dimensions["A"].width = 25
    ws_sum.column_dimensions["B"].width = 50

    # 2. Sheet: Bill of Materials
    ws_bom = wb.create_sheet(title="Bill of Materials")
    bom_headers, bom_rows_data = _bom_table(res, pct=True)
    ws_bom.append(bom_headers)
    for col_idx in range(1, len(bom_headers) + 1):
        cell = ws_bom.cell(1, col_idx)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_idx, row_data in enumerate(bom_rows_data, start=2):
        ws_bom.append(row_data)
        fill = ROW_BAND if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for c in range(1, len(row_data) + 1):
            cell = ws_bom.cell(row_idx, c)
            cell.border = CELL_BORDER
            if fill.fill_type:
                cell.fill = fill

    # Auto-adjust column widths
    for col in ws_bom.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws_bom.column_dimensions[col_letter].width = max(max_len + 3, 12)

    # 3. Sheet: Title Block
    ws_tb = wb.create_sheet(title="Title Block")
    tb_headers = ["Field Name", "Extracted Value", "Confidence", "Raw OCR"]
    ws_tb.append(tb_headers)
    for col_idx in range(1, len(tb_headers) + 1):
        cell = ws_tb.cell(1, col_idx)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT

    tb = res.get("title_block", {})
    for row_idx, (k, val_obj) in enumerate(tb.items(), start=2):
        if isinstance(val_obj, dict):
            val = val_obj.get("value", "")
            conf = f"{round((val_obj.get('confidence') or 0) * 100, 1)}%"
            raw = val_obj.get("raw", "")
        else:
            val = str(val_obj)
            conf = "100%"
            raw = str(val_obj)
        ws_tb.append([k.replace("_", " ").title(), val, conf, raw])
        for c in range(1, 5):
            ws_tb.cell(row_idx, c).border = CELL_BORDER

    ws_tb.column_dimensions["A"].width = 25
    ws_tb.column_dimensions["B"].width = 35
    ws_tb.column_dimensions["C"].width = 15
    ws_tb.column_dimensions["D"].width = 35

    # 4. Sheet: Revisions & Notes
    ws_rev = wb.create_sheet(title="Revisions & Notes")
    ws_rev.append(["Revision History"])
    ws_rev.cell(1, 1).font = Font(name="Arial", size=11, bold=True)
    rev_headers = ["Rev", "Description", "Date", "Approved By", "Confidence", "Flags",
                   "Raw Revision Cell", "Raw Date Cell"]
    ws_rev.append(rev_headers)
    for c in range(1, len(rev_headers) + 1):
        ws_rev.cell(2, c).fill = HDR_FILL
        ws_rev.cell(2, c).font = HDR_FONT

    revisions = res.get("revisions", [])
    for row_idx, r in enumerate(revisions, start=3):
        ws_rev.append([
            r.get("rev", ""),
            r.get("description", ""),
            r.get("date", ""),
            r.get("approved", ""),
            f"{round((r.get('confidence') or 1.0) * 100, 1)}%",
            "; ".join(r.get("flags") or []),
            (r.get("extra") or {}).get("raw_revision_text", ""),
            (r.get("extra") or {}).get("raw_date_text", ""),
        ])

    notes = res.get("notes", [])
    start_note_row = ws_rev.max_row + 2
    ws_rev.cell(start_note_row, 1, "Technical Notes").font = Font(name="Arial", size=11, bold=True)
    ws_rev.cell(start_note_row + 1, 1, "#").fill = HDR_FILL
    ws_rev.cell(start_note_row + 1, 1).font = HDR_FONT
    ws_rev.cell(start_note_row + 1, 2, "Note Text").fill = HDR_FILL
    ws_rev.cell(start_note_row + 1, 2).font = HDR_FONT

    for n_idx, n in enumerate(notes, start=start_note_row + 2):
        ws_rev.append([n.get("number") or str(n_idx - start_note_row - 1), n.get("text", "")])

    ws_rev.column_dimensions["A"].width = 10
    ws_rev.column_dimensions["B"].width = 60
    ws_rev.column_dimensions["C"].width = 15
    ws_rev.column_dimensions["D"].width = 20

    # Structured validation findings retain the original extracted value.
    ws_val = wb.create_sheet(title="Validation")
    ws_val.append(_VALIDATION_HEADERS)
    for c in range(1, len(_VALIDATION_HEADERS) + 1):
        ws_val.cell(1, c).fill = HDR_FILL
        ws_val.cell(1, c).font = HDR_FONT
    for row in _validation_rows(res):
        ws_val.append(row)
    for col_idx, width in enumerate((14, 18, 22, 10, 10, 70, 18, 32), start=1):
        ws_val.column_dimensions[get_column_letter(col_idx)].width = width
    for row in ws_val.iter_rows(min_row=2):
        for cell in row:
            cell.border = CELL_BORDER

    wb.save(out_path)
    return out_path


def generate_csv_export(drawing_id: str) -> io.BytesIO:
    """Generate a ZIP containing CSV files for all extracted sections."""
    res = storage.get_result(drawing_id)
    if not res:
        raise ValueError(f"No result found for drawing {drawing_id}")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # BOM CSV
        bom_buf = io.StringIO()
        bom_writer = csv.writer(bom_buf)
        bom_headers, bom_rows_data = _bom_table(res, pct=False)
        bom_writer.writerow(bom_headers)
        for row_data in bom_rows_data:
            bom_writer.writerow(row_data)
        zf.writestr("bill_of_materials.csv", bom_buf.getvalue())

        # Title Block CSV
        tb_buf = io.StringIO()
        tb_writer = csv.writer(tb_buf)
        tb_writer.writerow(["Field", "Value", "Confidence"])
        for k, v in res.get("title_block", {}).items():
            val = v.get("value") if isinstance(v, dict) else str(v)
            conf = v.get("confidence") if isinstance(v, dict) else 1.0
            tb_writer.writerow([k, val, conf])
        zf.writestr("title_block.csv", tb_buf.getvalue())

        # Revisions CSV
        rev_buf = io.StringIO()
        rev_writer = csv.writer(rev_buf)
        rev_writer.writerow(["Rev", "Description", "Date", "Approved", "Confidence", "Flags",
                             "Raw Revision Cell", "Raw Date Cell"])
        for r in res.get("revisions", []):
            extra = r.get("extra") or {}
            rev_writer.writerow([r.get("rev"), r.get("description"), r.get("date"), r.get("approved"),
                                 r.get("confidence"), "; ".join(r.get("flags") or []),
                                 extra.get("raw_revision_text", ""), extra.get("raw_date_text", "")])
        zf.writestr("revisions.csv", rev_buf.getvalue())

        validation_buf = io.StringIO()
        validation_writer = csv.writer(validation_buf)
        validation_writer.writerow(_VALIDATION_HEADERS)
        validation_writer.writerows(_validation_rows(res))
        zf.writestr("validation.csv", validation_buf.getvalue())

    zip_buffer.seek(0)
    return zip_buffer

