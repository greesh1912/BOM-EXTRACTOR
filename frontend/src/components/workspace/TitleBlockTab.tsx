import React from 'react';
import type { FieldValue } from '../../types/api';
import { Crosshair, Edit3 } from 'lucide-react';

interface TitleBlockTabProps {
  titleBlock: Record<string, FieldValue>;
  onChangeField: (key: string, value: string) => void;
  onLocateField: (key: string) => void;
  focusedFieldKey?: string | null;
  reviewFields?: string[]; // fields flagged by validation (error / warning)
}

// Always shown (empty when not found) - the core title block attributes
const CORE_TITLE_FIELDS = [
  { key: 'drawing_number', label: 'Drawing Number' },
  { key: 'drawing_title', label: 'Drawing Title' },
  { key: 'revision', label: 'Revision (current)' },
  { key: 'drawn_by', label: 'Drawn By (Author)' },
  { key: 'date', label: 'Date' },
  { key: 'scale', label: 'Scale' },
  { key: 'projection', label: 'Projection Type' },
  { key: 'units', label: 'Units' },
];

// Optional fields are shown when extracted; Units is always shown for the POC.
const OPTIONAL_TITLE_FIELDS: Record<string, string> = {
  checked_by: 'Checked By',
  checked_date: 'Checked Date',
  approved_by: 'Approved By',
  approved_date: 'Approved Date',
  drawn_date: 'Drawn Date',
  sheet: 'Sheet',
  size: 'Sheet Size',
  part_number: 'Part Number',
  material: 'Material',
  finish: 'Finish',
  heat_treatment: 'Heat Treatment',
  weight: 'Weight',
  tolerance: 'General Tolerance',
  company: 'Company',
};

// derived / internal keys that duplicate other fields
const HIDDEN_KEYS = new Set(['author_date']);

const SOURCE_NOTE: Record<string, string> = {
  revision_table: 'inferred from latest revision-table entry (no REV in title block)',
  page_text: 'found in drawing text outside the title block',
  cad_attribute: 'from CAD block attribute',
};

const prettify = (key: string) =>
  key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export const TitleBlockTab: React.FC<TitleBlockTabProps> = ({
  titleBlock,
  onChangeField,
  onLocateField,
  focusedFieldKey,
  reviewFields = [],
}) => {
  const coreKeys = new Set(CORE_TITLE_FIELDS.map((f) => f.key));
  const extraFields = Object.keys(titleBlock || {})
    .filter((k) => !coreKeys.has(k) && !HIDDEN_KEYS.has(k) &&
      (titleBlock[k]?.value || titleBlock[k]?.needs_review))
    .sort((a, b) => {
      const ia = Object.keys(OPTIONAL_TITLE_FIELDS).indexOf(a);
      const ib = Object.keys(OPTIONAL_TITLE_FIELDS).indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    })
    .map((k) => ({ key: k, label: OPTIONAL_TITLE_FIELDS[k] || prettify(k) }));
  const targetFields = [...CORE_TITLE_FIELDS, ...extraFields];

  const getFieldObj = (key: string): FieldValue =>
    titleBlock[key] || { value: '', confidence: 0, raw: '' };

  const handleFieldChange = (key: string, val: string) => onChangeField(key, val);

  const detectedCount = targetFields.filter((f) => {
    const obj = getFieldObj(f.key);
    return Boolean(obj && obj.value && obj.value.trim());
  }).length;

  return (
    <div className="p-4 space-y-3.5 text-xs">
      <div className="flex items-center justify-between pb-2 border-b border-[#E4EAF2]">
        <div>
          <h4 className="font-semibold text-sm text-[#172B4D]">Title Block Metadata</h4>
          <p className="text-[#64748B] text-[11px]">
            Extracted drawing header attributes
          </p>
        </div>
        <span className="text-[11px] text-[#64748B] font-mono">
          {detectedCount} of {targetFields.length} fields detected
        </span>
      </div>

      <div className="space-y-3">
        {targetFields.map(({ key, label }) => {
          const fieldObj = getFieldObj(key);
          const val = fieldObj.value ?? '';
          const hasValue = Boolean(val.trim());
          const collectCandidates = (items: Array<Record<string, any>>): string[] => items.flatMap((c) => [
            String(c.value ?? c.raw ?? '').trim(),
            ...collectCandidates(Array.isArray(c.candidates) ? c.candidates : []),
          ]);
          const candidates = [...new Set(collectCandidates(fieldObj.candidates || []))]
            .filter((candidate) => candidate && candidate !== val);
          const hasBbox = fieldObj.bbox && fieldObj.bbox.length === 4;
          const isFocused = focusedFieldKey === key;

          return (
            <div
              key={key}
              id={`tb-field-${key}`}
              className={`p-3 rounded-md border transition-all duration-150 ${
                isFocused
                  ? 'bg-blue-50/60 border-[#2563EB] shadow-xs'
                  : 'bg-white border-[#E4EAF2] hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <label className="font-semibold text-[#172B4D] flex items-center gap-1.5">
                  <span>{label}</span>
                  {reviewFields.includes(key) && (
                    <span className="text-[10px] font-medium px-1.5 rounded bg-amber-100 text-amber-800">
                      {val ? 'needs review' : 'missing'}
                    </span>
                  )}
                </label>
                <div className="flex items-center gap-2">
                  {fieldObj.source === 'manual' ? (
                    <span className="text-[10px] font-medium text-blue-700 border border-blue-200 bg-blue-50 rounded px-1.5 py-0.5">Manual entry · unverified</span>
                  ) : !hasValue ? (
                    <span className="text-[10px] font-medium text-slate-500 border border-slate-200 bg-slate-50 rounded px-1.5 py-0.5">Missing</span>
                  ) : null}
                  {hasBbox && (
                    <button
                      type="button"
                      onClick={() => onLocateField(key)}
                      title="Locate on drawing"
                      className="p-1 text-slate-400 hover:text-[#2563EB] hover:bg-blue-50 rounded transition-colors"
                    >
                      <Crosshair className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>

              <div className="relative">
                <input
                  type="text"
                  value={val}
                  onChange={(e) => handleFieldChange(key, e.target.value)}
                  placeholder={`Enter ${label.toLowerCase()}...`}
                  className="w-full px-2.5 py-1.5 text-xs text-[#172B4D] font-mono bg-slate-50/50 border border-[#E4EAF2] rounded focus:bg-white focus:outline-none focus:ring-1 focus:ring-[#2563EB]"
                />
              </div>

              {fieldObj.source && SOURCE_NOTE[fieldObj.source] && (
                <div className="mt-1 text-[10px] text-amber-700">
                  {SOURCE_NOTE[fieldObj.source]}
                </div>
              )}

              {fieldObj.needs_review && candidates.length > 0 && (
                <div className="mt-1 text-[10px] text-amber-800">
                  Candidates to review: {candidates.join(' / ')}
                </div>
              )}

              {key === 'scale' && fieldObj.normalized && fieldObj.normalized !== val && (
                <div className="mt-1 text-[10px] text-[#64748B] font-mono">
                  <span className="text-slate-400">Normalized:</span> {String(fieldObj.normalized)}
                </div>
              )}

              {fieldObj.raw && fieldObj.raw !== val && (
                <div className="mt-1 text-[10px] text-[#64748B] font-mono">
                  <span className="text-slate-400">OCR Raw:</span> {fieldObj.raw}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

