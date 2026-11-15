import React, { useMemo, useState } from 'react';
import type { DrawingResult, ValidationIssue } from '../../types/api';
import { AlertOctagon, AlertTriangle, Info, CheckCircle2, Crosshair } from 'lucide-react';

interface ValidationTabProps {
  result: DrawingResult;
  onLocateBomRow: (index: number) => void;
  onLocateTitleField: (key: string) => void;
}

const SEVERITY_STYLE: Record<string, { icon: React.ElementType; cls: string; label: string }> = {
  error: { icon: AlertOctagon, cls: 'text-rose-600 bg-rose-50 border-rose-200', label: 'Errors' },
  warning: { icon: AlertTriangle, cls: 'text-amber-600 bg-amber-50 border-amber-200', label: 'Review' },
  info: { icon: Info, cls: 'text-slate-500 bg-slate-50 border-slate-200', label: 'Info' },
};

const SECTION_LABEL: Record<string, string> = {
  bom: 'BOM',
  title_block: 'Title Block',
  revision: 'Revisions',
  notes: 'Notes',
  cross_check: 'Cross-check',
};

const pct = (v: number | null | undefined) => (v === null || v === undefined ? 'n/a' : `${Math.round(v * 100)}%`);

export const ValidationTab: React.FC<ValidationTabProps> = ({ result, onLocateBomRow, onLocateTitleField }) => {
  const v = result.validation;
  const [filter, setFilter] = useState<'all' | 'error' | 'warning' | 'info'>('all');

  const issues = useMemo(() => {
    const order = { error: 0, warning: 1, info: 2 } as Record<string, number>;
    return [...(v?.issues || [])]
      .filter((i) => filter === 'all' || i.severity === filter)
      .sort((a, b) => order[a.severity] - order[b.severity]);
  }, [v, filter]);

  if (!v) {
    return <div className="p-4 text-xs text-slate-400">No validation report for this result (re-run extraction).</div>;
  }

  const locate = (i: ValidationIssue) => {
    if (i.section === 'bom' && i.row !== null && i.row !== undefined) onLocateBomRow(i.row);
    else if (i.section === 'title_block' && i.field) onLocateTitleField(i.field);
  };
  const canLocate = (i: ValidationIssue) =>
    (i.section === 'bom' && i.row !== null && i.row !== undefined) || (i.section === 'title_block' && !!i.field);

  return (
    <div className="p-4 space-y-3 text-xs">
      <div className="flex items-center justify-between pb-2 border-b border-[#E4EAF2]">
        <div>
          <h4 className="font-semibold text-sm text-[#172B4D]">Validation &amp; Review</h4>
          <p className="text-[#64748B] text-[11px]">
            Values checked against source evidence, completeness and cross-section consistency
          </p>
        </div>
        {v.review_required ? (
          <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-semibold">Needs review</span>
        ) : (
          <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-semibold flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" /> No issues
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="p-2 rounded border border-[#E4EAF2]">
          <div className="text-[10px] text-slate-500">BOM completeness (mandatory fields)</div>
          <div className="text-sm font-semibold text-[#172B4D]">{pct(v.completeness.bom)}</div>
        </div>
        <div className="p-2 rounded border border-[#E4EAF2]">
          <div className="text-[10px] text-slate-500">Title block completeness</div>
          <div className="text-sm font-semibold text-[#172B4D]">{pct(v.completeness.title_block)}</div>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        {(['all', 'error', 'warning', 'info'] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`px-2 py-0.5 rounded border text-[11px] ${
              filter === f ? 'bg-blue-50 border-[#2563EB] text-[#2563EB]' : 'border-[#E4EAF2] text-slate-600'
            }`}
          >
            {f === 'all' ? `All (${v.issues.length})` : `${SEVERITY_STYLE[f].label} (${v.counts[f]})`}
          </button>
        ))}
      </div>

      <div className="space-y-1.5">
        {issues.length === 0 && <div className="text-slate-400 py-6 text-center">Nothing to show.</div>}
        {issues.map((i, k) => {
          const st = SEVERITY_STYLE[i.severity] || SEVERITY_STYLE.info;
          const Icon = st.icon;
          return (
            <div key={k} className={`flex items-start gap-2 p-2 rounded border ${st.cls}`}>
              <Icon className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-[#172B4D]">{i.message}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  {SECTION_LABEL[i.section] || i.section}
                  {i.field ? ` · ${i.field}` : ''}
                  {i.page ? ` · page ${i.page}` : ''}
                  {i.row !== null && i.row !== undefined && result.bom?.[i.row]
                    ? ` · row ${result.bom[i.row].item || i.row + 1}`
                    : ''}
                  {` · ${i.category.replace(/_/g, ' ')}`}
                </div>
              </div>
              {canLocate(i) && (
                <button
                  type="button"
                  onClick={() => locate(i)}
                  title="Show in drawing"
                  className="p-1 text-slate-400 hover:text-[#2563EB] rounded"
                >
                  <Crosshair className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
