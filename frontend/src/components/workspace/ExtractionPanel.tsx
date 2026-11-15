import React from 'react';
import type { DrawingResult, BomRow, RevisionRow, FieldValue } from '../../types/api';
import { TitleBlockTab } from './TitleBlockTab';
import { BomTableTab } from './BomTableTab';
import { SpecsTab } from './SpecsTab';
import { RevisionsTab } from './RevisionsTab';
import { NotesTab } from './NotesTab';
import { DimensionsTab } from './DimensionsTab';
import { ValidationTab } from './ValidationTab';
import {
  ShieldCheck,
  FileText,
  Table as TableIcon,
  Tag,
  GitCommit,
  StickyNote,
  Ruler,
} from 'lucide-react';

export type WorkspaceTabType =
  | 'title_block'
  | 'bom'
  | 'specs'
  | 'revisions'
  | 'notes'
  | 'dimensions'
  | 'validation';

interface ExtractionPanelProps {
  result: DrawingResult;
  activeTab: WorkspaceTabType;
  onTabChange: (tab: WorkspaceTabType) => void;
  // Title block
  onChangeTitleField: (key: string, val: string) => void;
  onLocateTitleField: (key: string) => void;
  focusedFieldKey?: string | null;
  // BOM
  onChangeBom: (bom: BomRow[]) => void;
  onLocateBomRow: (index: number) => void;
  selectedRowIndex?: number | null;
  // Revisions & Notes
  onChangeRevisions: (revs: RevisionRow[]) => void;
  onLocateRevision?: (index: number) => void;
  onChangeNotes: (notes: any[]) => void;
}

export const ExtractionPanel: React.FC<ExtractionPanelProps> = ({
  result,
  activeTab,
  onTabChange,
  onChangeTitleField,
  onLocateTitleField,
  focusedFieldKey,
  onChangeBom,
  onLocateBomRow,
  selectedRowIndex,
  onChangeRevisions,
  onLocateRevision,
  onChangeNotes,
}) => {
  const tabs = [
    { id: 'title_block', label: 'Title Block', icon: FileText, count: Object.keys(result.title_block || {}).length },
    { id: 'bom', label: 'BOM Table', icon: TableIcon, count: result.bom?.length ?? 0 },
    { id: 'specs', label: 'Specs & Finish', icon: Tag, count: result.specs?.length ?? 0 },
    { id: 'revisions', label: 'Revisions', icon: GitCommit, count: result.revisions?.length ?? 0 },
    { id: 'notes', label: 'Notes', icon: StickyNote, count: result.notes?.length ?? 0 },
    { id: 'dimensions', label: 'Dimensions', icon: Ruler, count: result.dimensions?.length ?? 0 },
    {
      id: 'validation',
      label: 'Review',
      icon: ShieldCheck,
      count: (result.validation?.counts.error ?? 0) + (result.validation?.counts.warning ?? 0),
    },
  ];

  return (
    <div className="flex flex-col h-full bg-white border-l border-[#E4EAF2] overflow-hidden">
      {/* Docsumo Tab Navigation Bar */}
      <div className="bg-[#F8FAFC] border-b border-[#E4EAF2] px-2 flex items-center gap-1 overflow-x-auto shrink-0 select-none">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id as WorkspaceTabType)}
              className={`flex items-center gap-1.5 py-2.5 px-3 border-b-2 text-xs font-medium whitespace-nowrap transition-colors ${
                isActive
                  ? 'border-[#2563EB] text-[#2563EB] bg-white'
                  : 'border-transparent text-[#64748B] hover:text-[#172B4D] hover:bg-slate-100/50'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              {tab.count !== undefined && tab.count > 0 && (
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isActive
                      ? 'bg-blue-100 text-[#2563EB] font-bold'
                      : 'bg-slate-200/80 text-slate-600'
                  }`}
                >
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Content Body */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'title_block' && (
          <TitleBlockTab
            titleBlock={result.title_block || {}}
            onChangeField={onChangeTitleField}
            onLocateField={onLocateTitleField}
            focusedFieldKey={focusedFieldKey}
            reviewFields={result.validation?.review_fields?.title_block}
          />
        )}

        {activeTab === 'bom' && (
          <BomTableTab
            bom={result.bom || []}
            onChangeBom={onChangeBom}
            onLocateRow={onLocateBomRow}
            selectedRowIndex={selectedRowIndex}
          />
        )}

        {activeTab === 'specs' && <SpecsTab specs={result.specs || []} />}

        {activeTab === 'revisions' && (
          <RevisionsTab
            revisions={result.revisions || []}
            onChangeRevisions={onChangeRevisions}
            onLocateRevision={onLocateRevision}
          />
        )}

        {activeTab === 'notes' && (
          <NotesTab notes={result.notes || []} onChangeNotes={onChangeNotes} />
        )}

        {activeTab === 'dimensions' && (
          <DimensionsTab dimensions={result.dimensions || []} />
        )}

        {activeTab === 'validation' && (
          <ValidationTab
            result={result}
            onLocateBomRow={onLocateBomRow}
            onLocateTitleField={onLocateTitleField}
          />
        )}
      </div>
    </div>
  );
};

