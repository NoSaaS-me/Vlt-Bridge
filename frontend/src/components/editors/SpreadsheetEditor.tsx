/**
 * SpreadsheetEditor — CSV editor with react-data-grid v7, papaparse, and basic formula evaluation.
 * Supports SUM, AVERAGE, MIN, MAX, COUNT and simple arithmetic on cell references.
 */
import { useState, useEffect, useCallback } from 'react';
import { DataGrid } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import Papa from 'papaparse';
// Basic formula implementations (inline — avoids UMD/CJS dependency issues with Vite)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const formulas: Record<string, (...args: any[]) => number> = {
  SUM: (ns: number[]) => ns.reduce((a: number, b: number) => a + b, 0),
  AVERAGE: (ns: number[]) => ns.length ? ns.reduce((a: number, b: number) => a + b, 0) / ns.length : 0,
  AVG: (ns: number[]) => ns.length ? ns.reduce((a: number, b: number) => a + b, 0) / ns.length : 0,
  MIN: (ns: number[]) => Math.min(...ns),
  MAX: (ns: number[]) => Math.max(...ns),
  COUNT: (ns: number[]) => ns.length,
  ABS: (n: number) => Math.abs(n),
  ROUND: (n: number, d = 0) => Math.round(n * Math.pow(10, d)) / Math.pow(10, d),
};
import { Save, Plus, Trash2, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

// ── helpers ──────────────────────────────────────────────────────────────────

function getApiBase(): string {
  return (window as unknown as { API_BASE_URL?: string }).API_BASE_URL || '';
}

function getAuthToken(): string {
  return localStorage.getItem('auth_token') || '';
}

/** Convert 0-based column index to spreadsheet letter(s): 0 → A, 25 → Z, 26 → AA */
function colName(idx: number): string {
  let name = '';
  let n = idx;
  do {
    name = String.fromCharCode(65 + (n % 26)) + name;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return name;
}

/** Parse a cell reference like "A1" to { row, col } (0-based). Returns null on bad input. */
function parseCellRef(ref: string): { row: number; col: number } | null {
  const m = ref.trim().match(/^([A-Z]+)(\d+)$/i);
  if (!m) return null;
  const colStr = m[1].toUpperCase();
  const rowIdx = parseInt(m[2], 10) - 1; // 1-based → 0-based
  let colIdx = 0;
  for (let i = 0; i < colStr.length; i++) {
    colIdx = colIdx * 26 + (colStr.charCodeAt(i) - 64);
  }
  colIdx -= 1; // 1-based → 0-based
  return { row: rowIdx, col: colIdx };
}

/**
 * Expand an A1:B3 range into an array of cell references.
 * Returns null if not a valid range.
 */
function expandRange(range: string): string[] | null {
  const parts = range.split(':');
  if (parts.length !== 2) return null;
  const start = parseCellRef(parts[0]);
  const end = parseCellRef(parts[1]);
  if (!start || !end) return null;
  const refs: string[] = [];
  for (let r = start.row; r <= end.row; r++) {
    for (let c = start.col; c <= end.col; c++) {
      refs.push(`${colName(c)}${r + 1}`);
    }
  }
  return refs;
}

/**
 * Resolve a cell reference or range string to numeric values from rows[][].
 * Returns an array of numbers (NaN for non-numeric cells, out-of-range cells are 0).
 */
function resolveArg(arg: string, rows: string[][]): number[] {
  const trimmed = arg.trim();
  if (trimmed.includes(':')) {
    const refs = expandRange(trimmed);
    if (!refs) return [parseFloat(trimmed)];
    return refs.map((ref) => {
      const pos = parseCellRef(ref);
      if (!pos) return NaN;
      const cell = rows[pos.row]?.[pos.col] ?? '';
      return parseFloat(cell);
    });
  }
  const pos = parseCellRef(trimmed);
  if (pos) {
    const cell = rows[pos.row]?.[pos.col] ?? '';
    return [parseFloat(cell)];
  }
  // Literal number
  return [parseFloat(trimmed)];
}

/**
 * Evaluate a formula string (starting with '=') against the current rows.
 * Supports: SUM, AVERAGE/AVG, MIN, MAX, COUNT, and simple arithmetic (+,-,*,/).
 * Returns the computed value as a string, or the original formula on any error.
 */
function evalFormula(formula: string, rows: string[][]): string {
  if (!formula.startsWith('=')) return formula;
  const expr = formula.slice(1).trim();

  try {
    // Match function calls: NAME(args)
    const fnMatch = expr.match(/^([A-Z]+)\((.+)\)$/i);
    if (fnMatch) {
      const fnName = fnMatch[1].toUpperCase();
      const argsStr = fnMatch[2];
      // Split top-level commas (not inside parens)
      const argParts: string[] = [];
      let depth = 0;
      let current = '';
      for (const ch of argsStr) {
        if (ch === '(') { depth++; current += ch; }
        else if (ch === ')') { depth--; current += ch; }
        else if (ch === ',' && depth === 0) { argParts.push(current.trim()); current = ''; }
        else { current += ch; }
      }
      if (current.trim()) argParts.push(current.trim());

      // Resolve all args to number arrays
      const numbers = argParts.flatMap((arg) => resolveArg(arg, rows)).filter((n) => !isNaN(n));

      let result: number;
      switch (fnName) {
        case 'SUM':
          result = formulas.SUM(numbers) as number;
          break;
        case 'AVERAGE':
        case 'AVG':
          result = formulas.AVERAGE(numbers) as number;
          break;
        case 'MIN':
          result = formulas.MIN(numbers) as number;
          break;
        case 'MAX':
          result = formulas.MAX(numbers) as number;
          break;
        case 'COUNT':
          result = formulas.COUNT(numbers) as number;
          break;
        case 'ABS':
          result = formulas.ABS(numbers[0] ?? 0) as number;
          break;
        case 'ROUND':
          result = formulas.ROUND(numbers[0] ?? 0, numbers[1] ?? 0) as number;
          break;
        default:
          return formula; // Unknown function — return raw
      }
      return isNaN(result) ? '#VALUE!' : String(result);
    }

    // Simple arithmetic with cell references: A1+B2*3 etc.
    // Replace cell references with their numeric values
    const resolved = expr.replace(/[A-Z]+\d+/gi, (ref) => {
      const pos = parseCellRef(ref);
      if (!pos) return '0';
      const cell = rows[pos.row]?.[pos.col] ?? '0';
      return isNaN(parseFloat(cell)) ? '0' : cell;
    });

    // Evaluate simple arithmetic (only digits, operators, dots, spaces, parens)
    if (/^[\d\s+\-*/().]+$/.test(resolved)) {
      // eslint-disable-next-line no-new-func
      const val = new Function(`"use strict"; return (${resolved})`)() as number;
      return String(val);
    }

    return formula;
  } catch {
    return '#ERR!';
  }
}

// ── Row type for react-data-grid ─────────────────────────────────────────────

type GridRow = Record<string, string>;

// ── Component ────────────────────────────────────────────────────────────────

interface SpreadsheetEditorProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

export function SpreadsheetEditor({ assetPath, projectId, fileName }: SpreadsheetEditorProps) {
  // Raw CSV data (rows × cols); row 0 is the header row
  const [rows, setRows] = useState<string[][]>([]);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);

  // ── Load CSV ──────────────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    async function load() {
      try {
        const base = getApiBase();
        const token = getAuthToken();
        const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
        const res = await fetch(`${base}/api/assets/${encodeURIComponent(assetPath)}${qs}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        const text = await res.text();
        const parsed = Papa.parse<string[]>(text, { skipEmptyLines: true });
        if (!cancelled) {
          // Ensure at least one header row and one data row
          const data = parsed.data as string[][];
          setRows(data.length > 0 ? data : [['A', 'B', 'C'], ['', '', '']]);
          setIsDirty(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load CSV');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void load();
    return () => { cancelled = true; };
  }, [assetPath]);

  // ── Save CSV ──────────────────────────────────────────────────────────────

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    setError(null);
    try {
      const base = getApiBase();
      const token = getAuthToken();
      const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
      const csv = Papa.unparse(rows);
      const res = await fetch(`${base}/api/assets/${encodeURIComponent(assetPath)}${qs}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'text/csv',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: csv,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      setIsDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save CSV');
    } finally {
      setIsSaving(false);
    }
  }, [assetPath, rows]);

  // ── Keyboard shortcut Ctrl/Cmd+S ─────────────────────────────────────────

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        if (isDirty && !isSaving) void handleSave();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isDirty, isSaving, handleSave]);

  // ── Build react-data-grid columns and rows ────────────────────────────────

  const headerRow = rows[0] ?? [];
  const dataRows = rows.slice(1);

  const columns = headerRow.map((header, colIdx) => ({
    key: String(colIdx),
    name: header || colName(colIdx),
    editable: true,
    resizable: true,
    width: 120,
    // Render the evaluated value (formula → computed), edit the raw formula
    renderCell: ({ row }: { row: GridRow }) => {
      const raw = row[String(colIdx)] ?? '';
      const displayed = raw.startsWith('=') ? evalFormula(raw, rows) : raw;
      const isFormula = raw.startsWith('=');
      return (
        <span
          className={
            isFormula
              ? 'text-blue-600 dark:text-blue-400 font-mono text-xs'
              : ''
          }
          title={isFormula ? raw : undefined}
        >
          {displayed}
        </span>
      );
    },
  }));

  // Row number column (frozen, non-editable)
  const rowNumberCol = {
    key: '__rowNum__',
    name: '#',
    frozen: true,
    width: 45,
    minWidth: 45,
    editable: false,
    renderCell: ({ rowIdx }: { rowIdx: number }) => (
      <span className="text-muted-foreground text-xs select-none">{rowIdx + 2}</span>
    ),
  };

  const allColumns = [rowNumberCol, ...columns];

  const gridRows: GridRow[] = dataRows.map((row, _rowIdx) => {
    const obj: GridRow = { __rowNum__: '' };
    headerRow.forEach((_, colIdx) => {
      obj[String(colIdx)] = row[colIdx] ?? '';
    });
    return obj;
  });

  // ── Grid change handler ───────────────────────────────────────────────────

  function handleRowsChange(updatedGridRows: GridRow[]) {
    const newData: string[][] = [
      headerRow,
      ...updatedGridRows.map((r) =>
        headerRow.map((_, i) => r[String(i)] ?? '')
      ),
    ];
    setRows(newData);
    setIsDirty(true);
  }

  // ── Header row editing (direct input in header) ───────────────────────────

  function handleHeaderChange(colIdx: number, value: string) {
    setRows((prev) => {
      const next = prev.map((r) => [...r]);
      if (!next[0]) next[0] = [];
      next[0][colIdx] = value;
      return next;
    });
    setIsDirty(true);
  }

  // ── Toolbar actions ───────────────────────────────────────────────────────

  function addRow() {
    setRows((prev) => {
      const colCount = (prev[0] ?? []).length || 1;
      return [...prev, Array(colCount).fill('')];
    });
    setIsDirty(true);
  }

  function addColumn() {
    setRows((prev) =>
      prev.map((row, i) => {
        const newRow = [...row];
        // For header row, generate next column letter
        newRow.push(i === 0 ? colName(row.length) : '');
        return newRow;
      })
    );
    setIsDirty(true);
  }

  function deleteSelectedRow() {
    if (!selectedCell) return;
    const rowIdx = selectedCell.row; // 0-based in data (excludes header)
    if (rows.length <= 2) {
      // Keep at least header + 1 data row
      setRows((prev) => [prev[0] ?? [], Array((prev[0] ?? []).length).fill('')]);
    } else {
      setRows((prev) => {
        const next = [...prev];
        next.splice(rowIdx + 1, 1); // +1 because row 0 is header
        return next;
      });
    }
    setIsDirty(true);
    setSelectedCell(null);
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div className="flex flex-col h-full items-center justify-center gap-3 text-muted-foreground">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="text-sm">Loading {fileName ?? assetPath}…</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-2 shrink-0 bg-background">
        {/* File name + dirty indicator */}
        <div className="flex items-center gap-2 mr-2 min-w-0">
          <span className="text-sm font-medium truncate">{fileName ?? assetPath}</span>
          {isDirty && (
            <Badge variant="secondary" className="text-xs shrink-0">
              Unsaved
            </Badge>
          )}
        </div>

        <div className="flex-1" />

        {/* Grid info */}
        <span className="text-xs text-muted-foreground mr-2 shrink-0">
          {rows.length > 0 ? rows.length - 1 : 0} rows &times; {headerRow.length} cols
        </span>

        {/* Selected cell info */}
        {selectedCell && (
          <Badge variant="outline" className="text-xs font-mono shrink-0">
            {colName(selectedCell.col)}{selectedCell.row + 2}
          </Badge>
        )}

        <Button variant="outline" size="sm" onClick={addRow} title="Add row">
          <Plus className="h-3.5 w-3.5 mr-1" />
          Row
        </Button>

        <Button variant="outline" size="sm" onClick={addColumn} title="Add column">
          <Plus className="h-3.5 w-3.5 mr-1" />
          Col
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={deleteSelectedRow}
          disabled={!selectedCell}
          title="Delete selected row"
        >
          <Trash2 className="h-3.5 w-3.5 mr-1" />
          Delete Row
        </Button>

        <Button
          size="sm"
          onClick={() => void handleSave()}
          disabled={!isDirty || isSaving}
          title="Save (Ctrl+S)"
        >
          {isSaving ? (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          ) : (
            <Save className="h-3.5 w-3.5 mr-1" />
          )}
          {isSaving ? 'Saving…' : 'Save'}
        </Button>
      </div>

      {/* Error alert */}
      {error && (
        <Alert variant="destructive" className="mx-4 mt-2 shrink-0">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Editable header row */}
      <div
        className="flex shrink-0 border-b border-border bg-muted/50"
        style={{ paddingLeft: 45 /* row number col width */ }}
      >
        {headerRow.map((header, colIdx) => (
          <div
            key={colIdx}
            style={{ width: 120, minWidth: 120 }}
            className="border-r border-border"
          >
            <input
              className="w-full h-full px-2 py-1 text-xs font-semibold bg-transparent outline-none focus:bg-background focus:ring-1 focus:ring-ring"
              value={header}
              onChange={(e) => handleHeaderChange(colIdx, e.target.value)}
              aria-label={`Column ${colName(colIdx)} header`}
            />
          </div>
        ))}
      </div>

      {/* Data grid */}
      <div className="flex-1 overflow-hidden">
        <DataGrid
          className="rdg-light h-full"
          columns={allColumns}
          rows={gridRows}
          onRowsChange={handleRowsChange}
          rowKeyGetter={(row: GridRow) => row.__rowNum__}
          defaultColumnOptions={{ resizable: true }}
          onSelectedCellChange={(args: { row?: GridRow; column: { key: string } }) => {
            if (args.row === undefined) {
              setSelectedCell(null);
              return;
            }
            const rowIdx = gridRows.indexOf(args.row);
            const colKey = args.column.key;
            if (colKey === '__rowNum__') return;
            const colIdx = parseInt(colKey, 10);
            setSelectedCell({ row: rowIdx, col: colIdx });
          }}
        />
      </div>

      {/* Footer */}
      <div className="border-t border-border px-4 py-1.5 text-xs text-muted-foreground shrink-0 flex items-center gap-4">
        <span>
          Tip: <kbd className="px-1.5 py-0.5 bg-muted rounded">Ctrl+S</kbd> to save
        </span>
        <span>Start a cell with <code>=</code> for formulas: <code>=SUM(A2:A10)</code></span>
      </div>
    </div>
  );
}
