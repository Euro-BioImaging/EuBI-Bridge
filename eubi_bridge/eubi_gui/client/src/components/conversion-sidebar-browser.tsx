import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Folder,
  File,
  Home,
  ArrowUp,
  Loader2,
  AlertCircle,
  ChevronRight,
  Check,
  ListChecks,
  X,
} from "lucide-react";

interface FileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  size?: number;
}

interface BrowserData {
  currentPath: string;
  parentPath: string;
  items: FileEntry[];
}

function isRemotePath(p: string) {
  return /^https?:\/\//i.test(p);
}

interface Props {
  initialPath?: string;
  /** Called when user selects a single file or clicks ✓ on a folder */
  onSelect: (path: string) => void;
  /** Called when user clicks "Use N files" after multi-selecting */
  onSelectMultiple?: (paths: string[]) => void;
  /** Controlled: currently checked file paths (managed by parent via store) */
  selectedPaths?: string[];
  /** Called to toggle a file path in the parent's selected set */
  onTogglePath?: (path: string) => void;
  /** If true, only directories can be selected (output browser mode) */
  selectDirOnly?: boolean;
  showFilters?: boolean;
  includePattern?: string;
  excludePattern?: string;
  onIncludePatternChange?: (v: string) => void;
  onExcludePatternChange?: (v: string) => void;
}

export function ConversionSidebarBrowser({
  initialPath,
  onSelect,
  onSelectMultiple,
  selectedPaths = [],
  onTogglePath,
  selectDirOnly = false,
  showFilters = false,
  includePattern = "",
  excludePattern = "",
  onIncludePatternChange,
  onExcludePatternChange,
}: Props) {
  const [homedir, setHomedir] = useState("");
  const [data, setData] = useState<BrowserData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState("");
  const inputEditingRef = useRef(false);

  const multiSelectEnabled = !!onSelectMultiple && !!onTogglePath && !selectDirOnly;

  const loadPath = async (rawPath: string) => {
    const p = rawPath.trim();
    if (!p) return;
    setLoading(true);
    setError(null);
    try {
      const apiUrl = isRemotePath(p)
        ? `/api/s3/list?url=${encodeURIComponent(p)}`
        : `/api/files?path=${encodeURIComponent(p)}`;
      const res = await fetch(apiUrl);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ message: "Request failed" }));
        throw new Error(err.message || `HTTP ${res.status}`);
      }
      const result: BrowserData = await res.json();
      setData(result);
      if (!inputEditingRef.current) setInputValue(result.currentPath);
    } catch (e: any) {
      setError(e.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch("/api/homedir")
      .then((r) => r.json())
      .then((d) => {
        setHomedir(d.homedir);
        const start = initialPath || d.homedir;
        const startDir = start.replace(/[\\/][^\\/]+[\\/]?$/, "") || d.homedir;
        loadPath(startDir || start);
      })
      .catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    inputEditingRef.current = false;
    loadPath(inputValue);
  };

  const atRoot =
    !data?.parentPath ||
    data.parentPath === data.currentPath ||
    (isRemotePath(data.currentPath ?? "") && data.parentPath === data.currentPath);

  const dirs = data?.items.filter((i) => i.isDirectory) ?? [];
  const files = selectDirOnly ? [] : (data?.items.filter((i) => !i.isDirectory) ?? []);

  const selectedInCurrentDir = files.filter((f) => selectedPaths.includes(f.path));

  return (
    <div className="flex flex-col gap-1.5 h-full">
      {/* ── Multi-select banner ──────────────────────────────────────────── */}
      {multiSelectEnabled && selectedPaths.length > 0 && (
        <div className="mx-2 flex items-center gap-1.5 rounded-md bg-primary/10 border border-primary/20 px-2 py-1">
          <ListChecks className="h-3 w-3 text-primary shrink-0" />
          <span className="text-xs text-primary flex-1 font-medium">
            {selectedPaths.length} file{selectedPaths.length !== 1 ? "s" : ""} selected
          </span>
          <Button
            type="button"
            variant="ghost" size="sm"
            className="h-5 px-1.5 text-xs text-primary hover:bg-primary/20"
            onClick={() => onSelectMultiple!(selectedPaths)}
          >
            Use selection
          </Button>
          <button
            className="p-0.5 rounded hover:bg-primary/20 text-primary/70 hover:text-primary transition-colors"
            onClick={() => {
              // clear all by calling toggle on each; parent should provide a clear method
              // Instead, use onSelectMultiple with empty array as a "clear" signal
              selectedPaths.forEach((p) => onTogglePath!(p));
            }}
            title="Clear selection"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* ── Path input ──────────────────────────────────────────────────── */}
      <form onSubmit={handleInputSubmit} className="px-2">
        <div className="flex gap-1">
          <Input
            value={inputValue}
            onChange={(e) => { setInputValue(e.target.value); inputEditingRef.current = true; }}
            onBlur={() => { inputEditingRef.current = false; }}
            placeholder="/path/to/folder  or  https://s3…"
            className="h-7 text-xs bg-sidebar-accent/50 font-mono min-w-0"
            spellCheck={false}
          />
          <Button type="submit" variant="ghost" size="icon" className="h-7 w-7 shrink-0"
            disabled={loading || !inputValue.trim()} title="Go">
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </Button>
        </div>
      </form>

      {/* ── Navigation bar ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-0.5 px-2">
        <Button type="button" variant="ghost" size="icon" className="h-6 w-6 shrink-0"
          onClick={() => { setInputValue(homedir || "/"); loadPath(homedir || "/"); }}
          title="Home" disabled={loading}>
          <Home className="h-3 w-3" />
        </Button>
        <Button type="button" variant="ghost" size="icon" className="h-6 w-6 shrink-0"
          onClick={() => { if (data?.parentPath) { setInputValue(data.parentPath); loadPath(data.parentPath); } }}
          disabled={atRoot || loading} title="Up">
          <ArrowUp className="h-3 w-3" />
        </Button>
        <button
          className="text-[10px] text-muted-foreground truncate flex-1 px-1 text-left hover:text-foreground transition-colors"
          title={data?.currentPath}
          onClick={() => data && loadPath(data.currentPath)}
        >
          {data?.currentPath
            ? data.currentPath.replace(/\\/g, "/").split("/").filter(Boolean).pop() ?? "/"
            : "…"}
        </button>
        {/* Select current folder */}
        {data && (
          <button
            type="button"
            className="opacity-60 hover:opacity-100 transition-opacity shrink-0 p-0.5 rounded hover:bg-primary/20"
            onClick={() => onSelect(data.currentPath)}
            title={`Use this folder: ${data.currentPath}`}
          >
            <Check className="h-3 w-3 text-primary" />
          </button>
        )}
        {/* Select-all / deselect-all for files in current directory */}
        {multiSelectEnabled && files.length > 0 && (
          <button
            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors shrink-0 px-1"
            onClick={() => {
              const allSelected = files.every((f) => selectedPaths.includes(f.path));
              files.forEach((f) => {
                const isSelected = selectedPaths.includes(f.path);
                if (allSelected && isSelected) onTogglePath!(f.path);
                else if (!allSelected && !isSelected) onTogglePath!(f.path);
              });
            }}
            title={files.every((f) => selectedPaths.includes(f.path)) ? "Deselect all" : "Select all files here"}
          >
            {selectedInCurrentDir.length === files.length && files.length > 0 ? "−all" : "+all"}
          </button>
        )}
      </div>

      {/* ── Include / Exclude filters ────────────────────────────────────── */}
      {showFilters && (
        <div className="px-2 space-y-1">
          <div className="space-y-0.5">
            <Label className="text-[10px] text-muted-foreground">Include pattern</Label>
            <Input value={includePattern} onChange={(e) => onIncludePatternChange?.(e.target.value)}
              placeholder="e.g. *.czi, *.lif"
              className="h-6 text-xs bg-sidebar-accent/50 font-mono" />
          </div>
          <div className="space-y-0.5">
            <Label className="text-[10px] text-muted-foreground">Exclude pattern</Label>
            <Input value={excludePattern} onChange={(e) => onExcludePatternChange?.(e.target.value)}
              placeholder="e.g. *temp*, *.bak"
              className="h-6 text-xs bg-sidebar-accent/50 font-mono" />
          </div>
        </div>
      )}

      {/* ── File list ───────────────────────────────────────────────────── */}
      <ScrollArea className="flex-1 px-1">
        {error ? (
          <div className="flex flex-col items-center py-10 gap-2 text-destructive text-xs">
            <AlertCircle className="h-4 w-4" />
            <span className="text-center px-2">{error}</span>
            <Button type="button" variant="ghost" size="sm" className="h-6 text-xs"
              onClick={() => loadPath(inputValue || data?.currentPath || homedir)}>
              Retry
            </Button>
          </div>
        ) : dirs.length === 0 && files.length === 0 && !loading ? (
          <div className="flex flex-col items-center py-10 text-xs text-muted-foreground gap-1">
            <Folder className="h-4 w-4 opacity-30" />
            <span>Empty folder</span>
          </div>
        ) : (
          <div className="space-y-0.5 pb-2">
            {/* Directories */}
            {dirs.map((item) => (
              <div key={item.path}
                className="flex items-center gap-1 px-1.5 py-1 rounded-md text-xs group hover:bg-accent/60 transition-colors">
                <button
                  className="flex items-center gap-1.5 flex-1 min-w-0 text-left"
                  onClick={() => { setInputValue(item.path); loadPath(item.path); }}
                  title={`Browse ${item.name}`}>
                  <Folder className="h-3.5 w-3.5 text-primary/70 shrink-0" />
                  <span className="truncate leading-tight">{item.name}</span>
                </button>
                {/* Select this folder as input */}
                <button
                  className="opacity-0 group-hover:opacity-70 hover:!opacity-100 transition-opacity shrink-0 p-0.5 rounded hover:bg-primary/20"
                  onClick={(e) => { e.stopPropagation(); onSelect(item.path); }}
                  title={`Use folder: ${item.name}`}>
                  <Check className="h-3 w-3 text-primary" />
                </button>
                <ChevronRight className="h-3 w-3 text-muted-foreground/40 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            ))}

            {/* Files */}
            {files.map((item) => {
              const isChecked = selectedPaths.includes(item.path);
              return (
                <div key={item.path}
                  className={`flex items-center gap-1.5 px-1.5 py-1 rounded-md text-xs transition-colors cursor-pointer
                    ${isChecked ? "bg-primary/10 hover:bg-primary/15" : "hover:bg-accent/60"}`}
                  onClick={() => {
                    if (multiSelectEnabled) {
                      onTogglePath!(item.path);
                    } else {
                      onSelect(item.path);
                    }
                  }}
                  title={multiSelectEnabled ? (isChecked ? `Deselect ${item.name}` : `Select ${item.name}`) : `Select ${item.name}`}
                >
                  {multiSelectEnabled ? (
                    <Checkbox
                      checked={isChecked}
                      onCheckedChange={() => onTogglePath!(item.path)}
                      className="h-3 w-3 shrink-0"
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <File className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                  )}
                  <span className={`truncate leading-tight flex-1 ${isChecked ? "text-primary font-medium" : ""}`}>
                    {item.name}
                  </span>
                  {item.size !== undefined && (
                    <span className="text-[10px] text-muted-foreground shrink-0">
                      {item.size < 1024 ? `${item.size}B`
                        : item.size < 1024 * 1024 ? `${(item.size / 1024).toFixed(0)}K`
                        : `${(item.size / (1024 * 1024)).toFixed(1)}M`}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
