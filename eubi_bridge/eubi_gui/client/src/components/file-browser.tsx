import { useState, useEffect, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Folder,
  File,
  ChevronRight,
  ChevronLeft,
  ArrowUp,
  Home,
  Loader2,
  AlertCircle,
  FolderOpen,
  FolderPlus,
  Filter,
  FilterX,
  Eye,
} from "lucide-react";

interface FileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  size?: number;
  modified?: string;
}

interface FileBrowserResponse {
  currentPath: string;
  parentPath: string;
  items: FileEntry[];
}

interface FileBrowserProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (path: string) => void;
  title?: string;
  selectMode?: "directory" | "file" | "both";
  initialPath?: string;
  showFilters?: boolean;
  allowCreate?: boolean;
  includePattern?: string;
  excludePattern?: string;
  onIncludePatternChange?: (pattern: string) => void;
  onExcludePatternChange?: (pattern: string) => void;
}


function matchGlobPattern(filename: string, pattern: string): boolean {
  if (!pattern.trim()) return true;
  const patterns = pattern.split(",").map((p) => p.trim()).filter(Boolean);
  return patterns.some((p) => {
    const regex = new RegExp(
      "^" +
        p
          .replace(/[.+^${}()|[\]\\]/g, "\\$&")
          .replace(/\*/g, ".*")
          .replace(/\?/g, ".") +
        "$",
      "i"
    );
    return regex.test(filename);
  });
}

export function FileBrowser({
  open,
  onOpenChange,
  onSelect,
  title = "Browse Files",
  selectMode = "directory",
  initialPath,
  showFilters = false,
  allowCreate = false,
  includePattern = "",
  excludePattern = "",
  onIncludePatternChange,
  onExcludePatternChange,
}: FileBrowserProps) {
  const [homedir, setHomedir] = useState<string>("");
  const [currentPath, setCurrentPath] = useState(initialPath || "");
  const [pathInput, setPathInput] = useState(initialPath || "");
  const [data, setData] = useState<FileBrowserResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [localInclude, setLocalInclude] = useState(includePattern);
  const [localExclude, setLocalExclude] = useState(excludePattern);
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [newFolderError, setNewFolderError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  useEffect(() => {
    setLocalInclude(includePattern);
    setLocalExclude(excludePattern);
  }, [includePattern, excludePattern, open]);

  const loadDirectory = async (dirPath: string) => {
    setLoading(true);
    setError(null);
    setSelectedPath(null);
    try {
      const res = await fetch(`/api/files?path=${encodeURIComponent(dirPath)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ message: "Failed to load directory" }));
        throw new Error(err.message);
      }
      const result: FileBrowserResponse = await res.json();
      setData(result);
      setCurrentPath(result.currentPath);
      setPathInput(result.currentPath);
      setPage(0);
    } catch (err: any) {
      setError(err.message || "Failed to load directory");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch("/api/homedir")
      .then((r) => r.json())
      .then((d) => setHomedir(d.homedir))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (open) {
      const startPath = initialPath && initialPath.length > 0 ? initialPath : (homedir || "/");
      loadDirectory(startPath);
    }
  }, [open]);

  const filteredItems = useMemo(() => {
    if (!data?.items || !showFilters) return data?.items || [];
    return data.items.filter((item) => {
      if (item.isDirectory) return true;
      if (localInclude.trim() && !matchGlobPattern(item.name, localInclude)) return false;
      if (localExclude.trim() && matchGlobPattern(item.name, localExclude)) return false;
      return true;
    });
  }, [data?.items, localInclude, localExclude, showFilters]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pagedItems = filteredItems.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  const fileStats = useMemo(() => {
    if (!data?.items || !showFilters) return null;
    const totalFiles = data.items.filter((i) => !i.isDirectory).length;
    const matchedFiles = filteredItems.filter((i) => !i.isDirectory).length;
    const dirs = data.items.filter((i) => i.isDirectory).length;
    return { totalFiles, matchedFiles, dirs };
  }, [data?.items, filteredItems, showFilters]);

  const handleNavigate = (path: string) => {
    loadDirectory(path);
  };

  const handleGoUp = () => {
    if (data?.parentPath && data.parentPath !== data.currentPath) {
      loadDirectory(data.parentPath);
    }
  };

  const handleGoHome = () => {
    loadDirectory(homedir || "/");
  };

  const handleCreateFolder = async () => {
    const name = newFolderName.trim();
    if (!name) return;
    setNewFolderError(null);
    try {
      const res = await fetch("/api/files/mkdir", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parentPath: currentPath, name }),
      });
      const contentType = res.headers.get("content-type") || "";
      if (!res.ok) {
        const err = contentType.includes("application/json")
          ? await res.json().catch(() => ({}))
          : { message: `Server error ${res.status}` };
        setNewFolderError(err.message || "Failed to create directory");
        return;
      }
      if (!contentType.includes("application/json")) {
        setNewFolderError("Unexpected server response — try restarting the server");
        return;
      }
      const { path: createdPath } = await res.json();
      setNewFolderOpen(false);
      setNewFolderName("");
      await loadDirectory(currentPath);
      setSelectedPath(createdPath);
    } catch (err: any) {
      setNewFolderError(err.message || "Failed to create directory");
    }
  };

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pathInput.trim()) {
      loadDirectory(pathInput.trim());
    }
  };

  const handleItemClick = (item: FileEntry) => {
    if (item.isDirectory) {
      handleNavigate(item.path);
    } else {
      if (selectMode === "file" || selectMode === "both") {
        setSelectedPath(item.path);
      }
    }
  };

  const handleItemSelect = (item: FileEntry) => {
    setSelectedPath(item.path);
  };

  const handleConfirm = () => {
    if (showFilters) {
      onIncludePatternChange?.(localInclude);
      onExcludePatternChange?.(localExclude);
    }
    if (selectMode === "directory") {
      onSelect(selectedPath || currentPath);
    } else if (selectedPath) {
      onSelect(selectedPath);
    }
    onOpenChange(false);
  };

  const handleSelectCurrent = () => {
    if (showFilters) {
      onIncludePatternChange?.(localInclude);
      onExcludePatternChange?.(localExclude);
    }
    onSelect(currentPath);
    onOpenChange(false);
  };

  const pathSegments = currentPath.split("/").filter(Boolean);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2" data-testid="text-browser-title">
            <FolderOpen className="h-5 w-5 text-primary" />
            {title}
          </DialogTitle>
        </DialogHeader>

        <form onSubmit={handlePathSubmit} className="flex gap-2">
          <Input
            data-testid="input-browser-path"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            placeholder="/path/to/directory"
            className="flex-1 font-mono text-sm"
          />
          <Button type="submit" variant="secondary" size="sm" data-testid="button-browser-go">
            Go
          </Button>
        </form>

        {showFilters && (
          <div className="space-y-2 p-3 rounded-md border bg-muted/30">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Eye className="h-3.5 w-3.5" />
              File Filters
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1">
                <Label className="text-xs flex items-center gap-1">
                  <Filter className="h-3 w-3" />
                  Include
                </Label>
                <Input
                  data-testid="input-filter-include"
                  value={localInclude}
                  onChange={(e) => setLocalInclude(e.target.value)}
                  placeholder="e.g. *.tif, *.czi, gradient"
                  className="h-7 text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs flex items-center gap-1">
                  <FilterX className="h-3 w-3" />
                  Exclude
                </Label>
                <Input
                  data-testid="input-filter-exclude"
                  value={localExclude}
                  onChange={(e) => setLocalExclude(e.target.value)}
                  placeholder="e.g. *.tmp, *.log"
                  className="h-7 text-xs"
                />
              </div>
            </div>
            {fileStats && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="secondary" className="text-[10px]">
                  {fileStats.matchedFiles} / {fileStats.totalFiles} files match
                </Badge>
                {fileStats.dirs > 0 && (
                  <span>{fileStats.dirs} folders</span>
                )}
              </div>
            )}
          </div>
        )}

        <div className="flex items-center gap-1 flex-wrap">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={handleGoHome}
            data-testid="button-browser-home"
          >
            <Home className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={handleGoUp}
            disabled={!data?.parentPath || data.parentPath === data.currentPath}
            data-testid="button-browser-up"
          >
            <ArrowUp className="h-3.5 w-3.5" />
          </Button>
          {allowCreate && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 shrink-0"
              onClick={() => { setNewFolderOpen(true); setNewFolderName(""); setNewFolderError(null); }}
              title="New folder"
              data-testid="button-browser-mkdir"
            >
              <FolderPlus className="h-3.5 w-3.5" />
            </Button>
          )}
          <div className="flex items-center gap-0.5 text-sm overflow-x-auto flex-1 min-w-0">
            <button
              className="text-muted-foreground hover:text-foreground px-1 shrink-0"
              onClick={() => handleNavigate("/")}
              data-testid="breadcrumb-root"
            >
              /
            </button>
            {pathSegments.map((segment, i) => {
              const segPath = "/" + pathSegments.slice(0, i + 1).join("/");
              return (
                <span key={segPath} className="flex items-center shrink-0">
                  <ChevronRight className="h-3 w-3 text-muted-foreground" />
                  <button
                    className={`px-1 hover:text-foreground rounded ${
                      i === pathSegments.length - 1
                        ? "text-foreground font-medium"
                        : "text-muted-foreground"
                    }`}
                    onClick={() => handleNavigate(segPath)}
                    data-testid={`breadcrumb-${segment}`}
                  >
                    {segment}
                  </button>
                </span>
              );
            })}
          </div>
        </div>

        <ScrollArea className="flex-1 min-h-[200px] rounded-md border overflow-y-auto">
          {newFolderOpen && (
            <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/40">
              <FolderPlus className="h-4 w-4 text-primary shrink-0" />
              <Input
                autoFocus
                value={newFolderName}
                onChange={(e) => { setNewFolderName(e.target.value); setNewFolderError(null); }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateFolder();
                  if (e.key === "Escape") { setNewFolderOpen(false); setNewFolderName(""); }
                }}
                placeholder="New folder name"
                className="h-7 text-sm flex-1"
              />
              <Button size="sm" className="h-7" onClick={handleCreateFolder} disabled={!newFolderName.trim()}>
                Create
              </Button>
              <Button size="sm" variant="ghost" className="h-7" onClick={() => { setNewFolderOpen(false); setNewFolderName(""); }}>
                Cancel
              </Button>
            </div>
          )}
          {newFolderError && (
            <div className="px-3 py-1 text-xs text-destructive bg-destructive/10 flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />{newFolderError}
            </div>
          )}
          {loading ? (
            <div className="flex items-center justify-center h-full py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full py-12 text-destructive">
              <AlertCircle className="h-6 w-6 mb-2" />
              <p className="text-sm">{error}</p>
              <Button variant="ghost" size="sm" className="mt-2" onClick={() => loadDirectory(currentPath)}>
                Retry
              </Button>
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full py-12 text-muted-foreground">
              <Folder className="h-6 w-6 mb-2 opacity-30" />
              <p className="text-sm">
                {showFilters && data?.items && data.items.length > 0
                  ? "No files match the current filters"
                  : "Empty directory"}
              </p>
            </div>
          ) : (
            <div className="divide-y">
              {pagedItems.map((item) => (
                <button
                  key={item.path}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-accent/50 transition-colors ${
                    selectedPath === item.path ? "bg-primary/10 ring-1 ring-primary/20" : ""
                  }`}
                  onClick={() => handleItemClick(item)}
                  onDoubleClick={() => {
                    if (item.isDirectory) handleNavigate(item.path);
                  }}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    handleItemSelect(item);
                  }}
                  data-testid={`file-item-${item.name}`}
                >
                  {item.isDirectory ? (
                    <Folder className="h-4 w-4 text-primary shrink-0" />
                  ) : (
                    <File className="h-4 w-4 text-muted-foreground shrink-0" />
                  )}
                  <span className="flex-1 text-sm truncate">{item.name}</span>
                  {item.isDirectory && (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                  )}
                </button>
              ))}
            </div>
          )}
        </ScrollArea>

        <div className="flex items-center gap-2 text-xs text-muted-foreground pt-1">
          <Badge variant="secondary" className="text-[10px]">
            {filteredItems.length} items
          </Badge>
          {totalPages > 1 && (
            <div className="flex items-center gap-1 ml-auto">
              <Button type="button" variant="ghost" size="icon" className="h-5 w-5"
                disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>
                <ChevronLeft className="h-3 w-3" />
              </Button>
              <span className="text-[10px]">{safePage + 1}/{totalPages}</span>
              <Button type="button" variant="ghost" size="icon" className="h-5 w-5"
                disabled={safePage >= totalPages - 1} onClick={() => setPage(safePage + 1)}>
                <ChevronRight className="h-3 w-3" />
              </Button>
            </div>
          )}
          {selectedPath && (
            <span className="truncate font-mono">Selected: {selectedPath}</span>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          {selectMode === "directory" && (
            <Button
              variant="outline"
              onClick={handleSelectCurrent}
              data-testid="button-browser-select-current"
            >
              <Folder className="h-4 w-4 mr-2" />
              Select Current Directory
            </Button>
          )}
          <Button
            onClick={handleConfirm}
            disabled={selectMode !== "directory" && !selectedPath}
            data-testid="button-browser-confirm"
          >
            Select
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
