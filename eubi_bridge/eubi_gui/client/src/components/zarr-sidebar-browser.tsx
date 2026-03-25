import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Folder,
  Home,
  ArrowUp,
  Microscope,
  Loader2,
  AlertCircle,
  ChevronRight,
  Eye,
} from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

interface FileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  isOmeZarr?: boolean;
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

export function ZarrSidebarBrowser() {
  const [, navigate] = useLocation();
  const { inspectPath, setInspectPath } = useConversionStore();

  const [homedir, setHomedir] = useState("");
  const [data, setData] = useState<BrowserData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The path input always reflects the current location;
  // while the user is editing we don't overwrite it with navigation updates.
  const [inputValue, setInputValue] = useState("");
  const inputEditingRef = useRef(false);

  // ── load a path (local or S3/HTTP) ────────────────────────────────────────
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
      // Sync input unless user is mid-edit
      if (!inputEditingRef.current) setInputValue(result.currentPath);
    } catch (e: any) {
      setError(e.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  // ── init at the parent directory of the current inspectPath (or home) ─────
  useEffect(() => {
    fetch("/api/homedir")
      .then((r) => r.json())
      .then((d) => {
        setHomedir(d.homedir);
        if (inspectPath && !isRemotePath(inspectPath)) {
          // local path: start in parent dir
          const parent = inspectPath.replace(/[\\/][^\\/]+[\\/]?$/, "") || d.homedir;
          loadPath(parent);
        } else if (inspectPath && isRemotePath(inspectPath)) {
          // remote: start one level up (strip last segment)
          const parent = inspectPath.replace(/\/[^/]+\/?$/, "/");
          loadPath(parent.length > 10 ? parent : inspectPath);
        } else {
          loadPath(d.homedir);
        }
      })
      .catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── open a zarr in the viewer ──────────────────────────────────────────────
  const handleOpen = (itemPath: string) => {
    setInspectPath(itemPath);
    navigate("/inspect");
  };

  // ── path input handlers ────────────────────────────────────────────────────
  const handleInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    inputEditingRef.current = false;
    loadPath(inputValue);
  };

  const atRoot =
    !data?.parentPath ||
    data.parentPath === data.currentPath ||
    // S3 bucket root (no prefix left to go up to)
    (isRemotePath(data.currentPath) && data.parentPath === data.currentPath);

  const dirs = data?.items.filter((i) => i.isDirectory) ?? [];
  const fileCount = data?.items.filter((i) => !i.isDirectory).length ?? 0;

  return (
    <div className="flex flex-col gap-1.5 h-full">
      {/* ── Path input ────────────────────────────────────────────────────── */}
      <form onSubmit={handleInputSubmit} className="px-2">
        <div className="flex gap-1">
          <Input
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              inputEditingRef.current = true;
            }}
            onBlur={() => { inputEditingRef.current = false; }}
            placeholder="/path/to/folder  or  https://s3…"
            className="h-7 text-xs bg-sidebar-accent/50 font-mono min-w-0"
            spellCheck={false}
          />
          <Button
            type="submit"
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            disabled={loading || !inputValue.trim()}
            title="Go"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </form>

      {/* ── Navigation bar ────────────────────────────────────────────────── */}
      <div className="flex items-center gap-0.5 px-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          onClick={() => {
            const target = homedir || "/";
            setInputValue(target);
            loadPath(target);
          }}
          title="Home"
          disabled={loading}
        >
          <Home className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          onClick={() => {
            if (data?.parentPath) {
              setInputValue(data.parentPath);
              loadPath(data.parentPath);
            }
          }}
          disabled={atRoot || loading}
          title="Up"
        >
          <ArrowUp className="h-3 w-3" />
        </Button>
        {/* Current folder name (clickable to reload) */}
        <button
          className="text-[10px] text-muted-foreground truncate flex-1 px-1 text-left hover:text-foreground transition-colors"
          title={data?.currentPath}
          onClick={() => data && loadPath(data.currentPath)}
        >
          {data?.currentPath
            ? data.currentPath.replace(/\\/g, "/").split("/").filter(Boolean).pop() ?? "/"
            : "…"}
        </button>
      </div>

      {/* ── File list ─────────────────────────────────────────────────────── */}
      <ScrollArea className="flex-1 px-1">
        {error ? (
          <div className="flex flex-col items-center py-10 gap-2 text-destructive text-xs">
            <AlertCircle className="h-4 w-4" />
            <span className="text-center px-2">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs"
              onClick={() => loadPath(inputValue || data?.currentPath || homedir)}
            >
              Retry
            </Button>
          </div>
        ) : dirs.length === 0 && !loading ? (
          <div className="flex flex-col items-center py-10 text-xs text-muted-foreground gap-1">
            <Folder className="h-4 w-4 opacity-30" />
            <span>No folders here</span>
            {fileCount > 0 && (
              <span className="text-[10px]">
                {fileCount} file{fileCount !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        ) : (
          <div className="space-y-0.5 pb-2">
            {dirs.map((item) => {
              const isActive = item.path === inspectPath;
              return (
                <div
                  key={item.path}
                  className={`grid px-1.5 py-1 rounded-md text-xs group transition-colors ${
                    item.isOmeZarr ? "grid-cols-[1fr_20px]" : "grid-cols-[1fr_12px]"
                  } ${
                    isActive
                      ? "bg-primary/15 ring-1 ring-primary/25"
                      : "hover:bg-accent/60"
                  }`}
                >
                  <button
                    type="button"
                    className="flex items-start gap-1.5 min-w-0 overflow-hidden text-left"
                    onClick={() => {
                      setInputValue(item.path);
                      loadPath(item.path);
                    }}
                    title={`Browse inside ${item.name}`}
                  >
                    {item.isOmeZarr ? (
                      <Microscope className="h-3.5 w-3.5 text-emerald-500 shrink-0 mt-0.5" />
                    ) : (
                      <Folder className="h-3.5 w-3.5 text-primary/70 shrink-0 mt-0.5" />
                    )}
                    <span
                      className={`break-all leading-snug ${
                        item.isOmeZarr
                          ? "text-emerald-600 dark:text-emerald-400 font-medium"
                          : ""
                      }`}
                    >
                      {item.name}
                    </span>
                  </button>

                  {/* OME-Zarr: open in viewer button — pinned to right column */}
                  {item.isOmeZarr ? (
                    <button
                      type="button"
                      className="flex items-start justify-center pt-0.5 rounded opacity-60 hover:opacity-100 hover:bg-primary/20 transition-opacity"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpen(item.path);
                      }}
                      title={`Open ${item.name} in viewer`}
                    >
                      <Eye className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                    </button>
                  ) : (
                    <ChevronRight className="h-3 w-3 text-muted-foreground/40 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  )}
                </div>
              );
            })}
            {fileCount > 0 && (
              <p className="text-[10px] text-muted-foreground text-center pt-1">
                + {fileCount} file{fileCount !== 1 ? "s" : ""}
              </p>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
