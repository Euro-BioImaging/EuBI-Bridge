import { useState } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from "@/components/ui/sidebar";
import {
  ChevronDown,
  ChevronRight,
  Download,
  Upload,
  FolderOpen,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";
import {
  useLoadConfig,
  useResetConfig,
  useApplyConfig,
} from "@/hooks/use-config";
import { FileBrowser } from "@/components/file-browser";

const DEFAULT_PATH = "~/.eubi_bridge/.eubi_config.json";
const CONFIG_DIR = "~/.eubi_bridge";

export function ConfigPanel() {
  const [open, setOpen] = useState(false);
  const [customPath, setCustomPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [status, setStatus] = useState<{ type: "ok" | "err"; msg: string } | null>(null);
  const [loadingDefault, setLoadingDefault] = useState(false);
  const [loadingCustom, setLoadingCustom] = useState(false);

  const store = useConversionStore();
  const applyConfig = useApplyConfig();

  // ── Hooks ──
  const {
    refetch: fetchDefault,
  } = useLoadConfig();

  // ── Reset ──
  const resetConfig = useResetConfig();

  // ── Helpers ──
  function showStatus(type: "ok" | "err", msg: string) {
    setStatus({ type, msg });
    setTimeout(() => setStatus(null), 4000);
  }

  async function handleLoadDefault() {
    setLoadingDefault(true);
    try {
      const result = await fetchDefault();
      if (result.error) {
        showStatus("err", result.error.message);
      } else if (result.data) {
        applyConfig(result.data);
        showStatus("ok", `Loaded from ${result.data._configPath ?? DEFAULT_PATH}`);
      }
    } finally {
      setLoadingDefault(false);
    }
  }

  async function handleLoadCustom() {
    if (!customPath.trim()) return;
    setLoadingCustom(true);
    try {
      const url = `/api/config?path=${encodeURIComponent(customPath)}`;
      const res = await fetch(url);
      if (!res.ok) {
        const body = await res
          .json()
          .catch(() => ({ message: res.statusText }));
        showStatus("err", body.message || "Failed to load config");
        return;
      }
      const data = await res.json();
      if (data) {
        applyConfig(data);
        showStatus("ok", `Loaded from ${data._configPath ?? customPath}`);
      }
    } catch (err) {
      showStatus("err", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoadingCustom(false);
    }
  }

  async function handleSave() {
    const target = customPath.trim() || undefined;
    const payload = {
      cluster: store.cluster,
      reader: store.reader,
      conversion: store.conversion,
      downscaling: store.downscaling,
      metadata: store.metadata,
    };

    try {
      const queryStr = target ? `?path=${encodeURIComponent(target)}` : "";
      const url = `/api/config${queryStr}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res
          .json()
          .catch(() => ({ message: res.statusText }));
        showStatus("err", body.message || "Failed to save config");
        return;
      }
      const data = await res.json();
      showStatus("ok", `Saved to ${data._configPath ?? target ?? DEFAULT_PATH}`);
    } catch (err) {
      showStatus("err", err instanceof Error ? err.message : "Unknown error");
    }
  }

  async function handleReset() {
    const target = customPath.trim() || undefined;
    resetConfig.mutate(
      { configPath: target },
      {
        onSuccess: (data) => {
          applyConfig(data);
          showStatus("ok", `Reset to defaults (${data._configPath ?? DEFAULT_PATH})`);
        },
        onError: (err) => showStatus("err", err.message),
      }
    );
  }

  const anyLoading =
    loadingDefault ||
    loadingCustom ||
    resetConfig.isPending;

  return (
    <>
    <SidebarGroup className="px-0">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger asChild>
          <SidebarGroupLabel className="flex cursor-pointer items-center justify-between w-full pr-2 hover:text-sidebar-foreground transition-colors">
            <span>Config Management</span>
            {open ? (
              <ChevronDown className="h-3.5 w-3.5 shrink-0" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 shrink-0" />
            )}
          </SidebarGroupLabel>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <SidebarGroupContent>
            <div className="space-y-3 px-2 pt-1 pb-2">

              {/* Status banner */}
              {status && (
                <div
                  className={`flex items-center gap-1.5 rounded px-2 py-1 text-xs ${
                    status.type === "ok"
                      ? "bg-green-500/10 text-green-600 dark:text-green-400"
                      : "bg-destructive/10 text-destructive"
                  }`}
                >
                  {status.type === "ok" ? (
                    <CheckCircle className="h-3 w-3 shrink-0" />
                  ) : (
                    <AlertCircle className="h-3 w-3 shrink-0" />
                  )}
                  <span className="truncate">{status.msg}</span>
                </div>
              )}

              {/* Load & Save from default path */}
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">{DEFAULT_PATH}</Label>
                <div className="flex gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 flex-1 gap-1.5 text-xs"
                    onClick={handleLoadDefault}
                    disabled={anyLoading}
                    title="Load defaults from ~/.eubi_bridge/.eubi_config.json"
                  >
                    {loadingDefault ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Download className="h-3 w-3" />
                    )}
                    Load
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 flex-1 gap-1.5 text-xs"
                    onClick={handleSave}
                    disabled={anyLoading || !!customPath.trim()}
                    title="Save current settings to ~/.eubi_bridge/.eubi_config.json"
                  >
                    <Upload className="h-3 w-3" />
                    Save
                  </Button>
                </div>
              </div>

              {/* Custom path */}
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Custom path</Label>
                <div className="flex gap-1">
                  <Input
                    value={customPath}
                    onChange={(e) => setCustomPath(e.target.value)}
                    placeholder="/path/to/.eubi_config.json"
                    className="h-7 text-xs bg-sidebar-accent/50 flex-1 min-w-0"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0"
                    onClick={() => setBrowserOpen(true)}
                    title="Browse for config file"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                  </Button>
                </div>
                <div className="flex gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 flex-1 gap-1.5 text-xs"
                    onClick={handleLoadCustom}
                    disabled={anyLoading || !customPath.trim()}
                    title="Load from custom path"
                  >
                    {loadingCustom ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <FolderOpen className="h-3 w-3" />
                    )}
                    Load
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 flex-1 gap-1.5 text-xs"
                    onClick={handleSave}
                    disabled={anyLoading || !customPath.trim()}
                    title="Save current settings to custom path"
                  >
                    <Upload className="h-3 w-3" />
                    Save
                  </Button>
                </div>
              </div>

              {/* Reset */}
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-full gap-1.5 text-xs hover:bg-destructive/10 hover:text-destructive"
                onClick={handleReset}
                disabled={anyLoading}
                title="Reset all settings to installation defaults"
              >
                {resetConfig.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <RotateCcw className="h-3 w-3" />
                )}
                Reset to defaults
              </Button>

            </div>
          </SidebarGroupContent>
        </CollapsibleContent>
      </Collapsible>
    </SidebarGroup>

    <FileBrowser
      open={browserOpen}
      onOpenChange={setBrowserOpen}
      title="Select Config File"
      selectMode="file"
      initialPath={CONFIG_DIR}
      onSelect={(path) => setCustomPath(path)}
    />
    </>
  );
}
