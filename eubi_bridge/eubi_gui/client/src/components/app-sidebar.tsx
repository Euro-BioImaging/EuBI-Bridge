import { useState } from "react";
import { useLocation, Link } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  ArrowRightLeft,
  Search,
  Microscope,
  FolderInput,
  FolderOutput,
  FolderOpen,
} from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";
import { ConfigPanel } from "@/components/config-panel";
import { ZarrSidebarBrowser } from "@/components/zarr-sidebar-browser";
import { ConversionSidebarBrowser } from "@/components/conversion-sidebar-browser";

export function AppSidebar() {
  const [location] = useLocation();
  const {
    mode, setMode,
    inputPath, setInputPath,
    selectedInputPaths, setSelectedInputPaths, toggleSelectedInputPath,
    outputPath, setOutputPath,
    includePattern, setIncludePattern,
    excludePattern, setExcludePattern,
  } = useConversionStore();

  type BrowsePanel = "input" | "output" | null;
  const [browsePanel, setBrowsePanel] = useState<BrowsePanel>(null);

  const isConvert = mode === "convert";
  const isInspect = mode === "inspect";

  return (
    <Sidebar>
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary">
            <Microscope className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-tight" data-testid="text-app-title">EuBI-Bridge</h2>
            <p className="text-xs text-muted-foreground">OME-Zarr Toolkit</p>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Mode</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  data-testid="button-mode-convert"
                  data-active={isConvert}
                  asChild
                >
                  <Link href="/" onClick={() => setMode("convert")}>
                    <ArrowRightLeft className="h-4 w-4" />
                    <span>Convert to OME-Zarr</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  data-testid="button-mode-inspect"
                  data-active={isInspect}
                  asChild
                >
                  <Link href="/inspect" onClick={() => setMode("inspect")}>
                    <Search className="h-4 w-4" />
                    <span>Inspect / Visualize</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {isConvert && (
          <>
            {/* ── Input section ──────────────────────────────────────────── */}
            <SidebarGroup className={browsePanel === "input" ? "flex-1 min-h-0 overflow-hidden" : ""}>
              <SidebarGroupLabel>
                <span className="flex items-center gap-1.5 flex-1">
                  <FolderInput className="h-3 w-3" />
                  Input Path
                </span>
                <Button
                  variant="ghost" size="icon"
                  className={`h-5 w-5 ${browsePanel === "input" ? "text-primary" : ""}`}
                  onClick={() => setBrowsePanel(browsePanel === "input" ? null : "input")}
                  data-testid="button-browse-input"
                  title={browsePanel === "input" ? "Close browser" : "Browse"}
                >
                  <FolderOpen className="h-3 w-3" />
                </Button>
              </SidebarGroupLabel>
              <SidebarGroupContent className={browsePanel === "input" ? "h-[calc(100svh-320px)]" : ""}>
                <div className="px-2">
                  {selectedInputPaths.length > 0 ? (
                    <div className="flex items-center gap-1 h-7 px-2 rounded-md bg-primary/10 border border-primary/20 text-xs">
                      <span className="text-primary flex-1 font-medium">
                        {selectedInputPaths.length} file{selectedInputPaths.length !== 1 ? "s" : ""} selected
                      </span>
                      <button
                        type="button"
                        className="text-primary/60 hover:text-primary transition-colors"
                        onClick={() => setSelectedInputPaths([])}
                        title="Clear selection"
                      >✕</button>
                    </div>
                  ) : (
                    <Input
                      data-testid="input-input-path"
                      value={inputPath}
                      onChange={(e) => setInputPath(e.target.value)}
                      placeholder="/path/to/input or directory"
                      className="h-7 text-xs bg-sidebar-accent/50"
                    />
                  )}
                </div>
                {browsePanel === "input" && (
                  <div className="mt-1.5 h-full">
                    <ConversionSidebarBrowser
                      initialPath={inputPath || undefined}
                      onSelect={(p) => { setInputPath(p); setSelectedInputPaths([]); setBrowsePanel(null); }}
                      onSelectMultiple={(paths) => { setSelectedInputPaths(paths); setBrowsePanel(null); }}
                      selectedPaths={selectedInputPaths}
                      onTogglePath={toggleSelectedInputPath}
                      showFilters={true}
                      includePattern={includePattern}
                      excludePattern={excludePattern}
                      onIncludePatternChange={setIncludePattern}
                      onExcludePatternChange={setExcludePattern}
                    />
                  </div>
                )}
              </SidebarGroupContent>
            </SidebarGroup>

            {/* ── Output section ─────────────────────────────────────────── */}
            {browsePanel !== "input" && (
              <SidebarGroup className={browsePanel === "output" ? "flex-1 min-h-0 overflow-hidden" : ""}>
                <SidebarGroupLabel>
                  <span className="flex items-center gap-1.5 flex-1">
                    <FolderOutput className="h-3 w-3" />
                    Output Path
                  </span>
                  <Button
                    variant="ghost" size="icon"
                    className={`h-5 w-5 ${browsePanel === "output" ? "text-primary" : ""}`}
                    onClick={() => setBrowsePanel(browsePanel === "output" ? null : "output")}
                    data-testid="button-browse-output"
                    title={browsePanel === "output" ? "Close browser" : "Browse"}
                  >
                    <FolderOpen className="h-3 w-3" />
                  </Button>
                </SidebarGroupLabel>
                <SidebarGroupContent className={browsePanel === "output" ? "h-[calc(100svh-280px)]" : ""}>
                  <div className="px-2">
                    <Input
                      data-testid="input-output-path"
                      value={outputPath}
                      onChange={(e) => setOutputPath(e.target.value)}
                      placeholder="/path/to/output"
                      className="h-7 text-xs bg-sidebar-accent/50"
                    />
                  </div>
                  {browsePanel === "output" && (
                    <div className="mt-1.5 h-full">
                      <ConversionSidebarBrowser
                        initialPath={outputPath || undefined}
                        onSelect={(p) => { setOutputPath(p); setBrowsePanel(null); }}
                        selectDirOnly={true}
                      />
                    </div>
                  )}
                </SidebarGroupContent>
              </SidebarGroup>
            )}
          </>
        )}

        {isInspect && (
          <SidebarGroup className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <SidebarGroupLabel>Browse Files</SidebarGroupLabel>
            <SidebarGroupContent className="flex-1 min-h-0 overflow-hidden">
              <ZarrSidebarBrowser />
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        {isConvert && browsePanel === null && <ConfigPanel />}
      </SidebarContent>
      <SidebarFooter className="p-4">
        <div className="flex items-center gap-2">
          <img
            src="/eurobioimaging-logo.webp"
            alt="Euro-BioImaging"
            className="h-6 rounded"
            data-testid="img-logo"
          />
          <span className="text-xs text-muted-foreground">Euro-BioImaging</span>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
