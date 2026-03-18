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
import { Label } from "@/components/ui/label";
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
import { FileBrowser } from "@/components/file-browser";
import { ConfigPanel } from "@/components/config-panel";

export function AppSidebar() {
  const [location] = useLocation();
  const {
    mode, setMode,
    inputPath, setInputPath,
    outputPath, setOutputPath,
    includePattern, setIncludePattern,
    excludePattern, setExcludePattern,
    inspectPath, setInspectPath,
  } = useConversionStore();

  const [inputBrowseOpen, setInputBrowseOpen] = useState(false);
  const [outputBrowseOpen, setOutputBrowseOpen] = useState(false);
  const [inspectBrowseOpen, setInspectBrowseOpen] = useState(false);

  const isConvert = mode === "convert" || location === "/";
  const isInspect = mode === "inspect" || location === "/inspect";

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
          <SidebarGroup>
            <SidebarGroupLabel>Input / Output</SidebarGroupLabel>
            <SidebarGroupContent>
              <div className="space-y-3 px-2">
                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center gap-1.5 text-sidebar-foreground/70">
                    <FolderInput className="h-3 w-3" />
                    Input Path
                  </Label>
                  <div className="flex gap-1">
                    <Input
                      data-testid="input-input-path"
                      value={inputPath}
                      onChange={(e) => setInputPath(e.target.value)}
                      placeholder="/path/to/input"
                      className="h-8 text-xs bg-sidebar-accent/50"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0"
                      onClick={() => setInputBrowseOpen(true)}
                      data-testid="button-browse-input"
                    >
                      <FolderOpen className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center gap-1.5 text-sidebar-foreground/70">
                    <FolderOutput className="h-3 w-3" />
                    Output Path
                  </Label>
                  <div className="flex gap-1">
                    <Input
                      data-testid="input-output-path"
                      value={outputPath}
                      onChange={(e) => setOutputPath(e.target.value)}
                      placeholder="/path/to/output"
                      className="h-8 text-xs bg-sidebar-accent/50"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0"
                      onClick={() => setOutputBrowseOpen(true)}
                      data-testid="button-browse-output"
                    >
                      <FolderOpen className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </div>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {isInspect && (
          <SidebarGroup>
            <SidebarGroupLabel>OME-Zarr Path</SidebarGroupLabel>
            <SidebarGroupContent>
              <div className="space-y-1.5 px-2">
                <div className="flex gap-1">
                  <Input
                    data-testid="input-inspect-path"
                    value={inspectPath}
                    onChange={(e) => setInspectPath(e.target.value)}
                    placeholder="/path/to/zarr"
                    className="h-8 text-xs bg-sidebar-accent/50"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={() => setInspectBrowseOpen(true)}
                    data-testid="button-browse-inspect"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        <ConfigPanel />
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

      <FileBrowser
        open={inputBrowseOpen}
        onOpenChange={setInputBrowseOpen}
        onSelect={setInputPath}
        title="Select Input Directory"
        selectMode="directory"
        initialPath={inputPath || undefined}
        showFilters={true}
        includePattern={includePattern}
        excludePattern={excludePattern}
        onIncludePatternChange={setIncludePattern}
        onExcludePatternChange={setExcludePattern}
      />
      <FileBrowser
        open={outputBrowseOpen}
        onOpenChange={setOutputBrowseOpen}
        onSelect={setOutputPath}
        title="Select Output Directory"
        selectMode="directory"
        allowCreate={true}
        initialPath={outputPath || undefined}
      />
      <FileBrowser
        open={inspectBrowseOpen}
        onOpenChange={setInspectBrowseOpen}
        onSelect={setInspectPath}
        title="Select OME-Zarr Directory"
        selectMode="directory"
        initialPath={inspectPath || undefined}
      />
    </Sidebar>
  );
}
