import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  FolderInput,
  FolderOutput,
  Play,
  Square,
  Terminal,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  FileJson,
  Trash2,
  Settings2,
  Copy,
  Cpu,
  BookOpen,
  Layers,
  FileText,
  Rocket,
  Filter,
  Link2,
  Box,
  Grid3X3,
  Ruler,
  Eye,
} from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { ClusterPanel } from "@/components/panels/cluster-panel";
import { ReaderPanel } from "@/components/panels/reader-panel";
import { ConversionPanel } from "@/components/panels/conversion-panel";
import { DownscalingPanel } from "@/components/panels/downscaling-panel";
import { MetadataPanel } from "@/components/panels/metadata-panel";

function SummaryRow({ icon: Icon, label, value, testId }: { icon: any; label: string; value: string; testId?: string }) {
  return (
    <div className="flex items-start gap-2 text-sm">
      <Icon className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
      <span className="text-muted-foreground shrink-0">{label}:</span>
      <span className="font-mono text-xs truncate" data-testid={testId}>{value}</span>
    </div>
  );
}

export default function ConvertPage() {
  const {
    inputPath,
    selectedInputPaths,
    outputPath,
    includePattern,
    excludePattern,
    concatenation, cluster, reader, conversion, downscaling, metadata,
    logs, addLog, clearLogs,
    isRunning, setIsRunning,
    currentJob, setCurrentJob,
  } = useConversionStore();

  const [showConfig, setShowConfig] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeParamTab, setActiveParamTab] = useState("cluster");
  const logEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // When specific files are selected, pass them as inputPaths list (ignores include/exclude).
  // Otherwise fall back to single inputPath + filters for directory/glob mode.
  const config = selectedInputPaths.length > 0
    ? {
        inputPaths: selectedInputPaths,
        outputPath,
        concatenation, cluster, reader, conversion, downscaling, metadata,
      }
    : {
        inputPath,
        outputPath,
        includePattern,
        excludePattern,
        concatenation, cluster, reader, conversion, downscaling, metadata,
      };

  const handleStartConversion = async () => {
    const hasInput = selectedInputPaths.length > 0 || !!inputPath;
    if (!hasInput) {
      toast({ title: "Missing input", description: "Select files or specify an input path in the sidebar.", variant: "destructive" });
      return;
    }
    if (!outputPath) {
      toast({ title: "Missing output path", description: "Please specify an output directory in the sidebar.", variant: "destructive" });
      return;
    }

    clearLogs();
    setIsRunning(true);
    setProgress(0);

    const timestamp = new Date().toLocaleTimeString("en-US", {
      hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true,
    });

    addLog(`${timestamp} - INFO - Starting conversion...`);
    addLog(`${timestamp} - INFO - Input: ${inputPath}`);
    addLog(`${timestamp} - INFO - Output: ${outputPath}`);
    addLog(`${timestamp} - INFO - Zarr Format: v${conversion.zarrFormat}`);
    addLog(`${timestamp} - INFO - Max Workers: ${cluster.maxWorkers}`);
    if (concatenation.enabled) {
      addLog(`${timestamp} - INFO - Concatenation Axes: ${concatenation.concatenationAxes}`);
    }

    try {
      const response = await apiRequest("POST", "/api/conversion/start", config);
      const job = await response.json();
      setCurrentJob(job);

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        socket.send(JSON.stringify({ type: "subscribe", jobId: job.id }));
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "log") {
          addLog(data.message);
        } else if (data.type === "progress") {
          setProgress(data.progress);
        } else if (data.type === "complete") {
          setIsRunning(false);
          setProgress(100);
          setCurrentJob({ ...job, status: "completed" });
          addLog(`${new Date().toLocaleTimeString()} - INFO - Conversion completed successfully!`);
          toast({ title: "Conversion Complete", description: "Your dataset has been successfully converted to OME-Zarr." });
          socket.close();
        } else if (data.type === "error") {
          setIsRunning(false);
          setCurrentJob({ ...job, status: "failed" });
          addLog(`${new Date().toLocaleTimeString()} - ERROR - ${data.message}`);
          toast({ title: "Conversion Failed", description: data.message, variant: "destructive" });
          socket.close();
        }
      };

      socket.onerror = () => {
        addLog(`${new Date().toLocaleTimeString()} - WARN - WebSocket connection interrupted. Check server logs for status.`);
      };
    } catch (err: any) {
      setIsRunning(false);
      addLog(`${new Date().toLocaleTimeString()} - ERROR - ${err.message || "Failed to start conversion"}`);
      toast({ title: "Error", description: err.message || "Failed to start conversion", variant: "destructive" });
    }
  };

  const handleStopConversion = async () => {
    if (currentJob) {
      try {
        await apiRequest("POST", `/api/conversion/${currentJob.id}/cancel`);
        setIsRunning(false);
        addLog(`${new Date().toLocaleTimeString()} - WARN - Conversion cancelled by user.`);
        toast({ title: "Cancelled", description: "Conversion has been cancelled." });
      } catch {
        toast({ title: "Error", description: "Failed to cancel conversion.", variant: "destructive" });
      }
    }
  };

  const getStatusBadge = () => {
    if (isRunning) return <Badge className="bg-chart-5 text-white"><Loader2 className="h-3 w-3 animate-spin mr-1" />Running</Badge>;
    if (currentJob?.status === "completed") return <Badge className="bg-chart-3 text-white"><CheckCircle2 className="h-3 w-3 mr-1" />Completed</Badge>;
    if (currentJob?.status === "failed") return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Failed</Badge>;
    return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Ready</Badge>;
  };

  const copyConfig = () => {
    navigator.clipboard.writeText(JSON.stringify(config, null, 2));
    toast({ title: "Copied", description: "Configuration copied to clipboard." });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight" data-testid="text-page-title">
          Convert to OME-Zarr
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Configure conversion parameters below, then review and run in the last tab.
        </p>
      </div>

      <Tabs value={activeParamTab} onValueChange={setActiveParamTab}>
        <TabsList className="grid w-full grid-cols-6" data-testid="tabs-parameters">
          <TabsTrigger value="cluster" data-testid="tab-cluster" className="text-xs gap-1.5">
            <Cpu className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Cluster</span>
          </TabsTrigger>
          <TabsTrigger value="reader" data-testid="tab-reader" className="text-xs gap-1.5">
            <BookOpen className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Reader</span>
          </TabsTrigger>
          <TabsTrigger value="conversion" data-testid="tab-conversion" className="text-xs gap-1.5">
            <Settings2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Conversion</span>
          </TabsTrigger>
          <TabsTrigger value="downscaling" data-testid="tab-downscaling" className="text-xs gap-1.5">
            <Layers className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Downscaling</span>
          </TabsTrigger>
          <TabsTrigger value="metadata" data-testid="tab-metadata" className="text-xs gap-1.5">
            <FileText className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Metadata</span>
          </TabsTrigger>
          <TabsTrigger value="run" data-testid="tab-run" className="text-xs gap-1.5">
            <Rocket className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Run</span>
          </TabsTrigger>
        </TabsList>

        <div className="mt-4">
          <TabsContent value="cluster" className="mt-0">
            <ClusterPanel />
          </TabsContent>
          <TabsContent value="reader" className="mt-0">
            <ReaderPanel />
          </TabsContent>
          <TabsContent value="conversion" className="mt-0">
            <ConversionPanel />
          </TabsContent>
          <TabsContent value="downscaling" className="mt-0">
            <DownscalingPanel />
          </TabsContent>
          <TabsContent value="metadata" className="mt-0">
            <MetadataPanel />
          </TabsContent>
          <TabsContent value="run" className="mt-0">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-run-title">
                  <Rocket className="h-5 w-5 text-primary" />
                  Run Conversion
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Review your configuration and start the conversion.
                </p>
              </div>

              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <FolderInput className="h-3.5 w-3.5" />
                      Input / Output
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={FolderInput} label="Input" value={inputPath || "Not set"} testId="text-summary-input" />
                    <SummaryRow icon={FolderOutput} label="Output" value={outputPath || "Not set"} testId="text-summary-output" />
                    {includePattern && <SummaryRow icon={Filter} label="Include" value={includePattern} />}
                    {excludePattern && <SummaryRow icon={Filter} label="Exclude" value={excludePattern} />}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <Cpu className="h-3.5 w-3.5" />
                      Cluster
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={Cpu} label="Workers" value={String(cluster.maxWorkers)} />
                    <SummaryRow icon={Cpu} label="Concurrency" value={String(cluster.maxConcurrency)} />
                    <SummaryRow icon={Cpu} label="Queue" value={String(cluster.queueSize)} />
                    <SummaryRow icon={Cpu} label="Region" value={`${cluster.regionSizeMb} MB`} />
                    {cluster.useLocalDask && <SummaryRow icon={Cpu} label="Dask" value="Local cluster" />}
                    {cluster.useSlurm && <SummaryRow icon={Cpu} label="SLURM" value="Enabled" />}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <BookOpen className="h-3.5 w-3.5" />
                      Reader
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={Eye} label="Scenes" value={reader.readAllScenes ? "All" : reader.sceneIndices || "None"} />
                    <SummaryRow icon={Eye} label="Tiles" value={reader.readAllTiles ? "All" : reader.mosaicTileIndices || "None"} />
                    {reader.readAsMosaic && <SummaryRow icon={Eye} label="Mosaic" value="Enabled" />}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <Settings2 className="h-3.5 w-3.5" />
                      Conversion
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={Settings2} label="Format" value={`Zarr v${conversion.zarrFormat}`} testId="text-summary-format" />
                    <SummaryRow icon={Settings2} label="Data Type" value={conversion.dataType} />
                    <SummaryRow icon={Box} label="Codec" value={conversion.compression.codec} />
                    {conversion.compression.codec === "blosc" && (
                      <SummaryRow icon={Box} label="Blosc" value={`${conversion.compression.bloscInnerCodec}, level ${conversion.compression.level}`} />
                    )}
                    <SummaryRow icon={Grid3X3} label="Chunks" value={conversion.autoChunk ? `auto (${conversion.targetChunkSizeMb} MB)` : "manual"} />
                    {conversion.overwrite && <SummaryRow icon={Settings2} label="Overwrite" value="Yes" />}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5" />
                      Downscaling
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={Layers} label="Layers" value={downscaling.autoDetectLayers ? `auto (min ${downscaling.minDimSize}px)` : `${downscaling.numLayers} layers`} />
                    <SummaryRow icon={Layers} label="Scale Y/X" value={`${downscaling.scaleY}x / ${downscaling.scaleX}x`} />
                    {(downscaling.scaleZ > 1 || downscaling.scaleTime > 1) && (
                      <SummaryRow icon={Layers} label="Scale Z/T" value={`${downscaling.scaleZ}x / ${downscaling.scaleTime}x`} />
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <FileText className="h-3.5 w-3.5" />
                      Metadata
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <SummaryRow icon={FileText} label="Reader" value={metadata.metadataReader} />
                    <SummaryRow icon={FileText} label="Intensity" value={metadata.channelIntensityLimits.replace("_", " ")} />
                    {metadata.overridePhysicalScale && <SummaryRow icon={Ruler} label="Scale" value="Custom overrides" />}
                  </CardContent>
                </Card>

                {concatenation.enabled && (
                  <Card className="md:col-span-2 lg:col-span-3">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                        <Link2 className="h-3.5 w-3.5" />
                        Aggregative Conversion
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <SummaryRow icon={Link2} label="Axes" value={concatenation.concatenationAxes || "Not set"} />
                      {concatenation.timeTag && <SummaryRow icon={Link2} label="Time Tag" value={concatenation.timeTag} />}
                      {concatenation.channelTag && <SummaryRow icon={Link2} label="Channel Tag" value={concatenation.channelTag} />}
                    </CardContent>
                  </Card>
                )}
              </div>

              <Separator />

              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Play className="h-4 w-4" />
                      Controls
                    </CardTitle>
                    {getStatusBadge()}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-3 flex-wrap">
                    {!isRunning ? (
                      <Button
                        data-testid="button-start-conversion"
                        onClick={handleStartConversion}
                        disabled={(!inputPath && selectedInputPaths.length === 0) || !outputPath}
                        size="lg"
                      >
                        <Play className="h-4 w-4 mr-2" />
                        Start Conversion
                      </Button>
                    ) : (
                      <Button
                        data-testid="button-stop-conversion"
                        variant="destructive"
                        onClick={handleStopConversion}
                        size="lg"
                      >
                        <Square className="h-4 w-4 mr-2" />
                        Stop Conversion
                      </Button>
                    )}
                    <Button
                      variant="secondary"
                      data-testid="button-show-config"
                      onClick={() => setShowConfig(!showConfig)}
                    >
                      <FileJson className="h-4 w-4 mr-2" />
                      {showConfig ? "Hide JSON" : "Show JSON"}
                    </Button>
                    <Button
                      variant="secondary"
                      size="icon"
                      data-testid="button-copy-config"
                      onClick={copyConfig}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>

                  {isRunning && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-sm text-muted-foreground">Progress</span>
                        <span className="text-sm font-mono">{Math.round(progress)}%</span>
                      </div>
                      <Progress value={progress} data-testid="progress-conversion" />
                    </div>
                  )}

                  {showConfig && (
                    <ScrollArea className="h-64">
                      <pre className="text-xs font-mono bg-muted/50 p-4 rounded-md whitespace-pre-wrap" data-testid="text-full-config">
                        {JSON.stringify(config, null, 2)}
                      </pre>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between gap-1">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Terminal className="h-4 w-4" />
                      Conversion Log
                      {logs.length > 0 && (
                        <Badge variant="secondary" className="text-[10px]">{logs.length} lines</Badge>
                      )}
                    </CardTitle>
                    {logs.length > 0 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        data-testid="button-clear-logs"
                        onClick={clearLogs}
                      >
                        <Trash2 className="h-3.5 w-3.5 mr-1" />
                        Clear
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-80 rounded-md border bg-sidebar p-4">
                    {logs.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                        <Terminal className="h-8 w-8 mb-2 opacity-30" />
                        <p className="text-sm">No logs yet. Start a conversion to see output here.</p>
                      </div>
                    ) : (
                      <div className="space-y-0.5 font-mono text-xs">
                        {logs.map((log, i) => (
                          <div
                            key={i}
                            data-testid={`text-log-line-${i}`}
                            className={`py-0.5 ${
                              log.includes("ERROR") ? "text-destructive" :
                              log.includes("WARN") ? "text-chart-5" :
                              log.includes("INFO") ? "text-sidebar-foreground" :
                              "text-muted-foreground"
                            }`}
                          >
                            {log}
                          </div>
                        ))}
                        <div ref={logEndRef} />
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
