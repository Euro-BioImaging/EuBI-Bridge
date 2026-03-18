import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Cpu, HardDrive, MemoryStick } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

export function ClusterPanel() {
  const { cluster, setCluster } = useConversionStore();

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-cluster-title">
          <Cpu className="h-5 w-5 text-primary" />
          Cluster Configuration
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure parallel processing for the conversion pipeline.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Workers & Queues</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="max-workers" className="text-sm">Max Workers</Label>
                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-max-workers-value">{cluster.maxWorkers}</span>
              </div>
              <Slider
                id="max-workers"
                data-testid="slider-max-workers"
                min={1}
                max={64}
                step={1}
                value={[cluster.maxWorkers]}
                onValueChange={([v]) => setCluster({ maxWorkers: v })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="queue-size" className="text-sm">Queue Size</Label>
                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-queue-size-value">{cluster.queueSize}</span>
              </div>
              <Slider
                id="queue-size"
                data-testid="slider-queue-size"
                min={1}
                max={100}
                step={1}
                value={[cluster.queueSize]}
                onValueChange={([v]) => setCluster({ queueSize: v })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="max-concurrency" className="text-sm">Max Concurrency</Label>
                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-max-concurrency-value">{cluster.maxConcurrency}</span>
              </div>
              <Slider
                id="max-concurrency"
                data-testid="slider-max-concurrency"
                min={1}
                max={32}
                step={1}
                value={[cluster.maxConcurrency]}
                onValueChange={([v]) => setCluster({ maxConcurrency: v })}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <HardDrive className="h-4 w-4" />
              Resources
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="region-size" className="text-sm">Region Size (MB)</Label>
                <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-region-size-value">{cluster.regionSizeMb}</span>
              </div>
              <Slider
                id="region-size"
                data-testid="slider-region-size"
                min={1}
                max={1024}
                step={1}
                value={[cluster.regionSizeMb]}
                onValueChange={([v]) => setCluster({ regionSizeMb: v })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="memory-per-worker" className="text-sm flex items-center gap-2">
                <MemoryStick className="h-3.5 w-3.5" />
                Memory per Worker
              </Label>
              <Input
                id="memory-per-worker"
                data-testid="input-memory-per-worker"
                value={cluster.memoryPerWorker}
                onChange={(e) => setCluster({ memoryPerWorker: e.target.value })}
                placeholder="e.g. 4GB"
              />
            </div>

            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="use-local-dask" className="text-sm">Use Local Dask Cluster</Label>
                <Switch
                  id="use-local-dask"
                  data-testid="switch-use-local-dask"
                  checked={cluster.useLocalDask}
                  onCheckedChange={(v) => setCluster({ useLocalDask: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="use-slurm" className="text-sm">Use SLURM Cluster</Label>
                <Switch
                  id="use-slurm"
                  data-testid="switch-use-slurm"
                  checked={cluster.useSlurm}
                  onCheckedChange={(v) => setCluster({ useSlurm: v })}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
