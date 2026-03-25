import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Layers, Sparkles } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

export function DownscalingPanel() {
  const { downscaling, setDownscaling } = useConversionStore();

  const smartOn = downscaling.applySmartDownscaling;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-downscaling-title">
          <Layers className="h-5 w-5 text-primary" />
          Downscaling / Pyramid Settings
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure multi-resolution pyramid generation for efficient viewing.
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Downscaling Method</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Algorithm used to compute each lower-resolution pyramid level.
          </p>
          <Select
            value={downscaling.downscaleMethod}
            onValueChange={(v) => setDownscaling({ downscaleMethod: v as typeof downscaling.downscaleMethod })}
          >
            <SelectTrigger data-testid="select-downscale-method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simple">Simple (stride / nearest)</SelectItem>
              <SelectItem value="mean">Mean</SelectItem>
              <SelectItem value="median">Median</SelectItem>
              <SelectItem value="min">Min</SelectItem>
              <SelectItem value="max">Max</SelectItem>
              <SelectItem value="mode">Mode</SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {/* Smart downscaling toggle */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-amber-500" />
            Smart Downscaling
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between gap-1">
            <div>
              <Label htmlFor="smart-downscale" className="text-sm">Apply Isotropic First Level</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                Auto-computes per-axis scale factors so the first pyramid level achieves
                near-isotropic pixel spacing (≤10% residual anisotropy).
              </p>
            </div>
            <Switch
              id="smart-downscale"
              data-testid="switch-smart-downscaling"
              checked={smartOn}
              onCheckedChange={(v) => setDownscaling({ applySmartDownscaling: v })}
            />
          </div>

          {smartOn && (
            <div className="space-y-3 pt-1 border-t">
              <p className="text-xs text-muted-foreground">
                Scale factors for level 1 are computed automatically from the source pixel sizes.
                Leave blank to auto-compute; enter a value to override.
              </p>
              <div className="space-y-2">
                {([
                  { label: "Time", key: "smartScaleTime" as const },
                  { label: "Z", key: "smartScaleZ" as const },
                  { label: "Y", key: "smartScaleY" as const },
                  { label: "X", key: "smartScaleX" as const },
                ] as const).map(({ label, key }) => (
                  <div key={key} className="flex items-center gap-3">
                    <Label className="text-sm w-16 shrink-0">{label}</Label>
                    <Input
                      data-testid={`input-smart-scale-${label.toLowerCase()}`}
                      type="number"
                      min={1}
                      max={16}
                      placeholder="auto"
                      className="w-24"
                      value={downscaling[key] ?? ""}
                      onChange={(e) => {
                        const raw = e.target.value;
                        setDownscaling({
                          [key]: raw === "" ? null : Math.max(1, parseInt(raw) || 1),
                        });
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Layer Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="auto-detect" className="text-sm">Auto-detect Number of Layers</Label>
              <Switch
                id="auto-detect"
                data-testid="switch-auto-detect-layers"
                checked={downscaling.autoDetectLayers}
                onCheckedChange={(v) => setDownscaling({ autoDetectLayers: v })}
              />
            </div>

            {downscaling.autoDetectLayers ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-1">
                  <Label className="text-sm">Min Dimension Size</Label>
                  <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-min-dim-size">{downscaling.minDimSize}</span>
                </div>
                <Slider
                  data-testid="slider-min-dim-size"
                  min={1}
                  max={512}
                  step={1}
                  value={[downscaling.minDimSize]}
                  onValueChange={([v]) => setDownscaling({ minDimSize: v })}
                />
                <p className="text-xs text-muted-foreground">
                  Layers are generated until all spatial dimensions are smaller than this value.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-1">
                  <Label className="text-sm">Number of Layers</Label>
                  <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-num-layers">{downscaling.numLayers}</span>
                </div>
                <Slider
                  data-testid="slider-num-layers"
                  min={1}
                  max={10}
                  step={1}
                  value={[downscaling.numLayers]}
                  onValueChange={([v]) => setDownscaling({ numLayers: v })}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">
              {smartOn ? "Post-Isotropy Scale Factors (Levels 2+)" : "Scale Factors per Dimension"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-4">
              {smartOn
                ? "Applied from pyramid level 2 onward, after the isotropic first level."
                : "Set the downscaling factor for each dimension between pyramid levels."}
            </p>
            <div className="space-y-4">
              {([
                { label: "Time", key: "scaleTime" as const },
                { label: "Channel", key: "scaleChannel" as const },
                { label: "Z", key: "scaleZ" as const },
                { label: "Y", key: "scaleY" as const },
                { label: "X", key: "scaleX" as const },
              ] as const).map(({ label, key }) => (
                <div key={key} className="flex items-center gap-3">
                  <Label className="text-sm w-16 shrink-0">{label}</Label>
                  <Input
                    data-testid={`input-scale-${label.toLowerCase()}`}
                    type="number"
                    min={1}
                    max={8}
                    className="w-20"
                    value={downscaling[key]}
                    onChange={(e) =>
                      setDownscaling({ [key]: Math.max(1, parseInt(e.target.value) || 1) })
                    }
                  />
                  <div className="flex-1">
                    <Slider
                      min={1}
                      max={8}
                      step={1}
                      value={[downscaling[key]]}
                      onValueChange={([v]) => setDownscaling({ [key]: v })}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
