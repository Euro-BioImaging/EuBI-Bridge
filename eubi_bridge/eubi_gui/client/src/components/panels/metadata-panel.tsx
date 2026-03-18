import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { FileText, Ruler } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

export function MetadataPanel() {
  const { metadata, setMetadata } = useConversionStore();

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-metadata-title">
          <FileText className="h-5 w-5 text-primary" />
          Metadata Settings
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure metadata reading and physical scale overrides.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Reading Options</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm">Metadata Reader</Label>
              <Select
                value={metadata.metadataReader}
                onValueChange={(v: any) => setMetadata({ metadataReader: v })}
              >
                <SelectTrigger data-testid="select-metadata-reader">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bfio">bfio (Bio-Formats)</SelectItem>
                  <SelectItem value="bioio">bioio</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm">Channel Intensity Limits</Label>
              <Select
                value={metadata.channelIntensityLimits}
                onValueChange={(v: any) => setMetadata({ channelIntensityLimits: v })}
              >
                <SelectTrigger data-testid="select-intensity-limits">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="from_datatype">From Data Type</SelectItem>
                  <SelectItem value="from_array">From Array Values</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Ruler className="h-4 w-4" />
              Physical Scale Override
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="override-scale" className="text-sm">Override Physical Scale</Label>
              <Switch
                id="override-scale"
                data-testid="switch-override-scale"
                checked={metadata.overridePhysicalScale}
                onCheckedChange={(v) => setMetadata({ overridePhysicalScale: v })}
              />
            </div>

            {metadata.overridePhysicalScale && (
              <>
                <Separator />
                <div className="space-y-4">
                  {([
                    { label: "Time", scaleKey: "scaleTime" as const, unitKey: "unitTime" as const, units: ["second", "millisecond", "microsecond", "nanosecond", "minute", "hour"] },
                    { label: "Z", scaleKey: "scaleZ" as const, unitKey: "unitZ" as const, units: ["micrometer", "nanometer", "millimeter", "centimeter", "meter"] },
                    { label: "Y", scaleKey: "scaleY" as const, unitKey: "unitY" as const, units: ["micrometer", "nanometer", "millimeter", "centimeter", "meter"] },
                    { label: "X", scaleKey: "scaleX" as const, unitKey: "unitX" as const, units: ["micrometer", "nanometer", "millimeter", "centimeter", "meter"] },
                  ] as const).map(({ label, scaleKey, unitKey, units }) => (
                    <div key={label} className="flex items-center gap-3">
                      <Label className="text-sm w-10 shrink-0">{label}</Label>
                      <Input
                        data-testid={`input-scale-meta-${label.toLowerCase()}`}
                        placeholder="Scale"
                        className="w-24"
                        value={metadata[scaleKey]}
                        onChange={(e) => setMetadata({ [scaleKey]: e.target.value })}
                      />
                      <Select
                        value={metadata[unitKey]}
                        onValueChange={(v) => setMetadata({ [unitKey]: v })}
                      >
                        <SelectTrigger data-testid={`select-unit-${label.toLowerCase()}`} className="flex-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {units.map((u) => (
                            <SelectItem key={u} value={u}>{u}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
