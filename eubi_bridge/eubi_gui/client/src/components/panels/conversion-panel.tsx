import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Settings2, Box, Grid3X3, Scissors, Link2, ChevronDown } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";
import { CODECS_BY_FORMAT, CODEC_PARAMS } from "@shared/schema";
import { useEffect, useState } from "react";

export function ConversionPanel() {
  const { conversion, setConversion, concatenation, setConcatenation } = useConversionStore();
  const [concatOpen, setConcatOpen] = useState(concatenation.enabled);

  const availableCodecs = CODECS_BY_FORMAT[conversion.zarrFormat] || CODECS_BY_FORMAT[2];
  const codecParams = CODEC_PARAMS[conversion.compression.codec] || CODEC_PARAMS["blosc"];

  useEffect(() => {
    const validCodecValues = availableCodecs.map(c => c.value);
    if (!validCodecValues.includes(conversion.compression.codec)) {
      setConversion({
        compression: { ...conversion.compression, codec: "blosc", level: Math.min(conversion.compression.level, 9) },
      });
    } else {
      const currentCodecParams = CODEC_PARAMS[conversion.compression.codec];
      if (currentCodecParams && conversion.compression.level > currentCodecParams.maxLevel) {
        setConversion({
          compression: { ...conversion.compression, level: currentCodecParams.maxLevel },
        });
      }
    }
  }, [conversion.zarrFormat]);

  useEffect(() => {
    if (codecParams.maxLevel > 0 && conversion.compression.level > codecParams.maxLevel) {
      setConversion({
        compression: { ...conversion.compression, level: codecParams.maxLevel },
      });
    }
  }, [conversion.compression.codec]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-conversion-title">
          <Settings2 className="h-5 w-5 text-primary" />
          Conversion Settings
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure the output format, compression, and chunking parameters.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">General Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-sm">Zarr Format</Label>
                <Select
                  value={String(conversion.zarrFormat)}
                  onValueChange={(v) => setConversion({ zarrFormat: Number(v) })}
                >
                  <SelectTrigger data-testid="select-zarr-format">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="2">Zarr v2</SelectItem>
                    <SelectItem value="3">Zarr v3</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-sm">Data Type</Label>
                <Select
                  value={conversion.dataType}
                  onValueChange={(v: any) => setConversion({ dataType: v })}
                >
                  <SelectTrigger data-testid="select-data-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto</SelectItem>
                    <SelectItem value="uint8">uint8</SelectItem>
                    <SelectItem value="uint16">uint16</SelectItem>
                    <SelectItem value="uint32">uint32</SelectItem>
                    <SelectItem value="float32">float32</SelectItem>
                    <SelectItem value="float64">float64</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Separator />

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="verbose" className="text-sm">Verbose Output</Label>
                <Switch
                  id="verbose"
                  data-testid="switch-verbose"
                  checked={conversion.verbose}
                  onCheckedChange={(v) => setConversion({ verbose: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="overwrite" className="text-sm">Overwrite Existing</Label>
                <Switch
                  id="overwrite"
                  data-testid="switch-overwrite"
                  checked={conversion.overwrite}
                  onCheckedChange={(v) => setConversion({ overwrite: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="squeeze" className="text-sm">Squeeze Dims</Label>
                <Switch
                  id="squeeze"
                  data-testid="switch-squeeze"
                  checked={conversion.squeezeDimensions}
                  onCheckedChange={(v) => setConversion({ squeezeDimensions: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="save-ome-xml" className="text-sm">Save OME-XML</Label>
                <Switch
                  id="save-ome-xml"
                  data-testid="switch-save-ome-xml"
                  checked={conversion.saveOmeXml}
                  onCheckedChange={(v) => setConversion({ saveOmeXml: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="override-channels" className="text-sm">Override Channel Names</Label>
                <Switch
                  id="override-channels"
                  data-testid="switch-override-channels"
                  checked={conversion.overrideChannelNames}
                  onCheckedChange={(v) => setConversion({ overrideChannelNames: v })}
                />
              </div>
              <div className="flex items-center justify-between gap-1">
                <Label htmlFor="skip-dask" className="text-sm">Skip Dask</Label>
                <Switch
                  id="skip-dask"
                  data-testid="switch-skip-dask"
                  checked={conversion.skipDask}
                  onCheckedChange={(v) => setConversion({ skipDask: v })}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Box className="h-4 w-4" />
              Compression
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-sm">Codec</Label>
              <Select
                value={conversion.compression.codec}
                onValueChange={(v: any) =>
                  setConversion({
                    compression: { ...conversion.compression, codec: v },
                  })
                }
              >
                <SelectTrigger data-testid="select-codec">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableCodecs.map((c) => (
                    <SelectItem key={c.value} value={c.value}>{c.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {conversion.zarrFormat === 2 ? "Zarr v2" : "Zarr v3"} supported codecs
              </p>
            </div>

            {codecParams.hasLevel && (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-1">
                  <Label className="text-sm">Compression Level</Label>
                  <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-compression-level">
                    {conversion.compression.level} / {codecParams.maxLevel}
                  </span>
                </div>
                <Slider
                  data-testid="slider-compression-level"
                  min={0}
                  max={codecParams.maxLevel}
                  step={1}
                  value={[conversion.compression.level]}
                  onValueChange={([v]) =>
                    setConversion({
                      compression: { ...conversion.compression, level: v },
                    })
                  }
                />
              </div>
            )}

            {codecParams.hasBlosc && (
              <>
                <Separator />
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Blosc Parameters</p>

                <div className="space-y-2">
                  <Label className="text-sm">Inner Codec</Label>
                  <Select
                    value={conversion.compression.bloscInnerCodec}
                    onValueChange={(v: any) =>
                      setConversion({
                        compression: { ...conversion.compression, bloscInnerCodec: v },
                      })
                    }
                  >
                    <SelectTrigger data-testid="select-blosc-inner-codec">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="lz4">LZ4</SelectItem>
                      <SelectItem value="lz4hc">LZ4HC</SelectItem>
                      <SelectItem value="blosclz">BloscLZ</SelectItem>
                      <SelectItem value="zstd">Zstd</SelectItem>
                      <SelectItem value="snappy">Snappy</SelectItem>
                      <SelectItem value="zlib">Zlib</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Shuffle Mode</Label>
                  <Select
                    value={conversion.compression.bloscShuffle}
                    onValueChange={(v: any) =>
                      setConversion({
                        compression: { ...conversion.compression, bloscShuffle: v },
                      })
                    }
                  >
                    <SelectTrigger data-testid="select-blosc-shuffle">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="noshuffle">No Shuffle</SelectItem>
                      <SelectItem value="shuffle">Byte Shuffle</SelectItem>
                      <SelectItem value="bitshuffle">Bit Shuffle</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Grid3X3 className="h-4 w-4" />
              Chunking
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="auto-chunk" className="text-sm">Auto Chunk</Label>
              <Switch
                id="auto-chunk"
                data-testid="switch-auto-chunk"
                checked={conversion.autoChunk}
                onCheckedChange={(v) => setConversion({ autoChunk: v })}
              />
            </div>

            {conversion.autoChunk && (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-1">
                  <Label className="text-sm">Target Chunk Size (MB)</Label>
                  <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded" data-testid="text-chunk-size-value">{conversion.targetChunkSizeMb}</span>
                </div>
                <Slider
                  data-testid="slider-target-chunk-size"
                  min={1}
                  max={512}
                  step={1}
                  value={[conversion.targetChunkSizeMb]}
                  onValueChange={([v]) => setConversion({ targetChunkSizeMb: v })}
                />
              </div>
            )}

            {!conversion.autoChunk && (
              <div className="grid gap-3 sm:grid-cols-2">
                {(["Time", "Z", "X", "Channel", "Y"] as const).map((dim) => {
                  const key = `chunk${dim}` as keyof typeof conversion;
                  return (
                    <div key={dim} className="space-y-1">
                      <Label className="text-xs">{dim}</Label>
                      <Input
                        data-testid={`input-chunk-${dim.toLowerCase()}`}
                        type="number"
                        min={1}
                        value={conversion[key] as number}
                        onChange={(e) =>
                          setConversion({ [key]: Math.max(1, parseInt(e.target.value) || 1) })
                        }
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {conversion.zarrFormat === 3 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Sharding Coefficients</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2">
                {(["Time", "Z", "X", "Channel", "Y"] as const).map((dim) => {
                  const key = `shard${dim}` as keyof typeof conversion;
                  return (
                    <div key={dim} className="space-y-1">
                      <Label className="text-xs">{dim}</Label>
                      <Input
                        data-testid={`input-shard-${dim.toLowerCase()}`}
                        type="number"
                        min={1}
                        value={conversion[key] as number}
                        onChange={(e) =>
                          setConversion({ [key]: Math.max(1, parseInt(e.target.value) || 1) })
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        <Card className={conversion.zarrFormat === 3 ? "" : "md:col-span-1"}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Scissors className="h-4 w-4" />
              Dimension Ranges
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-3">Optional cropping. Format: start,end</p>
            <div className="grid gap-3 sm:grid-cols-2">
              {(["Time", "Channel", "Z", "Y", "X"] as const).map((dim) => {
                const key = `dimRange${dim}` as keyof typeof conversion;
                return (
                  <div key={dim} className="space-y-1">
                    <Label className="text-xs">{dim}</Label>
                    <Input
                      data-testid={`input-dimrange-${dim.toLowerCase()}`}
                      value={conversion[key] as string}
                      onChange={(e) => setConversion({ [key]: e.target.value })}
                      placeholder="start,end"
                    />
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      <Collapsible open={concatOpen} onOpenChange={setConcatOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="pb-3 cursor-pointer hover:bg-accent/50 transition-colors rounded-t-lg">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Link2 className="h-4 w-4" />
                  Aggregative Conversion
                  {concatenation.enabled && (
                    <Badge variant="default" className="text-[10px] ml-1">Active</Badge>
                  )}
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Switch
                    data-testid="switch-concatenation"
                    checked={concatenation.enabled}
                    onCheckedChange={(v) => {
                      setConcatenation({ enabled: v });
                      if (v) setConcatOpen(true);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${concatOpen ? "rotate-180" : ""}`} />
                </div>
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-4 pt-0">
              <p className="text-xs text-muted-foreground">
                Concatenate multiple source files along specified dimensions into a single OME-Zarr.
              </p>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {([
                  { label: "Time Tag", key: "timeTag" as const, badge: "T" },
                  { label: "Channel Tag", key: "channelTag" as const, badge: "C" },
                  { label: "X Tag", key: "xTag" as const, badge: "X" },
                  { label: "Z Tag", key: "zTag" as const, badge: "Z" },
                  { label: "Y Tag", key: "yTag" as const, badge: "Y" },
                ] as const).map(({ label, key, badge }) => (
                  <div key={key} className="space-y-1.5">
                    <Label className="text-xs flex items-center gap-1.5">
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0">{badge}</Badge>
                      {label}
                    </Label>
                    <Input
                      data-testid={`input-${key}`}
                      value={concatenation[key]}
                      onChange={(e) => setConcatenation({ [key]: e.target.value })}
                      placeholder={`e.g. ${label.split(" ")[0]}`}
                      disabled={!concatenation.enabled}
                    />
                  </div>
                ))}
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Concatenation Axes</Label>
                <Input
                  data-testid="input-concatenation-axes"
                  value={concatenation.concatenationAxes}
                  onChange={(e) => setConcatenation({ concatenationAxes: e.target.value })}
                  placeholder="e.g. tc"
                  disabled={!concatenation.enabled}
                />
                <p className="text-xs text-muted-foreground">
                  Axes to concatenate along (e.g. "tc" for time and channel).
                </p>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
}
