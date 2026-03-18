import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { BookOpen, Eye, Grid3X3, RotateCcw } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

export function ReaderPanel() {
  const { reader, setReader } = useConversionStore();

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2" data-testid="text-reader-title">
          <BookOpen className="h-5 w-5 text-primary" />
          Reader Configuration
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure how input files are read and parsed.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Scene Selection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="read-all-scenes" className="text-sm">Read All Scenes</Label>
              <Switch
                id="read-all-scenes"
                data-testid="switch-read-all-scenes"
                checked={reader.readAllScenes}
                onCheckedChange={(v) => setReader({ readAllScenes: v })}
              />
            </div>
            {!reader.readAllScenes && (
              <div className="space-y-2">
                <Label htmlFor="scene-indices" className="text-sm">Scene Indices</Label>
                <Input
                  id="scene-indices"
                  data-testid="input-scene-indices"
                  value={reader.sceneIndices}
                  onChange={(e) => setReader({ sceneIndices: e.target.value })}
                  placeholder="e.g. 0,1,2 or 0-5"
                />
                <p className="text-xs text-muted-foreground">Comma-separated or range</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Grid3X3 className="h-4 w-4" />
              Tile Selection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="read-all-tiles" className="text-sm">Read All Tiles</Label>
              <Switch
                id="read-all-tiles"
                data-testid="switch-read-all-tiles"
                checked={reader.readAllTiles}
                onCheckedChange={(v) => setReader({ readAllTiles: v })}
              />
            </div>
            {!reader.readAllTiles && (
              <div className="space-y-2">
                <Label htmlFor="mosaic-tile-indices" className="text-sm">Mosaic Tile Indices</Label>
                <Input
                  id="mosaic-tile-indices"
                  data-testid="input-mosaic-tile-indices"
                  value={reader.mosaicTileIndices}
                  onChange={(e) => setReader({ mosaicTileIndices: e.target.value })}
                  placeholder="e.g. 0,1,2 or 0-5"
                />
              </div>
            )}
            <div className="flex items-center justify-between gap-1">
              <Label htmlFor="read-as-mosaic" className="text-sm">Read as Mosaic</Label>
              <Switch
                id="read-as-mosaic"
                data-testid="switch-read-as-mosaic"
                checked={reader.readAsMosaic}
                onCheckedChange={(v) => setReader({ readAsMosaic: v })}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <RotateCcw className="h-4 w-4" />
              Advanced Indices
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
              <div className="space-y-2">
                <Label htmlFor="view-index" className="text-sm">View Index</Label>
                <Input
                  id="view-index"
                  data-testid="input-view-index"
                  value={reader.viewIndex}
                  onChange={(e) => setReader({ viewIndex: e.target.value })}
                  placeholder="Optional"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="phase-index" className="text-sm">Phase Index</Label>
                <Input
                  id="phase-index"
                  data-testid="input-phase-index"
                  value={reader.phaseIndex}
                  onChange={(e) => setReader({ phaseIndex: e.target.value })}
                  placeholder="Optional"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="illumination-index" className="text-sm">Illumination Index</Label>
                <Input
                  id="illumination-index"
                  data-testid="input-illumination-index"
                  value={reader.illuminationIndex}
                  onChange={(e) => setReader({ illuminationIndex: e.target.value })}
                  placeholder="Optional"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rotation-index" className="text-sm">Rotation Index</Label>
                <Input
                  id="rotation-index"
                  data-testid="input-rotation-index"
                  value={reader.rotationIndex}
                  onChange={(e) => setReader({ rotationIndex: e.target.value })}
                  placeholder="Optional"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sample-index" className="text-sm">Sample Index</Label>
                <Input
                  id="sample-index"
                  data-testid="input-sample-index"
                  value={reader.sampleIndex}
                  onChange={(e) => setReader({ sampleIndex: e.target.value })}
                  placeholder="Optional"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
