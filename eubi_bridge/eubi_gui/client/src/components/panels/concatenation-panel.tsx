import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Link2 } from "lucide-react";
import { useConversionStore } from "@/lib/conversion-store";

export function ConcatenationPanel() {
  const { concatenation, setConcatenation } = useConversionStore();

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-1">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Link2 className="h-4 w-4" />
            Aggregative Conversion
          </CardTitle>
          <Switch
            data-testid="switch-concatenation"
            checked={concatenation.enabled}
            onCheckedChange={(v) => setConcatenation({ enabled: v })}
          />
        </div>
      </CardHeader>
      {concatenation.enabled && (
        <CardContent className="space-y-4">
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
            />
            <p className="text-xs text-muted-foreground">
              Axes to concatenate along (e.g. "tc" for time and channel).
            </p>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
