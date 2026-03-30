import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Search,
  FolderOpen,
  Layers,
  Palette,
  Info,
  Eye,
  EyeOff,
  Save,
  RotateCcw,
  Database,
  Ruler,
  FileText,
  Grid3X3,
  Microscope,
  ZoomIn,
  Loader2,
  Move,
  Maximize2,
  AlertTriangle,
  ChevronDown,
  BarChart2,
} from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useConversionStore } from "@/lib/conversion-store";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import type { ZarrMetadata } from "@shared/schema";

const FOV_SIZE_OPTIONS = [128, 256, 512, 1024];

function formatPixelSize(v: number): string {
  if (v === 0) return "0";
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(2);
  return parseFloat(v.toPrecision(4)).toString();
}

interface ViewerChannel {
  index: number;
  label: string;
  color: string;
  visible: boolean;
  intensityMin: number;
  intensityMax: number;
  window: { min: number; max: number; start: number; end: number };
}

interface ZarrInfo {
  axes: string;
  shape: number[];
  resolutionPaths: string[];
  levelsInfo: Array<{ path: string; shape: number[]; dtype: string }>;
  numLevels: number;
  channels: Array<{
    index: number;
    label: string;
    color: string;
    visible: boolean;
    window: { min: number; max: number; start: number; end: number };
  }>;
  dimSizes: Record<string, number>;
  dtype: string;
}

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}


export default function InspectPage() {
  const { inspectPath, setInspectPath } = useConversionStore();
  const [activeTab, setActiveTab] = useState("metadata");
  const { toast } = useToast();

  const [zoomLevel, setZoomLevel] = useState(0);
  const [timeSlice, setTimeSlice] = useState(0);
  const [zSlice, setZSlice] = useState(0);
  const [orientation, setOrientation] = useState<"XY" | "XZ" | "YZ">("XY");
  const [fovSize, setFovSize] = useState(512);
  const [fovCenterY, setFovCenterY] = useState(0);
  const [fovCenterX, setFovCenterX] = useState(0);
  const [viewerChannels, setViewerChannels] = useState<ViewerChannel[]>([]);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Client-side decoded-bitmap cache.  Keyed by full plane URL; value is a
  // GPU-ready ImageBitmap.  Insertion order = eviction order (oldest first).
  // Drawing from this cache costs < 1 ms vs 15–30 ms for createImageBitmap.
  const bitmapCacheRef = useRef<Map<string, ImageBitmap>>(new Map());
  // Computed from system RAM: 25 % of total RAM, capped at 1.5 GB, expressed
  // as a frame count for the current fovSize.  Starts at a safe default (60)
  // and is refined once the /api/system/memory response arrives.
  const bitmapCacheMaxRef = useRef(60);
  // AbortControllers for background prefetch requests (cancelled on each new main fetch)
  const prefetchAbortRef = useRef<Set<AbortController>>(new Set());
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startX: number; startY: number; startCenterX: number; startCenterY: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [histAdjLoading, setHistAdjLoading] = useState<Record<number, boolean>>({});
  const [editableScales, setEditableScales] = useState<{ axis: string; value: number; unit: string }[]>([]);
  const [openLayers, setOpenLayers] = useState<Set<number>>(new Set([0]));
  const [openChannels, setOpenChannels] = useState<Set<number>>(new Set());
  const planeServerUrlRef = useRef<string>("");

  // Discover the plane server's direct port once on mount for bypass-proxy tile fetches
  useEffect(() => {
    fetch("/api/zarr/port")
      .then((r) => r.json())
      .then((d) => { if (d.url) planeServerUrlRef.current = d.url; })
      .catch(() => {});
  }, []);

  // Compute bitmap cache capacity from system RAM and current fovSize.
  // Budget: 25 % of total RAM, capped at 1.5 GB.
  // Each ImageBitmap occupies fovSize × fovSize × 4 bytes (RGBA).
  useEffect(() => {
    fetch("/api/system/memory")
      .then((r) => r.json())
      .then(({ totalMb }: { totalMb: number }) => {
        const budgetBytes = Math.min(totalMb * 1024 * 1024 * 0.25, 1.5 * 1024 ** 3);
        const bytesPerBitmap = fovSize * fovSize * 4;
        const computed = Math.floor(budgetBytes / bytesPerBitmap);
        bitmapCacheMaxRef.current = Math.max(30, Math.min(computed, 1000));
      })
      .catch(() => {});
  }, [fovSize]);

  // Clear the bitmap cache when the zarr path changes (different dataset = stale bitmaps)
  useEffect(() => {
    for (const bm of bitmapCacheRef.current.values()) bm.close();
    bitmapCacheRef.current.clear();
    for (const ctrl of prefetchAbortRef.current) ctrl.abort();
    prefetchAbortRef.current.clear();
  }, [inspectPath]);

  const debouncedZoom = useDebounce(zoomLevel, 100);
  const debouncedT = useDebounce(timeSlice, 16);
  const debouncedZ = useDebounce(zSlice, 16);
  const debouncedFovCenterY = useDebounce(fovCenterY, 100);
  const debouncedFovCenterX = useDebounce(fovCenterX, 100);
  const debouncedChannels = useDebounce(viewerChannels, 150);

  const { data: zarrMetadata, isLoading, error } = useQuery<ZarrMetadata>({
    queryKey: ["/api/zarr/metadata", inspectPath],
    queryFn: async () => {
      const res = await fetch(`/api/zarr/metadata?path=${encodeURIComponent(inspectPath)}`);
      if (!res.ok) throw new Error("Failed to load metadata");
      return res.json();
    },
    enabled: !!inspectPath && inspectPath.length > 2,
  });

  useEffect(() => {
    if (!zarrMetadata) return;
    setEditableScales(
      zarrMetadata.axes.map((ax: any) => {
        const existing = zarrMetadata.scales.find((s: any) => s.axis === ax.name);
        return { axis: ax.name, value: existing?.value ?? 1, unit: existing?.unit ?? ax.unit ?? "" };
      })
    );
  }, [zarrMetadata]);

  const { data: zarrInfo } = useQuery<ZarrInfo>({
    queryKey: ["/api/zarr/info", inspectPath],
    queryFn: async () => {
      const res = await fetch(`/api/zarr/info?path=${encodeURIComponent(inspectPath)}`);
      if (!res.ok) throw new Error("Failed to load zarr info");
      return res.json();
    },
    enabled: !!inspectPath && inspectPath.length > 2,
  });

  useEffect(() => {
    if (zarrInfo) {
      const newChannels: ViewerChannel[] = zarrInfo.channels.map((ch) => ({
        index: ch.index,
        label: ch.label,
        color: `#${ch.color.replace('#', '')}`,
        visible: ch.visible,
        intensityMin: ch.window.start,
        intensityMax: ch.window.end,
        window: ch.window,
      }));
      setViewerChannels(newChannels);

      const axes = zarrInfo.axes;
      const shape = zarrInfo.shape;
      const yIdx = axes.indexOf('y');
      const xIdx = axes.indexOf('x');
      const centerY = yIdx >= 0 ? Math.floor(shape[yIdx] / 2) : 0;
      const centerX = xIdx >= 0 ? Math.floor(shape[xIdx] / 2) : 0;
      setFovCenterY(centerY);
      setFovCenterX(centerX);

      const numLevels = zarrInfo.numLevels;
      let bestZoom = 0;
      for (let z = numLevels - 1; z >= 0; z--) {
        const invertedIdx = numLevels - 1 - z;
        if (invertedIdx < zarrInfo.levelsInfo.length) {
          const levelShape = zarrInfo.levelsInfo[invertedIdx].shape;
          const h = levelShape[levelShape.length - 2] || 0;
          const w = levelShape[levelShape.length - 1] || 0;
          if (h <= fovSize && w <= fovSize) {
            bestZoom = z;
            break;
          }
        }
      }
      setZoomLevel(bestZoom);
      setTimeSlice(0);
      setZSlice(0);
    }
  }, [zarrInfo]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number } | null>(null);

  const viewParams = useMemo(() => {
    if (!inspectPath || !zarrInfo || debouncedChannels.length === 0) return null;
    const base: Record<string, string> = {
      path: inspectPath,
      level: String(debouncedZoom),
      orientation,
      fovSize: String(fovSize),
      fovCenterY: String(debouncedFovCenterY),
      fovCenterX: String(debouncedFovCenterX),
    };
    if (zarrInfo.dimSizes.t !== undefined) base.t = String(debouncedT);
    if (zarrInfo.dimSizes.z !== undefined && orientation === "XY") base.z = String(debouncedZ);
    if (orientation === "XZ" && zarrInfo.dimSizes.y !== undefined) base.sliceIdx = String(debouncedZ);
    if (orientation === "YZ" && zarrInfo.dimSizes.x !== undefined) base.sliceIdx = String(debouncedZ);

    const channelsParam = debouncedChannels.map((ch) => ({
      index: ch.index, color: ch.color, visible: ch.visible,
      intensityMin: ch.intensityMin, intensityMax: ch.intensityMax,
    }));
    base.channels = JSON.stringify(channelsParam);
    return base;
  }, [inspectPath, zarrInfo, debouncedZoom, orientation, fovSize, debouncedFovCenterY, debouncedFovCenterX, debouncedT, debouncedZ, debouncedChannels]);

  const loadingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!viewParams || isDragging || activeTab !== "viewer") return;

    const directBase = planeServerUrlRef.current;
    const planeParams = new URLSearchParams(viewParams);
    const planeUrl = directBase
      ? `${directBase}/plane?${planeParams.toString()}`
      : `/api/zarr/plane?${planeParams.toString()}`;

    // ── Helper: add a decoded bitmap to the LRU cache ──────────────────────
    const storeBitmap = (url: string, bm: ImageBitmap) => {
      const cache = bitmapCacheRef.current;
      if (cache.has(url)) return; // already present
      cache.set(url, bm);
      while (cache.size > bitmapCacheMaxRef.current) {
        const oldest = cache.keys().next().value!;
        cache.get(oldest)!.close();
        cache.delete(oldest);
      }
    };

    // ── Helper: draw a bitmap to the canvas ────────────────────────────────
    const drawBitmap = (bm: ImageBitmap) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (canvas.width !== bm.width || canvas.height !== bm.height) {
        canvas.width = bm.width;
        canvas.height = bm.height;
      }
      // Always call setCanvasSize so display:block is applied even when dimensions are unchanged
      setCanvasSize(prev =>
        prev?.width === bm.width && prev?.height === bm.height
          ? prev
          : { width: bm.width, height: bm.height }
      );
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(bm, 0, 0);
    };

    // ── Helper: background prefetch of neighbouring z-slices ───────────────
    const schedulePrefetch = () => {
      // Cancel previous prefetch batch — they may be for a different position
      for (const ctrl of prefetchAbortRef.current) ctrl.abort();
      prefetchAbortRef.current.clear();

      // Determine which param carries the slice index (z or sliceIdx)
      const zKey = viewParams.z !== undefined ? "z"
        : viewParams.sliceIdx !== undefined ? "sliceIdx"
        : null;
      if (!zKey) return;

      const currentZ = parseInt(viewParams[zKey], 10);
      // Use t-axis size as a proxy when z is named 'sliceIdx' (XZ/YZ view)
      const zMax = zarrInfo
        ? (zKey === "z" ? (zarrInfo.dimSizes.z ?? 1)
          : zKey === "sliceIdx" && viewParams.sliceIdx !== undefined
            ? (zarrInfo.dimSizes.y ?? zarrInfo.dimSizes.x ?? 1)
            : 1)
        : 1;

      // Prefetch ±1, ±2, ±3 slices; skip if already in bitmap cache
      for (const delta of [-1, 1, -2, 2, -3, 3]) {
        const nz = currentZ + delta;
        if (nz < 0 || nz >= zMax) continue;
        const neighborParams = new URLSearchParams({ ...viewParams, [zKey]: String(nz) });
        const neighborUrl = directBase
          ? `${directBase}/plane?${neighborParams.toString()}`
          : `/api/zarr/plane?${neighborParams.toString()}`;
        if (bitmapCacheRef.current.has(neighborUrl)) continue;

        const ctrl = new AbortController();
        prefetchAbortRef.current.add(ctrl);
        fetch(neighborUrl, { signal: ctrl.signal })
          .then((r) => (r.ok ? r.blob() : Promise.reject()))
          .then((blob) => createImageBitmap(blob))
          .then((bm) => {
            if (ctrl.signal.aborted) { bm.close(); return; }
            storeBitmap(neighborUrl, bm);
          })
          .catch(() => {})
          .finally(() => prefetchAbortRef.current.delete(ctrl));
      }
    };

    // ── Fast path: bitmap already decoded in local cache ───────────────────
    const cachedBitmap = bitmapCacheRef.current.get(planeUrl);
    if (cachedBitmap) {
      if (loadingTimerRef.current) { clearTimeout(loadingTimerRef.current); loadingTimerRef.current = null; }
      drawBitmap(cachedBitmap);
      setImageLoading(false);
      setImageError(null);
      schedulePrefetch();
      return;
    }

    // ── Slow path: fetch from plane server ─────────────────────────────────
    // Show spinner only after 300 ms so fast responses don't blank the canvas
    if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
    loadingTimerRef.current = setTimeout(() => setImageLoading(true), 300);

    setImageError(null);
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const t0 = performance.now();
    console.log(`[Plane] fetch start  via=${directBase ? "direct" : "proxy"}`);

    fetch(planeUrl, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const xCache = res.headers.get("X-Cache") ?? (res.headers.get("ETag") ? "hit?" : "miss");
        console.log(`[Plane] headers      ${(performance.now()-t0).toFixed(0)}ms  cache=${xCache}  size=${res.headers.get("Content-Length") ?? "?"}B`);
        return res.blob();
      })
      .then((blob) => {
        console.log(`[Plane] blob ready   ${(performance.now()-t0).toFixed(0)}ms  blobSize=${blob.size}B`);
        return createImageBitmap(blob);
      })
      .then((bitmap) => {
        if (controller.signal.aborted) { bitmap.close(); return; }
        if (!canvasRef.current) { bitmap.close(); return; }
        console.log(`[Plane] bitmap ready ${(performance.now()-t0).toFixed(0)}ms`);
        storeBitmap(planeUrl, bitmap);
        drawBitmap(bitmap);
        console.log(`[Plane] drawn        ${(performance.now()-t0).toFixed(0)}ms  total`);
        if (loadingTimerRef.current) { clearTimeout(loadingTimerRef.current); loadingTimerRef.current = null; }
        setImageLoading(false);
        schedulePrefetch();
      })
      .catch((err) => {
        if (loadingTimerRef.current) { clearTimeout(loadingTimerRef.current); loadingTimerRef.current = null; }
        if (err.name === "AbortError" || controller.signal.aborted) return;
        setImageError(err.message || "Failed to load plane");
        setImageLoading(false);
      });

    return () => {
      controller.abort();
      if (loadingTimerRef.current) { clearTimeout(loadingTimerRef.current); loadingTimerRef.current = null; }
    };
  }, [viewParams, isDragging, zarrInfo, activeTab]);

  const updateMetadataMutation = useMutation({
    mutationFn: async (data: { path: string; scales: { axis: string; value: number; unit: string }[] }) => {
      const res = await apiRequest("POST", "/api/zarr/metadata/update", data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/zarr/metadata", inspectPath] });
      toast({ title: "Updated", description: "Metadata updated successfully." });
    },
  });

  const updateChannelMetadataMutation = useMutation({
    mutationFn: async (data: { path: string; channels: Array<{ index: number; label: string; color: string; window: { start: number; end: number; min: number; max: number } }> }) => {
      const res = await apiRequest("POST", "/api/zarr/channels/update", data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/zarr/metadata", inspectPath] });
      queryClient.invalidateQueries({ queryKey: ["/api/zarr/info", inspectPath] });
      toast({ title: "Saved", description: "Channel metadata saved to OME-Zarr." });
    },
    onError: (err: Error) => {
      toast({ title: "Error", description: err.message || "Failed to update channel metadata.", variant: "destructive" });
    },
  });

  const handleLoadZarr = () => {
    if (!inspectPath) {
      toast({ title: "Missing path", description: "Please enter a path to an OME-Zarr dataset.", variant: "destructive" });
      return;
    }
    queryClient.invalidateQueries({ queryKey: ["/api/zarr/metadata", inspectPath] });
    queryClient.invalidateQueries({ queryKey: ["/api/zarr/info", inspectPath] });
  };

  const updateViewerChannel = useCallback((index: number, updates: Partial<ViewerChannel>) => {
    setViewerChannels((prev) =>
      prev.map((ch) => (ch.index === index ? { ...ch, ...updates } : ch))
    );
  }, []);

  const resetHistogram = useCallback((channelIndex: number) => {
    const originalCh = zarrInfo?.channels?.find((c: any) => c.index === channelIndex);
    if (!originalCh) return;
    setViewerChannels((prev) =>
      prev.map((ch) =>
        ch.index === channelIndex
          ? {
              ...ch,
              window: {
                min: originalCh.window?.min ?? 0,
                max: originalCh.window?.max ?? 255,
                start: originalCh.window?.start ?? 0,
                end: originalCh.window?.end ?? 255,
              },
              intensityMin: originalCh.window?.start ?? 0,
              intensityMax: originalCh.window?.end ?? 255,
            }
          : ch
      )
    );
  }, [zarrInfo]);

  const adjustHistogram = useCallback((channelIndex: number) => {
    if (!inspectPath) return;
    setHistAdjLoading((prev) => ({ ...prev, [channelIndex]: true }));
    fetch(`/api/zarr/channel_minmax?path=${encodeURIComponent(inspectPath)}&channel=${channelIndex}`, { cache: 'no-store' })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { channel: number; min: number; max: number }) => {
        setViewerChannels((prev) =>
          prev.map((ch) =>
            ch.index === channelIndex
              ? {
                  ...ch,
                  window: { ...ch.window, min: data.min, max: data.max },
                  intensityMin: data.min,
                  intensityMax: data.max,
                }
              : ch
          )
        );
      })
      .catch((err) => {
        console.error(`Failed to adjust histogram for channel ${channelIndex}:`, err);
      })
      .finally(() => {
        setHistAdjLoading((prev) => ({ ...prev, [channelIndex]: false }));
      });
  }, [inspectPath]);

  const handleFitToView = useCallback(() => {
    if (!zarrInfo) return;
    const axes = zarrInfo.axes;
    const shape = zarrInfo.shape;
    const yIdx = axes.indexOf('y');
    const xIdx = axes.indexOf('x');
    setFovCenterY(yIdx >= 0 ? Math.floor(shape[yIdx] / 2) : 0);
    setFovCenterX(xIdx >= 0 ? Math.floor(shape[xIdx] / 2) : 0);
    setZoomLevel(0);
  }, [zarrInfo]);

  const dimSizes = zarrInfo?.dimSizes || {};
  const hasT = dimSizes.t !== undefined && dimSizes.t > 1;
  const hasZ = dimSizes.z !== undefined && dimSizes.z > 1;
  const hasC = dimSizes.c !== undefined && dimSizes.c > 1;
  const maxY = dimSizes.y || zarrInfo?.shape?.[zarrInfo.shape.length - 2] || 128;
  const maxX = dimSizes.x || zarrInfo?.shape?.[zarrInfo.shape.length - 1] || 128;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startCenterX: fovCenterX,
      startCenterY: fovCenterY,
    };
    setIsDragging(true);
  }, [fovCenterX, fovCenterY]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      setDragOffset({ x: dx, y: dy });
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (dragRef.current && canvasRef.current) {
        const cvs = canvasRef.current;
        const displayW = cvs.clientWidth;
        const displayH = cvs.clientHeight;
        const naturalW = cvs.width;
        const naturalH = cvs.height;
        if (displayW > 0 && displayH > 0) {
          const scaleX = naturalW / displayW;
          const scaleY = naturalH / displayH;
          const dx = e.clientX - dragRef.current.startX;
          const dy = e.clientY - dragRef.current.startY;
          const newCenterX = Math.round(dragRef.current.startCenterX - dx * scaleX);
          const newCenterY = Math.round(dragRef.current.startCenterY - dy * scaleY);
          setFovCenterX(Math.max(0, Math.min(newCenterX, maxX - 1)));
          setFovCenterY(Math.max(0, Math.min(newCenterY, maxY - 1)));
        }
      }
      dragRef.current = null;
      setDragOffset({ x: 0, y: 0 });
      setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, maxX, maxY]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!zarrInfo) return;
    const zMax = (dimSizes.z || 1) - 1;
    if (zMax <= 0) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    setZSlice((prev) => Math.max(0, Math.min(prev + delta, zMax)));
  }, [zarrInfo, dimSizes.z]);

  const currentLevelShape = useMemo(() => {
    if (!zarrInfo) return null;
    const numLevels = zarrInfo.numLevels;
    const invertedIdx = Math.max(0, Math.min(numLevels - 1 - zoomLevel, numLevels - 1));
    const levelInfo = zarrInfo.levelsInfo?.[invertedIdx];
    if (!levelInfo) return null;
    const shape = levelInfo.shape;
    return {
      height: shape[shape.length - 2] || 0,
      width: shape[shape.length - 1] || 0,
    };
  }, [zarrInfo, zoomLevel]);

  const fovWarning = useMemo(() => {
    if (!currentLevelShape) return null;
    const { height, width } = currentLevelShape;
    if (fovSize > height && fovSize > width) {
      return `FOV (${fovSize}×${fovSize}) exceeds image dimensions (${width}×${height}) at this zoom level. Showing full image.`;
    }
    if (fovSize > width) {
      return `FOV width (${fovSize}) exceeds image width (${width}) at this zoom level. Width clamped to ${width}.`;
    }
    if (fovSize > height) {
      return `FOV height (${fovSize}) exceeds image height (${height}) at this zoom level. Height clamped to ${height}.`;
    }
    return null;
  }, [currentLevelShape, fovSize]);

  // Physical aspect ratio (height/width) for the current view plane and pyramid level.
  // Derived from the coordinateTransformations scale stored in each pyramid level.
  // Orientation XY → row=y / col=x; XZ → row=z / col=x; YZ → row=z / col=y.
  const pixelAspectRatio = useMemo(() => {
    if (!zarrMetadata || !zarrInfo) return 1;
    const axisNames = (zarrMetadata.axes as any[]).map((a) => a.name as string);
    const numLevels = zarrInfo.numLevels;
    const invertedIdx = Math.max(0, Math.min(numLevels - 1 - zoomLevel, numLevels - 1));
    const rawScales = (zarrMetadata.pyramidLayers[invertedIdx]?.scales ?? []) as any[];
    if (rawScales.length === 0) return 1;

    const getPixelSize = (axisName: string): number => {
      const idx = axisNames.indexOf(axisName);
      if (idx < 0 || idx >= rawScales.length) return 1;
      const v = rawScales[idx];
      const num = typeof v === "object" && v !== null ? v.value : v;
      return typeof num === "number" && num > 0 ? num : 1;
    };

    const [rowAxis, colAxis] =
      orientation === "XY" ? ["y", "x"] :
      orientation === "XZ" ? ["z", "x"] :
                             ["z", "y"];
    const ratio = getPixelSize(rowAxis) / getPixelSize(colAxis);
    return isFinite(ratio) && ratio > 0 ? ratio : 1;
  }, [zarrMetadata, zarrInfo, zoomLevel, orientation]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight" data-testid="text-inspect-title">
          Inspect / Visualize OME-Zarr
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Browse metadata, edit channels, and preview converted datasets.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground flex-1 min-w-0">
          <FolderOpen className="h-4 w-4 shrink-0" />
          <span className="font-mono text-xs truncate" data-testid="text-inspect-current-path">
            {inspectPath || "Set path in sidebar"}
          </span>
        </div>
        <Button
          data-testid="button-load-zarr"
          onClick={handleLoadZarr}
          disabled={!inspectPath}
          size="sm"
        >
          <Search className="h-4 w-4 mr-2" />
          Load
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2" data-testid="tabs-inspect-mode">
          <TabsTrigger value="metadata" data-testid="tab-metadata">
            <Info className="h-4 w-4 mr-1.5" />
            Metadata
          </TabsTrigger>
          <TabsTrigger value="viewer" data-testid="tab-viewer">
            <Microscope className="h-4 w-4 mr-1.5" />
            Viewer
          </TabsTrigger>
        </TabsList>

        <TabsContent value="metadata" className="space-y-4 mt-4">
          {!zarrMetadata && !isLoading && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <Database className="h-10 w-10 mb-3 opacity-30" />
                <p className="text-sm">Load an OME-Zarr dataset to view its metadata.</p>
              </CardContent>
            </Card>
          )}

          {isLoading && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="animate-pulse flex items-center gap-2 text-muted-foreground">
                  <Search className="h-5 w-5" />
                  <span>Loading metadata...</span>
                </div>
              </CardContent>
            </Card>
          )}

          {zarrMetadata && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Dataset Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-2 text-sm">
                    <div className="flex justify-between gap-2 min-w-0">
                      <span className="text-muted-foreground shrink-0">Name</span>
                      <span className="font-mono text-xs truncate text-right" data-testid="text-zarr-name">{zarrMetadata.name}</span>
                    </div>
                    <Separator />
                    <div className="flex justify-between gap-2 min-w-0">
                      <span className="text-muted-foreground shrink-0">NGFF Version</span>
                      <span className="font-mono text-xs">{zarrMetadata.ngffVersion}</span>
                    </div>
                    <Separator />
                    <div className="flex justify-between gap-1">
                      <span className="text-muted-foreground">Resolution Levels</span>
                      <Badge variant="secondary">{zarrMetadata.resolutionLevels}</Badge>
                    </div>
                    <Separator />
                    <div className="flex justify-between gap-1">
                      <span className="text-muted-foreground">Data Type</span>
                      <Badge variant="secondary" className="font-mono">{zarrMetadata.dataType}</Badge>
                    </div>
                    <Separator />
                    <div className="flex justify-between gap-2 min-w-0">
                      <span className="text-muted-foreground shrink-0">Compression</span>
                      <div className="flex flex-col items-end gap-0.5 min-w-0">
                        <Badge variant="secondary" className="font-mono text-[10px]">
                          {zarrMetadata.compression.name}
                        </Badge>
                        {Object.entries(zarrMetadata.compression.params).map(([k, v]) => (
                          <span key={k} className="font-mono text-[10px] text-muted-foreground">
                            {k}: {String(v)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Ruler className="h-4 w-4" />
                    Axes &amp; Pixel Sizes
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {zarrMetadata.axes.map((axis: any, i: number) => (
                      <Badge key={i} variant="secondary" className="font-mono">
                        {axis.name} ({axis.type})
                      </Badge>
                    ))}
                  </div>
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-1.5 pr-3 text-muted-foreground font-medium w-12">Axis</th>
                        <th className="text-left py-1.5 pr-3 text-muted-foreground font-medium w-20">Type</th>
                        <th className="text-left py-1.5 pr-3 text-muted-foreground font-medium">Pixel size</th>
                        <th className="text-left py-1.5 text-muted-foreground font-medium w-44">Unit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {editableScales.map((row, i) => {
                        const axMeta = zarrMetadata.axes[i] as any;
                        const isTime = axMeta?.type === "time";
                        const spaceUnits = ["picometer","nanometer","micrometer","millimeter","centimeter","decimeter","meter","decameter","hectometer","kilometer"];
                        const timeUnits  = ["nanosecond","microsecond","millisecond","second","minute","hour"];
                        const units = isTime ? timeUnits : spaceUnits;
                        const noScale = axMeta?.type === "channel" || axMeta?.type === "other";
                        return (
                          <tr key={row.axis} className="border-b border-border/40">
                            <td className="py-1.5 pr-3 font-mono font-semibold">{row.axis}</td>
                            <td className="py-1.5 pr-3 text-muted-foreground">{axMeta?.type ?? "—"}</td>
                            <td className="py-1.5 pr-3">
                              {noScale ? <span className="text-muted-foreground">—</span> : (
                                <input
                                  type="number"
                                  step="any"
                                  title={`Pixel size for ${row.axis} axis`}
                                  value={row.value}
                                  onChange={(e) => setEditableScales(prev => prev.map((r, j) => j === i ? { ...r, value: parseFloat(e.target.value) || 0 } : r))}
                                  className="w-28 h-6 px-1.5 rounded border border-input bg-background text-xs font-mono"
                                />
                              )}
                            </td>
                            <td className="py-1.5">
                              {noScale ? <span className="text-muted-foreground">—</span> : (
                                <Select
                                  value={row.unit}
                                  onValueChange={(v) => setEditableScales(prev => prev.map((r, j) => j === i ? { ...r, unit: v } : r))}
                                >
                                  <SelectTrigger className="h-6 text-xs w-40">
                                    <SelectValue placeholder="select unit" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {units.map(u => (
                                      <SelectItem key={u} value={u} className="text-xs">{u}</SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className="flex justify-end pt-1">
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      disabled={updateMetadataMutation.isPending}
                      onClick={() => inspectPath && updateMetadataMutation.mutate({
                        path: inspectPath,
                        scales: editableScales.filter((_, i) => {
                          const axMeta = zarrMetadata.axes[i] as any;
                          return axMeta?.type !== "channel" && axMeta?.type !== "other";
                        }),
                      })}
                    >
                      <Save className="h-3 w-3 mr-1.5" />
                      {updateMetadataMutation.isPending ? "Saving…" : "Save pixel sizes"}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {zarrMetadata.channels.length > 0 && (
                <Card className="md:col-span-2">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Palette className="h-4 w-4" />
                      OME Channels ({zarrMetadata.channels.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1">
                    {zarrMetadata.channels.map((ch) => {
                      const isOpen = openChannels.has(ch.index);
                      return (
                        <Collapsible
                          key={ch.index}
                          open={isOpen}
                          onOpenChange={() =>
                            setOpenChannels((prev) => {
                              const next = new Set(prev);
                              next.has(ch.index) ? next.delete(ch.index) : next.add(ch.index);
                              return next;
                            })
                          }
                        >
                          <CollapsibleTrigger asChild>
                            <button
                              type="button"
                              className="flex items-center justify-between w-full px-3 py-2 rounded-md text-sm hover:bg-muted/50 transition-colors"
                            >
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary" className="text-[10px] font-mono">
                                  Ch {ch.index}
                                </Badge>
                                <div
                                  className="h-3 w-3 rounded-full shrink-0 border border-border"
                                  style={{ backgroundColor: ch.color }}
                                />
                                <span className="text-sm">{ch.label}</span>
                                {!ch.visible && (
                                  <EyeOff className="h-3 w-3 text-muted-foreground" />
                                )}
                              </div>
                              <ChevronDown
                                className={`h-3.5 w-3.5 text-muted-foreground transition-transform duration-150 ${isOpen ? "rotate-180" : ""}`}
                              />
                            </button>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <div className="px-3 pb-3 pt-1">
                              <table className="w-full text-xs border-collapse">
                                <tbody>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium w-28">Label</td>
                                    <td className="py-1.5 font-mono">{ch.label}</td>
                                  </tr>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Color</td>
                                    <td className="py-1.5">
                                      <div className="flex items-center gap-2">
                                        <div
                                          className="h-3.5 w-3.5 rounded border border-border shrink-0"
                                          style={{ backgroundColor: ch.color }}
                                        />
                                        <span className="font-mono">{ch.color}</span>
                                      </div>
                                    </td>
                                  </tr>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Visible</td>
                                    <td className="py-1.5 font-mono">{ch.visible ? "true" : "false"}</td>
                                  </tr>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Data range</td>
                                    <td className="py-1.5 font-mono">{ch.window.min} – {ch.window.max}</td>
                                  </tr>
                                  <tr>
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Display range</td>
                                    <td className="py-1.5 font-mono">{ch.window.start} – {ch.window.end}</td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      );
                    })}
                  </CardContent>
                </Card>
              )}

              {zarrMetadata.pyramidLayers.length > 0 && (
                <Card className="md:col-span-2">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <Layers className="h-4 w-4" />
                      Pyramid Layers (Click to explore individual layers.)
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1">
                    {zarrMetadata.pyramidLayers.map((layer) => {
                      const axes = zarrMetadata.axes;
                      const isOpen = openLayers.has(layer.level);

                      // Per-axis pixel size: use stored scales if present, else derive from level-0 scales
                      const derivedScales = axes.map((ax, i) => {
                        if (layer.scales && layer.scales.length > i) {
                          const s = layer.scales[i] as any;
                          return typeof s === "object" && s !== null ? s.value : s;
                        }
                        const base = zarrMetadata.scales.find((s) => s.axis === ax.name);
                        if (base == null) return null;
                        const baseSize = zarrMetadata.shape[i] ?? 1;
                        const layerSize = layer.shape[i] ?? 1;
                        return layerSize > 0 ? base.value * (baseSize / layerSize) : base.value;
                      });

                      return (
                        <Collapsible
                          key={layer.level}
                          open={isOpen}
                          onOpenChange={() =>
                            setOpenLayers((prev) => {
                              const next = new Set(prev);
                              next.has(layer.level) ? next.delete(layer.level) : next.add(layer.level);
                              return next;
                            })
                          }
                        >
                          <CollapsibleTrigger asChild>
                            <button
                              type="button"
                              className="flex items-center justify-between w-full px-3 py-2 rounded-md text-sm hover:bg-muted/50 transition-colors"
                            >
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary" className="text-[10px] font-mono">
                                  Level {layer.level}
                                </Badge>
                                <span className="font-mono text-xs text-muted-foreground">
                                  {layer.shape.join(" × ")}
                                </span>
                              </div>
                              <ChevronDown
                                className={`h-3.5 w-3.5 text-muted-foreground transition-transform duration-150 ${isOpen ? "rotate-180" : ""}`}
                              />
                            </button>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <div className="overflow-x-auto px-3 pb-3 pt-1">
                              <table className="w-full text-xs border-collapse">
                                <thead>
                                  <tr className="border-b border-border">
                                    <th className="text-left py-1.5 pr-6 text-muted-foreground font-medium w-24" />
                                    {axes.map((ax) => (
                                      <th
                                        key={ax.name}
                                        className="text-center py-1.5 px-3 font-semibold uppercase tracking-wide"
                                      >
                                        {ax.name}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Shape</td>
                                    {axes.map((ax, i) => (
                                      <td key={ax.name} className="text-center py-1.5 px-3 font-mono">
                                        {layer.shape[i] ?? "—"}
                                      </td>
                                    ))}
                                  </tr>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Chunks</td>
                                    {axes.map((ax, i) => (
                                      <td key={ax.name} className="text-center py-1.5 px-3 font-mono">
                                        {layer.chunks[i] ?? "—"}
                                      </td>
                                    ))}
                                  </tr>
                                  <tr className="border-b border-border/40">
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Pixel size</td>
                                    {axes.map((ax, i) => {
                                      const v = derivedScales[i];
                                      return (
                                        <td key={ax.name} className="text-center py-1.5 px-3 font-mono">
                                          {v != null ? formatPixelSize(v) : "—"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td className="py-1.5 pr-6 text-muted-foreground font-medium">Unit</td>
                                    {axes.map((ax) => (
                                      <td key={ax.name} className="text-center py-1.5 px-3 text-muted-foreground">
                                        {ax.unit || "—"}
                                      </td>
                                    ))}
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      );
                    })}
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="viewer" forceMount className="space-y-4 mt-4 data-[state=inactive]:hidden">
          {!zarrInfo && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                <Microscope className="h-12 w-12 mb-4 opacity-20" />
                <p className="text-sm font-medium mb-1">OME-Zarr Viewer</p>
                <p className="text-xs text-center max-w-md">
                  Load a dataset to view 2D planes with lazy loading and automatic resolution switching.
                </p>
              </CardContent>
            </Card>
          )}

          {zarrInfo && (
            <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
              <div className="space-y-3">
                <Card className="overflow-hidden">
                  <CardContent className="p-0 relative">
                    <div
                      ref={containerRef}
                      className="relative bg-black flex items-center justify-center p-4 select-none overflow-hidden"
                      style={{ minHeight: 300, cursor: isDragging ? "grabbing" : "grab" }}
                      data-testid="viewer-canvas-container"
                      onMouseDown={handleMouseDown}
                      onWheel={handleWheel}
                    >
                      {imageLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                          <div className="flex flex-col items-center gap-2">
                            <Loader2 className="h-8 w-8 animate-spin text-white/70" />
                          </div>
                        </div>
                      )}
                      {imageError && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black z-10">
                          <p className="text-red-400 text-sm">{imageError}</p>
                        </div>
                      )}
                      {!canvasSize && !imageLoading && !imageError && (
                        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                          <Microscope className="h-10 w-10 mb-3 opacity-20" />
                          <p className="text-xs">Waiting for plane data...</p>
                        </div>
                      )}
                      <canvas
                        ref={canvasRef}
                        data-testid="viewer-canvas"
                        style={{
                          imageRendering: zoomLevel >= (zarrInfo.numLevels - 1) ? "pixelated" : "auto",
                          pointerEvents: "none",
                          display: canvasSize ? "block" : "none",
                          transform: isDragging ? `translate(${dragOffset.x}px, ${dragOffset.y}px)` : "none",
                          transition: isDragging ? "none" : "transform 0.1s ease-out",
                          aspectRatio: `1 / ${pixelAspectRatio}`,
                          maxWidth: "100%",
                          maxHeight: "70vh",
                          width: "auto",
                          height: "auto",
                        }}
                      />
                    </div>
                    <div className="px-3 py-1.5 bg-muted/50 border-t flex items-center justify-between text-[10px] font-mono text-muted-foreground">
                      <span data-testid="text-viewer-info">
                        Level {zarrInfo.resolutionPaths[Math.max(0, Math.min(zarrInfo.numLevels - 1 - zoomLevel, zarrInfo.numLevels - 1))] || "0"}
                        {currentLevelShape ? ` (${currentLevelShape.width}×${currentLevelShape.height})` : ""}
                        {" · "}FOV {fovSize}×{fovSize}
                        {currentLevelShape && (fovSize > currentLevelShape.width || fovSize > currentLevelShape.height)
                          ? ` → ${Math.min(fovSize, currentLevelShape.width)}×${Math.min(fovSize, currentLevelShape.height)}`
                          : ""}
                        {" · "}{zarrInfo.dtype}
                        {" · "}{orientation}
                      </span>
                      <span>
                        {hasZ && <span>Z={zSlice} · </span>}
                        Center ({fovCenterY}, {fovCenterX})
                        {pixelAspectRatio !== 1 && (
                          <span className="ml-2 text-yellow-400/80" title="Canvas stretched to account for pixel anisotropy">
                            · AR {pixelAspectRatio.toFixed(2)}
                          </span>
                        )}
                        {hasZ && <span className="ml-1 opacity-50">⎍scroll for Z</span>}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {(hasT || hasZ) && (
                  <Card>
                    <CardContent className="p-3 space-y-3">
                      {hasT && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs">Time (T)</Label>
                            <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                              {timeSlice} / {(dimSizes.t || 1) - 1}
                            </span>
                          </div>
                          <Slider
                            data-testid="slider-viewer-time"
                            min={0}
                            max={Math.max(0, (dimSizes.t || 1) - 1)}
                            step={1}
                            value={[timeSlice]}
                            onValueChange={([v]) => setTimeSlice(v)}
                          />
                        </div>
                      )}
                      {hasZ && orientation === "XY" && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs">Z Slice</Label>
                            <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                              {zSlice} / {(dimSizes.z || 1) - 1}
                            </span>
                          </div>
                          <Slider
                            data-testid="slider-viewer-z"
                            min={0}
                            max={Math.max(0, (dimSizes.z || 1) - 1)}
                            step={1}
                            value={[zSlice]}
                            onValueChange={([v]) => setZSlice(v)}
                          />
                        </div>
                      )}
                      {orientation === "XZ" && dimSizes.y !== undefined && dimSizes.y > 1 && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs">Y Slice</Label>
                            <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                              {zSlice} / {(dimSizes.y || 1) - 1}
                            </span>
                          </div>
                          <Slider
                            data-testid="slider-viewer-slice-xz"
                            min={0}
                            max={Math.max(0, (dimSizes.y || 1) - 1)}
                            step={1}
                            value={[zSlice]}
                            onValueChange={([v]) => setZSlice(v)}
                          />
                        </div>
                      )}
                      {orientation === "YZ" && dimSizes.x !== undefined && dimSizes.x > 1 && (
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs">X Slice</Label>
                            <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                              {zSlice} / {(dimSizes.x || 1) - 1}
                            </span>
                          </div>
                          <Slider
                            data-testid="slider-viewer-slice-yz"
                            min={0}
                            max={Math.max(0, (dimSizes.x || 1) - 1)}
                            step={1}
                            value={[zSlice]}
                            onValueChange={([v]) => setZSlice(v)}
                          />
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </div>

              <div className="space-y-3">
                <Card>
                  <CardHeader className="pb-2 pt-3 px-3">
                    <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                      <ZoomIn className="h-3.5 w-3.5" />
                      View Controls
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 space-y-3">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Zoom</Label>
                        <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                          {zoomLevel} / {Math.max(0, (zarrInfo.numLevels || 1) - 1)}
                        </span>
                      </div>
                      <Slider
                        data-testid="slider-viewer-zoom"
                        min={0}
                        max={Math.max(0, (zarrInfo.numLevels || 1) - 1)}
                        step={1}
                        value={[zoomLevel]}
                        onValueChange={([v]) => setZoomLevel(v)}
                      />
                    </div>

                    <div className="space-y-1">
                      <Label className="text-xs">Orientation</Label>
                      <div className="flex gap-1">
                        {(["XY", "XZ", "YZ"] as const).map((o) => (
                          <Button
                            key={o}
                            data-testid={`button-orientation-${o}`}
                            variant={orientation === o ? "default" : "outline"}
                            size="sm"
                            className="flex-1 h-7 text-xs"
                            onClick={() => setOrientation(o)}
                          >
                            {o}
                          </Button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <Label className="text-xs">FOV Size</Label>
                      <Select value={String(fovSize)} onValueChange={(v) => setFovSize(Number(v))}>
                        <SelectTrigger data-testid="select-fov-size" className="h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {FOV_SIZE_OPTIONS.map((s) => (
                            <SelectItem key={s} value={String(s)}>{s}×{s}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {fovWarning && (
                        <div className="flex items-start gap-1.5 p-1.5 rounded bg-amber-500/10 border border-amber-500/30" data-testid="text-fov-warning">
                          <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0 mt-0.5" />
                          <span className="text-[10px] text-amber-600 dark:text-amber-400 leading-tight">{fovWarning}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2 pt-3 px-3">
                    <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                      <Move className="h-3.5 w-3.5" />
                      Pan Position
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 space-y-3">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Y Center</Label>
                        <span className="text-[10px] font-mono text-muted-foreground">{fovCenterY}</span>
                      </div>
                      <Slider
                        data-testid="slider-fov-center-y"
                        min={0}
                        max={Math.max(0, maxY - 1)}
                        step={1}
                        value={[fovCenterY]}
                        onValueChange={([v]) => setFovCenterY(v)}
                      />
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">X Center</Label>
                        <span className="text-[10px] font-mono text-muted-foreground">{fovCenterX}</span>
                      </div>
                      <Slider
                        data-testid="slider-fov-center-x"
                        min={0}
                        max={Math.max(0, maxX - 1)}
                        step={1}
                        value={[fovCenterX]}
                        onValueChange={([v]) => setFovCenterX(v)}
                      />
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full h-7 text-xs"
                      data-testid="button-fit-to-view"
                      onClick={handleFitToView}
                    >
                      <Maximize2 className="h-3 w-3 mr-1" />
                      Fit to View
                    </Button>
                  </CardContent>
                </Card>

                {viewerChannels.length > 0 && (
                  <Card>
                    <CardHeader className="pb-2 pt-3 px-3">
                      <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                        <Palette className="h-3.5 w-3.5" />
                        Channels ({viewerChannels.length})
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-3 pb-3 space-y-3">
                      <div className="overflow-y-auto" style={{ maxHeight: 400 }}>
                        <div className="space-y-3 pr-1">
                          {viewerChannels.map((ch) => (
                            <div key={ch.index} className="space-y-2">
                              <div className="flex items-center gap-2">
                                <button
                                  data-testid={`button-viewer-toggle-ch-${ch.index}`}
                                  className="shrink-0"
                                  onClick={() => updateViewerChannel(ch.index, { visible: !ch.visible })}
                                >
                                  {ch.visible ? (
                                    <Eye className="h-3.5 w-3.5 text-foreground" />
                                  ) : (
                                    <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
                                  )}
                                </button>
                                <div
                                  className="h-3 w-3 rounded-full shrink-0 border border-border"
                                  style={{ backgroundColor: ch.color }}
                                />
                                <input
                                  data-testid={`input-viewer-label-${ch.index}`}
                                  value={ch.label}
                                  onChange={(e) => updateViewerChannel(ch.index, { label: e.target.value })}
                                  className="text-xs truncate flex-1 h-6 px-1.5 rounded border border-transparent hover:border-border focus:border-primary focus:outline-none bg-transparent"
                                  title="Edit channel label"
                                />
                              </div>
                              {ch.visible && (
                                <div className="pl-5 space-y-1.5">
                                  <div className="flex items-center gap-2">
                                    <button
                                      type="button"
                                      title="Set slider range to actual data min/max"
                                      disabled={histAdjLoading[ch.index]}
                                      onClick={() => adjustHistogram(ch.index)}
                                      className="flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground disabled:opacity-50 transition-colors"
                                    >
                                      {histAdjLoading[ch.index]
                                        ? <Loader2 className="h-3 w-3 animate-spin" />
                                        : <BarChart2 className="h-3 w-3" />}
                                      <span>Adjust range</span>
                                    </button>
                                    <button
                                      type="button"
                                      title="Reset slider range to full dtype range"
                                      onClick={() => resetHistogram(ch.index)}
                                      className="flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                      <RotateCcw className="h-3 w-3" />
                                      <span>Reset range</span>
                                    </button>
                                  </div>
                                  <div className="flex items-center gap-1.5">
                                    <input
                                      type="number"
                                      data-testid={`input-viewer-intensity-min-${ch.index}`}
                                      value={ch.intensityMin}
                                      min={ch.window.min}
                                      max={ch.intensityMax}
                                      step={(ch.window.max - ch.window.min) / 1000 || 1}
                                      onChange={(e) => {
                                        const v = parseFloat(e.target.value);
                                        if (!isNaN(v)) updateViewerChannel(ch.index, { intensityMin: Math.max(ch.window.min, Math.min(v, ch.intensityMax)) });
                                      }}
                                      className="h-5 w-12 text-[10px] font-mono text-center px-0.5 rounded border border-border bg-background text-foreground disabled:opacity-50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                    />
                                    <Slider
                                      data-testid={`slider-viewer-intensity-${ch.index}`}
                                      min={ch.window.min}
                                      max={ch.window.max}
                                      step={(ch.window.max - ch.window.min) / 1000 || 1}
                                      value={[ch.intensityMin, ch.intensityMax]}
                                      onValueChange={([min, max]) =>
                                        updateViewerChannel(ch.index, { intensityMin: min, intensityMax: max })
                                      }
                                      className="flex-1"
                                    />
                                    <input
                                      type="number"
                                      data-testid={`input-viewer-intensity-max-${ch.index}`}
                                      value={ch.intensityMax}
                                      min={ch.intensityMin}
                                      max={ch.window.max}
                                      step={(ch.window.max - ch.window.min) / 1000 || 1}
                                      onChange={(e) => {
                                        const v = parseFloat(e.target.value);
                                        if (!isNaN(v)) updateViewerChannel(ch.index, { intensityMax: Math.max(ch.intensityMin, Math.min(v, ch.window.max)) });
                                      }}
                                      className="h-5 w-12 text-[10px] font-mono text-center px-0.5 rounded border border-border bg-background text-foreground disabled:opacity-50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                    />
                                  </div>
                                  <div className="flex items-center gap-1.5">
                                    <div className="flex flex-wrap gap-1">
                                      {["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFFFFF"].map((c) => (
                                        <button
                                          key={c}
                                          data-testid={`button-viewer-color-${ch.index}-${c.replace("#", "")}`}
                                          className={`h-4 w-4 rounded-sm border transition-all ${
                                            ch.color === c ? "border-foreground scale-125" : "border-transparent"
                                          }`}
                                          style={{ backgroundColor: c }}
                                          onClick={() => updateViewerChannel(ch.index, { color: c })}
                                        />
                                      ))}
                                    </div>
                                    <input
                                      type="color"
                                      data-testid={`input-viewer-color-picker-${ch.index}`}
                                      value={ch.color}
                                      onChange={(e) => updateViewerChannel(ch.index, { color: e.target.value.toUpperCase() })}
                                      className="h-4 w-4 shrink-0 cursor-pointer border-0 p-0 bg-transparent"
                                      title="Pick custom color"
                                    />
                                  </div>
                                  <div className="flex items-center gap-1.5">
                                    <span className="text-[10px] text-muted-foreground">Hex</span>
                                    <input
                                      type="text"
                                      data-testid={`input-viewer-hex-${ch.index}`}
                                      value={ch.color}
                                      onChange={(e) => {
                                        let val = e.target.value.toUpperCase();
                                        if (!val.startsWith("#")) val = "#" + val;
                                        if (/^#[0-9A-F]{0,6}$/.test(val)) {
                                          if (val.length === 7) {
                                            updateViewerChannel(ch.index, { color: val });
                                          }
                                        }
                                      }}
                                      onBlur={(e) => {
                                        let val = e.target.value.toUpperCase();
                                        if (!val.startsWith("#")) val = "#" + val;
                                        if (/^#[0-9A-F]{6}$/.test(val)) {
                                          updateViewerChannel(ch.index, { color: val });
                                        }
                                      }}
                                      className="h-5 w-[4.5rem] text-[10px] font-mono px-1 rounded border border-border bg-background text-foreground"
                                      maxLength={7}
                                      placeholder="#FF0000"
                                    />
                                  </div>
                                </div>
                              )}
                              {ch.index < viewerChannels.length - 1 && <Separator />}
                            </div>
                          ))}
                        </div>
                      </div>
                      <Separator />
                      <div className="flex flex-col gap-1.5">
                        <Button
                          size="sm"
                          variant="secondary"
                          className="w-full h-7 text-xs"
                          data-testid="button-save-channel-metadata"
                          disabled={updateChannelMetadataMutation.isPending}
                          onClick={() => {
                            if (!inspectPath) return;
                            updateChannelMetadataMutation.mutate({
                              path: inspectPath,
                              channels: viewerChannels.map((ch) => ({
                                index: ch.index,
                                label: ch.label,
                                color: ch.color,
                                window: { start: ch.intensityMin, end: ch.intensityMax, min: ch.window.min, max: ch.window.max },
                              })),
                            });
                          }}
                        >
                          <Save className="h-3 w-3 mr-1.5" />
                          {updateChannelMetadataMutation.isPending ? "Saving..." : "Save to Metadata"}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="w-full h-7 text-xs"
                          data-testid="button-reset-channels"
                          onClick={() => {
                            if (zarrInfo) {
                              setViewerChannels(zarrInfo.channels.map((ch: any) => ({
                                index: ch.index,
                                label: ch.label || `Channel ${ch.index}`,
                                color: ch.color || "#FFFFFF",
                                visible: ch.visible !== false,
                                intensityMin: ch.window?.start ?? 0,
                                intensityMax: ch.window?.end ?? 255,
                                window: { min: ch.window?.min ?? 0, max: ch.window?.max ?? 255, start: ch.window?.start ?? 0, end: ch.window?.end ?? 255 },
                              })));
                            }
                          }}
                        >
                          <RotateCcw className="h-3 w-3 mr-1.5" />
                          Reset
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
