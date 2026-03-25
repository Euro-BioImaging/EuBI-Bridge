import { z } from "zod";

export const concatenationSettingsSchema = z.object({
  enabled: z.boolean().default(false),
  timeTag: z.string().default(""),
  channelTag: z.string().default(""),
  xTag: z.string().default(""),
  zTag: z.string().default(""),
  yTag: z.string().default(""),
  concatenationAxes: z.string().default(""),
});

export const clusterSettingsSchema = z.object({
  maxWorkers: z.number().min(1).max(64).default(4),
  queueSize: z.number().min(1).max(100).default(10),
  maxConcurrency: z.number().min(1).max(32).default(4),
  regionSizeMb: z.number().min(1).max(1024).default(64),
  memoryPerWorker: z.string().default("4GB"),
  useLocalDask: z.boolean().default(false),
  useSlurm: z.boolean().default(false),
});

export const readerSettingsSchema = z.object({
  readAllScenes: z.boolean().default(true),
  sceneIndices: z.string().default(""),
  readAllTiles: z.boolean().default(true),
  mosaicTileIndices: z.string().default(""),
  readAsMosaic: z.boolean().default(false),
  viewIndex: z.string().default(""),
  phaseIndex: z.string().default(""),
  illuminationIndex: z.string().default(""),
  rotationIndex: z.string().default(""),
  sampleIndex: z.string().default(""),
});

export const compressionSettingsSchema = z.object({
  codec: z.enum(["blosc", "zstd", "gzip", "bz2", "none"]).default("blosc"),
  level: z.number().min(0).max(22).default(5),
  bloscInnerCodec: z.enum(["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]).default("lz4"),
  bloscShuffle: z.enum(["noshuffle", "shuffle", "bitshuffle"]).default("shuffle"),
});

export const CODECS_BY_FORMAT: Record<number, Array<{ value: string; label: string }>> = {
  2: [
    { value: "blosc", label: "Blosc" },
    { value: "gzip", label: "GZip" },
    { value: "zstd", label: "Zstd" },
    { value: "bz2", label: "BZ2" },
    { value: "none", label: "None" },
  ],
  3: [
    { value: "blosc", label: "Blosc" },
    { value: "gzip", label: "GZip" },
    { value: "zstd", label: "Zstd" },
    { value: "none", label: "None" },
  ],
};

export const CODEC_PARAMS: Record<string, { hasLevel: boolean; maxLevel: number; hasBlosc: boolean }> = {
  blosc: { hasLevel: true, maxLevel: 9, hasBlosc: true },
  gzip: { hasLevel: true, maxLevel: 9, hasBlosc: false },
  zstd: { hasLevel: true, maxLevel: 22, hasBlosc: false },
  bz2: { hasLevel: true, maxLevel: 9, hasBlosc: false },
  none: { hasLevel: false, maxLevel: 0, hasBlosc: false },
};

export const conversionSettingsSchema = z.object({
  zarrFormat: z.number().min(2).max(3).default(2),
  dataType: z.enum(["auto", "uint8", "uint16", "uint32", "float32", "float64"]).default("auto"),
  compression: compressionSettingsSchema.default({}),
  verbose: z.boolean().default(false),
  overwrite: z.boolean().default(false),
  squeezeDimensions: z.boolean().default(true),
  saveOmeXml: z.boolean().default(false),
  overrideChannelNames: z.boolean().default(false),
  skipDask: z.boolean().default(false),
  autoChunk: z.boolean().default(true),
  targetChunkSizeMb: z.number().min(1).max(512).default(32),
  chunkTime: z.number().min(1).default(1),
  chunkZ: z.number().min(1).default(1),
  chunkX: z.number().min(1).default(256),
  chunkChannel: z.number().min(1).default(1),
  chunkY: z.number().min(1).default(256),
  shardTime: z.number().min(1).default(1),
  shardZ: z.number().min(1).default(1),
  shardX: z.number().min(1).default(1),
  shardChannel: z.number().min(1).default(1),
  shardY: z.number().min(1).default(1),
  dimRangeTime: z.string().default(""),
  dimRangeChannel: z.string().default(""),
  dimRangeZ: z.string().default(""),
  dimRangeY: z.string().default(""),
  dimRangeX: z.string().default(""),
});

export const downscalingSettingsSchema = z.object({
  autoDetectLayers: z.boolean().default(true),
  numLayers: z.number().min(1).max(10).default(4),
  minDimSize: z.number().min(1).default(64),
  downscaleMethod: z.enum(["simple", "mean", "median", "min", "max", "mode"]).default("simple"),
  scaleTime: z.number().min(1).default(1),
  scaleChannel: z.number().min(1).default(1),
  scaleZ: z.number().min(1).default(1),
  scaleY: z.number().min(1).default(2),
  scaleX: z.number().min(1).default(2),
});

export const metadataSettingsSchema = z.object({
  metadataReader: z.enum(["bfio", "bioio"]).default("bioio"),
  channelIntensityLimits: z.enum(["from_datatype", "from_array"]).default("from_datatype"),
  overridePhysicalScale: z.boolean().default(false),
  scaleTime: z.string().default(""),
  unitTime: z.string().default("second"),
  scaleZ: z.string().default(""),
  unitZ: z.string().default("micrometer"),
  scaleY: z.string().default(""),
  unitY: z.string().default("micrometer"),
  scaleX: z.string().default(""),
  unitX: z.string().default("micrometer"),
});

export const conversionJobSchema = z.object({
  id: z.string(),
  inputPath: z.string(),
  outputPath: z.string(),
  includePattern: z.string().default(""),
  excludePattern: z.string().default(""),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]).default("pending"),
  progress: z.number().min(0).max(100).default(0),
  logs: z.array(z.string()).default([]),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  config: z.object({
    concatenation: concatenationSettingsSchema,
    cluster: clusterSettingsSchema,
    reader: readerSettingsSchema,
    conversion: conversionSettingsSchema,
    downscaling: downscalingSettingsSchema,
    metadata: metadataSettingsSchema,
  }),
});

export const channelInfoSchema = z.object({
  index: z.number(),
  label: z.string(),
  color: z.string(),
  visible: z.boolean().default(true),
  intensityMin: z.number().default(0),
  intensityMax: z.number().default(65535),
  window: z.object({
    start: z.number(),
    end: z.number(),
    min: z.number(),
    max: z.number(),
  }),
});

export const zarrMetadataSchema = z.object({
  name: z.string(),
  ngffVersion: z.string(),
  resolutionLevels: z.number(),
  dataType: z.string(),
  axes: z.array(z.object({
    name: z.string(),
    type: z.string(),
    unit: z.string().optional(),
  })),
  shape: z.array(z.number()),
  chunks: z.array(z.number()),
  compression: z.object({
    name: z.string(),
    params: z.record(z.unknown()),
  }),
  channels: z.array(channelInfoSchema),
  pyramidLayers: z.array(z.object({
    level: z.number(),
    shape: z.array(z.number()),
    chunks: z.array(z.number()),
    scales: z.array(z.number()).optional(),
  })),
  scales: z.array(z.object({
    axis: z.string(),
    value: z.number(),
    unit: z.string(),
  })),
});

export const fileEntrySchema = z.object({
  name: z.string(),
  path: z.string(),
  isDirectory: z.boolean(),
  size: z.number().optional(),
  modified: z.string().optional(),
});

export type ConcatenationSettings = z.infer<typeof concatenationSettingsSchema>;
export type ClusterSettings = z.infer<typeof clusterSettingsSchema>;
export type ReaderSettings = z.infer<typeof readerSettingsSchema>;
export type CompressionSettings = z.infer<typeof compressionSettingsSchema>;
export type ConversionSettings = z.infer<typeof conversionSettingsSchema>;
export type DownscalingSettings = z.infer<typeof downscalingSettingsSchema>;
export type MetadataSettings = z.infer<typeof metadataSettingsSchema>;
export type ConversionJob = z.infer<typeof conversionJobSchema>;
export type ChannelInfo = z.infer<typeof channelInfoSchema>;
export type ZarrMetadata = z.infer<typeof zarrMetadataSchema>;
export type FileEntry = z.infer<typeof fileEntrySchema>;

export type ConversionConfig = {
  concatenation: ConcatenationSettings;
  cluster: ClusterSettings;
  reader: ReaderSettings;
  conversion: ConversionSettings;
  downscaling: DownscalingSettings;
  metadata: MetadataSettings;
};
