import { create } from "zustand";
import type {
  ConcatenationSettings,
  ClusterSettings,
  ReaderSettings,
  ConversionSettings,
  DownscalingSettings,
  MetadataSettings,
  ConversionJob,
} from "@shared/schema";

type AppMode = "convert" | "inspect";
type ConvertTab = "cluster" | "reader" | "conversion" | "downscaling" | "metadata";

interface ConversionStore {
  mode: AppMode;
  setMode: (mode: AppMode) => void;

  activeTab: ConvertTab;
  setActiveTab: (tab: ConvertTab) => void;

  inputPath: string;
  setInputPath: (path: string) => void;
  selectedInputPaths: string[];
  setSelectedInputPaths: (paths: string[]) => void;
  toggleSelectedInputPath: (path: string) => void;
  outputPath: string;
  setOutputPath: (path: string) => void;
  includePattern: string;
  setIncludePattern: (pattern: string) => void;
  excludePattern: string;
  setExcludePattern: (pattern: string) => void;

  inspectPath: string;
  setInspectPath: (path: string) => void;

  concatenation: ConcatenationSettings;
  setConcatenation: (settings: Partial<ConcatenationSettings>) => void;

  cluster: ClusterSettings;
  setCluster: (settings: Partial<ClusterSettings>) => void;

  reader: ReaderSettings;
  setReader: (settings: Partial<ReaderSettings>) => void;

  conversion: ConversionSettings;
  setConversion: (settings: Partial<ConversionSettings>) => void;

  downscaling: DownscalingSettings;
  setDownscaling: (settings: Partial<DownscalingSettings>) => void;

  metadata: MetadataSettings;
  setMetadata: (settings: Partial<MetadataSettings>) => void;

  currentJob: ConversionJob | null;
  setCurrentJob: (job: ConversionJob | null) => void;

  logs: string[];
  addLog: (log: string) => void;
  clearLogs: () => void;

  isRunning: boolean;
  setIsRunning: (running: boolean) => void;
}

export const useConversionStore = create<ConversionStore>((set) => ({
  mode: "convert",
  setMode: (mode) => set({ mode }),

  activeTab: "cluster",
  setActiveTab: (activeTab) => set({ activeTab }),

  inputPath: "",
  setInputPath: (inputPath) => set({ inputPath }),
  selectedInputPaths: [],
  setSelectedInputPaths: (selectedInputPaths) => set({ selectedInputPaths }),
  toggleSelectedInputPath: (path) =>
    set((s) => ({
      selectedInputPaths: s.selectedInputPaths.includes(path)
        ? s.selectedInputPaths.filter((p) => p !== path)
        : [...s.selectedInputPaths, path],
    })),
  outputPath: "",
  setOutputPath: (outputPath) => set({ outputPath }),
  includePattern: "",
  setIncludePattern: (includePattern) => set({ includePattern }),
  excludePattern: "",
  setExcludePattern: (excludePattern) => set({ excludePattern }),

  inspectPath: "",
  setInspectPath: (inspectPath) => set({ inspectPath }),

  concatenation: {
    enabled: false,
    timeTag: "",
    channelTag: "",
    xTag: "",
    zTag: "",
    yTag: "",
    concatenationAxes: "",
  },
  setConcatenation: (settings) =>
    set((state) => ({ concatenation: { ...state.concatenation, ...settings } })),

  cluster: {
    maxWorkers: 4,
    queueSize: 10,
    maxConcurrency: 4,
    regionSizeMb: 64,
    memoryPerWorker: "4GB",
    useLocalDask: false,
    useSlurm: false,
  },
  setCluster: (settings) =>
    set((state) => ({ cluster: { ...state.cluster, ...settings } })),

  reader: {
    readAllScenes: true,
    sceneIndices: "",
    readAllTiles: true,
    mosaicTileIndices: "",
    readAsMosaic: false,
    viewIndex: "",
    phaseIndex: "",
    illuminationIndex: "",
    rotationIndex: "",
    sampleIndex: "",
  },
  setReader: (settings) =>
    set((state) => ({ reader: { ...state.reader, ...settings } })),

  conversion: {
    zarrFormat: 2,
    dataType: "auto",
    compression: {
      codec: "blosc",
      level: 5,
      bloscInnerCodec: "lz4",
      bloscShuffle: "shuffle",
    },
    verbose: false,
    overwrite: false,
    squeezeDimensions: true,
    saveOmeXml: false,
    overrideChannelNames: false,
    skipDask: false,
    autoChunk: true,
    targetChunkSizeMb: 32,
    chunkTime: 1,
    chunkZ: 1,
    chunkX: 256,
    chunkChannel: 1,
    chunkY: 256,
    shardTime: 1,
    shardZ: 1,
    shardX: 1,
    shardChannel: 1,
    shardY: 1,
    dimRangeTime: "",
    dimRangeChannel: "",
    dimRangeZ: "",
    dimRangeY: "",
    dimRangeX: "",
  },
  setConversion: (settings) =>
    set((state) => ({ conversion: { ...state.conversion, ...settings } })),

  downscaling: {
    autoDetectLayers: true,
    numLayers: 4,
    minDimSize: 64,
    downscaleMethod: "simple" as const,
    scaleTime: 1,
    scaleChannel: 1,
    scaleZ: 1,
    scaleY: 2,
    scaleX: 2,
  },
  setDownscaling: (settings) =>
    set((state) => ({ downscaling: { ...state.downscaling, ...settings } })),

  metadata: {
    metadataReader: "bioio",
    channelIntensityLimits: "from_datatype",
    overridePhysicalScale: false,
    scaleTime: "",
    unitTime: "second",
    scaleZ: "",
    unitZ: "micrometer",
    scaleY: "",
    unitY: "micrometer",
    scaleX: "",
    unitX: "micrometer",
  },
  setMetadata: (settings) =>
    set((state) => ({ metadata: { ...state.metadata, ...settings } })),

  currentJob: null,
  setCurrentJob: (currentJob) => set({ currentJob }),

  logs: [],
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  clearLogs: () => set({ logs: [] }),

  isRunning: false,
  setIsRunning: (isRunning) => set({ isRunning }),
}));
