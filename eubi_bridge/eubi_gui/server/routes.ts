import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { z } from "zod";
import { spawn, spawnSync, type ChildProcess } from "child_process";
import * as readline from "readline";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In production (node dist/index.cjs) __dirname is dist/.
// gui_react.py sets EUBI_SCRIPTS_DIR to <package>/eubi_bridge/eubi_gui/server/
// so Python worker scripts are found correctly in both dev and production.
const SCRIPTS_DIR: string = process.env.EUBI_SCRIPTS_DIR ?? __dirname;

/**
 * Resolve the Python executable to use for subprocesses.
 * Priority:
 *  1. PYTHON_EXECUTABLE environment variable
 *  2. venv Python at ../../venv (relative to server/ dir) — works when eubi_gui
 *     lives inside the WS-EuBI-Bridge project tree
 *  3. Fall back to "python" (Windows) or "python3" (Unix)
 */
function findPythonExecutable(): string {
  if (process.env.PYTHON_EXECUTABLE) return process.env.PYTHON_EXECUTABLE;
  // __dirname = <project-root>/eubi_gui/server  →  project root is two levels up
  const projectRoot = path.resolve(__dirname, "..", "..");
  const venvWin = path.join(projectRoot, "venv", "Scripts", "python.exe");
  const venvUnix = path.join(projectRoot, "venv", "bin", "python");
  if (os.platform() === "win32" && fs.existsSync(venvWin)) return venvWin;
  if (os.platform() !== "win32" && fs.existsSync(venvUnix)) return venvUnix;
  return os.platform() === "win32" ? "python" : "python3";
}

const PYTHON_EXECUTABLE = findPythonExecutable();
console.log(`[Python] Using executable: ${PYTHON_EXECUTABLE}`);

const activeProcesses = new Map<string, ChildProcess>();

const PLANE_SERVER_PORT = 5555;
let planeServerProcess: ChildProcess | null = null;
let planeServerReady = false;

function startPlaneServer() {
  if (planeServerProcess) return;

  const scriptPath = path.join(SCRIPTS_DIR, "zarr_plane_server.py");
  const proc = spawn(PYTHON_EXECUTABLE, [scriptPath, String(PLANE_SERVER_PORT)], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env },
  });

  planeServerProcess = proc;

  const rl = readline.createInterface({ input: proc.stdout! });
  rl.on("line", (line) => {
    try {
      const msg = JSON.parse(line);
      if (msg.type === "ready") {
        planeServerReady = true;
        console.log(`[PlaneServer] Ready on port ${msg.port}`);
      }
    } catch {}
  });

  proc.stderr?.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) console.log(`[PlaneServer] ${text}`);
  });

  proc.on("close", (code) => {
    console.log(`[PlaneServer] Exited with code ${code}`);
    planeServerProcess = null;
    planeServerReady = false;
    setTimeout(() => {
      console.log("[PlaneServer] Restarting...");
      startPlaneServer();
    }, 2000);
  });

  proc.on("error", (err) => {
    console.error(`[PlaneServer] Error: ${err.message}`);
    planeServerProcess = null;
    planeServerReady = false;
  });
}

function stopPlaneServer() {
  if (planeServerProcess) {
    planeServerProcess.kill("SIGTERM");
    planeServerProcess = null;
    planeServerReady = false;
  }
}

async function invalidatePlaneServerCache(zarrPath: string) {
  if (!planeServerReady) return;
  try {
    await fetch(`http://127.0.0.1:${PLANE_SERVER_PORT}/invalidate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: zarrPath }),
    });
  } catch { /* best-effort */ }
}

async function proxyToPlaneServer(reqPath: string, res: any) {
  if (!planeServerReady) {
    return res.status(503).json({ error: "Plane server not ready" });
  }
  try {
    const url = `http://127.0.0.1:${PLANE_SERVER_PORT}${reqPath}`;
    const response = await fetch(url);
    const contentType = response.headers.get("content-type") || "application/octet-stream";
    res.status(response.status);
    res.set("Content-Type", contentType);
    if (contentType.startsWith("image/")) {
      // Pass through caching headers from the Python server (ETag, Cache-Control)
      const cc = response.headers.get("Cache-Control");
      const etag = response.headers.get("ETag");
      if (cc) res.set("Cache-Control", cc);
      if (etag) res.set("ETag", etag);
    } else {
      res.set("Cache-Control", "no-store, no-cache, must-revalidate");
      res.set("Pragma", "no-cache");
      res.removeHeader("ETag");
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    res.end(buffer);
  } catch (err: any) {
    res.status(502).json({ error: `Plane server unreachable: ${err.message}` });
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  startPlaneServer();

  process.on("SIGTERM", stopPlaneServer);
  process.on("SIGINT", stopPlaneServer);
  const wss = new WebSocketServer({ server: httpServer, path: "/ws" });
  const jobSubscriptions = new Map<string, Set<WebSocket>>();

  wss.on("connection", (ws) => {
    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "subscribe" && msg.jobId) {
          if (!jobSubscriptions.has(msg.jobId)) {
            jobSubscriptions.set(msg.jobId, new Set());
          }
          jobSubscriptions.get(msg.jobId)!.add(ws);
        }
      } catch {}
    });

    ws.on("close", () => {
      jobSubscriptions.forEach((subs) => {
        subs.delete(ws);
      });
    });
  });

  function broadcastToJob(jobId: string, message: Record<string, any>) {
    const subs = jobSubscriptions.get(jobId);
    if (!subs) return;
    const payload = JSON.stringify(message);
    subs.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(payload);
      }
    });
  }

  const startConversionSchema = z.object({
    inputPath: z.string().optional(),
    inputPaths: z.array(z.string()).optional(),
    outputPath: z.string().min(1),
    includePattern: z.string().optional(),
    excludePattern: z.string().optional(),
    concatenation: z.record(z.any()).optional(),
    cluster: z.record(z.any()).optional(),
    reader: z.record(z.any()).optional(),
    conversion: z.record(z.any()).optional(),
    downscaling: z.record(z.any()).optional(),
    metadata: z.record(z.any()).optional(),
  });

  app.post("/api/conversion/start", async (req, res) => {
    try {
      const parsed = startConversionSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ message: "Invalid configuration", errors: parsed.error.flatten() });
      }
      const config = parsed.data;

      const zarrFormat = config.conversion?.zarrFormat || 2;
      const codec = config.conversion?.compression?.codec || "blosc";
      const v2Codecs = ["blosc", "gzip", "zstd", "bz2", "none"];
      const v3Codecs = ["blosc", "gzip", "zstd", "none"];
      const allowedCodecs = zarrFormat === 3 ? v3Codecs : v2Codecs;
      if (!allowedCodecs.includes(codec)) {
        return res.status(400).json({ message: `Codec "${codec}" is not supported for Zarr v${zarrFormat}. Supported: ${allowedCodecs.join(", ")}` });
      }

      const maxLevels: Record<string, number> = { blosc: 9, gzip: 9, zstd: 22, bz2: 9, none: 0 };
      const level = config.conversion?.compression?.level ?? 5;
      const maxLevel = maxLevels[codec] ?? 9;
      if (level > maxLevel) {
        if (config.conversion && config.conversion.compression) {
          config.conversion.compression.level = maxLevel;
        }
      }

      const job = await storage.createJob(config);

      await storage.updateJob(job.id, { status: "running" });

      runConversion(job.id, config, broadcastToJob);

      res.json(job);
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to start conversion" });
    }
  });

  app.get("/api/conversion/:id", async (req, res) => {
    const job = await storage.getJob(req.params.id);
    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }
    res.json(job);
  });

  app.post("/api/conversion/:id/cancel", async (req, res) => {
    const jobId = req.params.id;
    const job = await storage.updateJob(jobId, { status: "cancelled" });
    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }

    const proc = activeProcesses.get(jobId);
    if (proc && !proc.killed) {
      proc.kill("SIGTERM");
      setTimeout(() => {
        if (proc && !proc.killed) {
          proc.kill("SIGKILL");
        }
      }, 3000);
    }

    broadcastToJob(jobId, { type: "error", message: "Conversion cancelled by user" });
    res.json(job);
  });

  app.get("/api/conversion", async (_req, res) => {
    const jobs = await storage.getAllJobs();
    res.json(jobs);
  });

  app.get("/api/homedir", (_req, res) => {
    res.json({ homedir: os.homedir() });
  });

  // ── Config management ──────────────────────────────────────────────────────
  const CONFIG_SCRIPT = path.join(SCRIPTS_DIR, "config_manager.py");

  /** Run config_manager.py synchronously.
   *  All input (action, path, payload) is sent as a single JSON object via stdin,
   *  completely avoiding Windows command-line backslash / quoting issues. */
  function runConfigManager(
    request: { action: string; configPath?: string; config?: any },
  ): { ok: true; data: any } | { ok: false; error: string } {
    const stdinData = JSON.stringify(request);
    const result = spawnSync(PYTHON_EXECUTABLE, [CONFIG_SCRIPT], {
      encoding: "utf8",
      windowsHide: true,
      input: stdinData,
      maxBuffer: 10 * 1024 * 1024,
    });
    const stdout = (result.stdout || "").trim();
    const stderr = (result.stderr || "").trim();
    if (result.status !== 0 || !stdout) {
      try {
        const parsed = JSON.parse(stdout);
        return { ok: false, error: parsed.error || stderr || "Unknown error" };
      } catch {
        return { ok: false, error: stderr || stdout || "config_manager.py failed" };
      }
    }
    try {
      return { ok: true, data: JSON.parse(stdout) };
    } catch {
      return { ok: false, error: `Invalid JSON from config_manager: ${stdout.slice(0, 200)}` };
    }
  }

  /** GET /api/config?path=<optional> — load config, return camelCase JSON */
  app.get("/api/config", (req, res) => {
    const configPath = (req.query.path as string) || "";
    const result = runConfigManager({ action: "get", configPath: configPath || undefined });
    if (!result.ok) return res.status(500).json({ message: result.error });
    res.json(result.data);
  });

  /** POST /api/config?path=<optional> — save camelCase form state to JSON file */
  app.post("/api/config", (req, res) => {
    const configPath = (req.query.path as string) || path.join(os.homedir(), ".eubi_bridge");
    const result = runConfigManager({ action: "save", configPath, config: req.body });
    if (!result.ok) return res.status(500).json({ message: result.error });
    res.json(result.data);
  });

  /** POST /api/config/reset?path=<optional> — reset to root_defaults */
  app.post("/api/config/reset", (req, res) => {
    const configPath = (req.query.path as string) || "";
    const result = runConfigManager({ action: "reset", configPath: configPath || undefined });
    if (!result.ok) return res.status(500).json({ message: result.error });
    res.json(result.data);
  });
  // ──────────────────────────────────────────────────────────────────────────

  app.post("/api/files/mkdir", async (req, res) => {
    try {
      const { dirPath, parentPath, name } = req.body;
      let resolved: string;
      if (parentPath && name) {
        resolved = path.resolve(path.join(parentPath, name));
      } else if (dirPath && typeof dirPath === "string") {
        resolved = path.resolve(dirPath);
      } else {
        return res.status(400).json({ message: "parentPath+name or dirPath is required" });
      }
      fs.mkdirSync(resolved, { recursive: true });
      res.json({ path: resolved });
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to create directory" });
    }
  });

  app.get("/api/files", async (req, res) => {
    try {
      const rawPath = (req.query.path as string) || os.homedir();
      const dirPath = rawPath.replace(/^~(?=$|\/|\\)/, os.homedir());
      const resolvedPath = path.resolve(dirPath);

      if (!fs.existsSync(resolvedPath)) {
        return res.status(404).json({ message: "Path not found" });
      }

      const stat = fs.statSync(resolvedPath);
      if (!stat.isDirectory()) {
        return res.status(400).json({ message: "Path is not a directory" });
      }

      const entries = fs.readdirSync(resolvedPath, { withFileTypes: true });
      const items = entries
        .map((entry) => {
          const fullPath = path.join(resolvedPath, entry.name);
          let size: number | undefined;
          let modified: string | undefined;
          try {
            const s = fs.statSync(fullPath);
            size = entry.isFile() ? s.size : undefined;
            modified = s.mtime.toISOString();
          } catch {
            modified = undefined;
          }
          let isOmeZarr = false;
          if (entry.isDirectory()) {
            try {
              const zattrsFile = path.join(fullPath, ".zattrs");
              const zarrJsonFile = path.join(fullPath, "zarr.json");
              if (fs.existsSync(zattrsFile)) {
                const z = JSON.parse(fs.readFileSync(zattrsFile, "utf-8"));
                isOmeZarr = Array.isArray(z?.multiscales) && z.multiscales.length > 0;
              } else if (fs.existsSync(zarrJsonFile)) {
                const z = JSON.parse(fs.readFileSync(zarrJsonFile, "utf-8"));
                isOmeZarr =
                  (Array.isArray(z?.attributes?.multiscales) && z.attributes.multiscales.length > 0) ||
                  (Array.isArray(z?.attributes?.ome?.multiscales) && z.attributes.ome.multiscales.length > 0);
              }
            } catch { /* not an OME-Zarr */ }
          }
          return {
            name: entry.name,
            path: fullPath,
            isDirectory: entry.isDirectory(),
            isOmeZarr,
            size,
            modified,
          };
        })
        .sort((a, b) => {
          if (a.isDirectory && !b.isDirectory) return -1;
          if (!a.isDirectory && b.isDirectory) return 1;
          return a.name.localeCompare(b.name);
        });

      res.json({
        currentPath: resolvedPath,
        parentPath: path.dirname(resolvedPath),
        items,
      });
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to list files" });
    }
  });

  // ── S3 / HTTP object-store listing ─────────────────────────────────────────
  // Parses a ListObjectsV2 XML response and returns prefixes + file entries.
  function parseS3Xml(xml: string): { prefixes: string[]; contents: { key: string; size: number }[] } {
    const prefixes: string[] = [];
    for (const m of xml.matchAll(/<CommonPrefixes>\s*<Prefix>([^<]+)<\/Prefix>/gs))
      prefixes.push(m[1]);
    const contents: { key: string; size: number }[] = [];
    for (const m of xml.matchAll(/<Contents>([\s\S]*?)<\/Contents>/g)) {
      const key = m[1].match(/<Key>([^<]+)<\/Key>/)?.[1];
      const size = parseInt(m[1].match(/<Size>(\d+)<\/Size>/)?.[1] ?? "0", 10);
      if (key && !key.endsWith("/")) contents.push({ key, size });
    }
    return { prefixes, contents };
  }

  app.get("/api/s3/list", async (req, res) => {
    const rawUrl = (req.query.url as string) || "";
    if (!rawUrl) return res.status(400).json({ message: "url parameter required" });
    try {
      const parsed = new URL(rawUrl);
      const endpoint = parsed.origin; // e.g. https://s3.embl.de
      const pathParts = parsed.pathname.replace(/^\//, "").split("/").filter(Boolean);
      if (pathParts.length === 0)
        return res.status(400).json({ message: "URL must include at least a bucket name" });

      const bucket = pathParts[0];
      const prefixParts = pathParts.slice(1);
      const prefix = prefixParts.length > 0 ? prefixParts.join("/") + "/" : "";

      const listUrl = new URL(`${endpoint}/${bucket}`);
      listUrl.searchParams.set("list-type", "2");
      listUrl.searchParams.set("delimiter", "/");
      if (prefix) listUrl.searchParams.set("prefix", prefix);

      const listRes = await fetch(listUrl.toString(), { signal: AbortSignal.timeout(15000) });
      if (!listRes.ok)
        return res.status(listRes.status).json({ message: `S3 error: ${listRes.status} ${listRes.statusText}` });

      const xml = await listRes.text();
      const { prefixes: subPrefixes, contents } = parseS3Xml(xml);

      const currentPath = `${endpoint}/${bucket}/${prefix}`;
      const prefixTrimmed = prefix.replace(/\/$/, "");
      const parentPrefix = prefixTrimmed.includes("/")
        ? prefixTrimmed.split("/").slice(0, -1).join("/") + "/"
        : "";
      const parentPath = prefixTrimmed
        ? `${endpoint}/${bucket}/${parentPrefix}`
        : currentPath;

      // ── OME-Zarr detection ─────────────────────────────────────────────────
      // 1. Name heuristic (free): ends with .zarr / .ome.zarr
      // 2. Metadata probe (network): fetch .zattrs or zarr.json and check for
      //    a "multiscales" key — handles arbitrarily named zarr stores.
      //    Probes run with a concurrency cap of 3 so they don't saturate the
      //    S3 connection pool while the viewer is also fetching chunks.
      const probeOmeZarr = async (itemPath: string): Promise<boolean> => {
        // itemPath ends with "/"; try .zattrs and zarr.json at that level
        for (const fname of [".zattrs", "zarr.json"]) {
          try {
            const r = await fetch(`${itemPath}${fname}`, {
              signal: AbortSignal.timeout(2000),
            });
            if (!r.ok) continue;
            const j = await r.json().catch(() => null);
            if (!j) continue;
            // zarr v2 (.zattrs):          { multiscales: [...] }
            // zarr v3 early draft:         { attributes: { multiscales: [...] } }
            // OME-NGFF 0.5 / zarr v3:     { attributes: { ome: { multiscales: [...] } } }
            if (Array.isArray(j?.multiscales) && j.multiscales.length > 0) return true;
            if (Array.isArray(j?.attributes?.multiscales) && j.attributes.multiscales.length > 0) return true;
            if (Array.isArray(j?.attributes?.ome?.multiscales) && j.attributes.ome.multiscales.length > 0) return true;
          } catch { /* timeout or network error */ }
        }
        return false;
      }

      // Worker-pool concurrency limiter (no external deps)
      const mapWithLimit = async <T, R>(
        items: T[],
        fn: (item: T) => Promise<R>,
        limit: number,
      ): Promise<R[]> => {
        const results: R[] = new Array(items.length);
        let idx = 0;
        const worker = async () => {
          while (idx < items.length) {
            const i = idx++;
            results[i] = await fn(items[i]);
          }
        };
        await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
        return results;
      };

      const candidates = subPrefixes
        .filter((p) => p !== prefix)
        .map((p) => ({
          name: p.slice(prefix.length).replace(/\/$/, ""),
          path: `${endpoint}/${bucket}/${p}`,
        }));

      // Probe every folder — name convention alone is unreliable.
      const dirItems = await mapWithLimit(
        candidates,
        async ({ name, path: itemPath }) => ({
          name,
          path: itemPath,
          isDirectory: true,
          isOmeZarr: await probeOmeZarr(itemPath),
        }),
        3, // max 3 concurrent S3 metadata probes
      );

      const fileItems = contents.map(({ key, size }) => ({
        name: key.split("/").pop() || key,
        path: `${endpoint}/${bucket}/${key}`,
        isDirectory: false,
        isOmeZarr: false,
        size,
      }));

      const items = [
        ...dirItems.sort((a, b) => a.name.localeCompare(b.name)),
        ...fileItems.sort((a, b) => a.name.localeCompare(b.name)),
      ];

      res.json({ currentPath, parentPath, items });
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to list S3 path" });
    }
  });

  app.get("/api/system/memory", (_req, res) => {
    res.json({ totalMb: Math.round(os.totalmem() / (1024 * 1024)) });
  });

  app.get("/api/zarr/port", (_req, res) => {
    res.json({ port: PLANE_SERVER_PORT, url: `http://127.0.0.1:${PLANE_SERVER_PORT}` });
  });

  app.get("/api/zarr/plane", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/plane?${queryString}`, res);
  });

  app.get("/api/zarr/info", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/info?${queryString}`, res);
  });

  app.get("/api/zarr/tile_grid", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/tile_grid?${queryString}`, res);
  });

  app.get("/api/zarr/tile", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/tile?${queryString}`, res);
  });

  app.get("/api/zarr/channel_minmax", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/channel_minmax?${queryString}`, res);
  });

  app.get("/api/zarr/metadata", async (req, res) => {
    const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
    await proxyToPlaneServer(`/metadata?${queryString}`, res);
  });

  app.get("/api/zarr/metadata_unused", async (req, res) => {
    const zarrPath = req.query.path as string;
    if (!zarrPath) {
      return res.status(400).json({ message: "Path is required" });
    }

    const isRemote = /^https?:\/\/|^s3:\/\/|^gs:\/\//.test(zarrPath);

    // ── helpers ────────────────────────────────────────────────────────
    /** Fetch a JSON file at a URL; returns null on any error. */
    async function fetchRemoteJson(url: string): Promise<any | null> {
      try {
        const r = await fetch(url);
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    }

    /** Join a base URL/path with a relative child (always uses /). */
    function joinPath(base: string, child: string): string {
      return `${base.replace(/\/+$/, "")}/${child}`;
    }

    /** Extract array info (dtype, shape, chunks, compression) from .zarray or zarr.json JSON. */
    function parseArrayInfo(obj: any, isV3: boolean) {
      if (isV3) {
        const skip = new Set(["bytes", "transpose", "crc32c", "zstd_checksum"]);
        const findCompressor = (list: any[]): any => {
          for (const c of list) {
            if (c.name === "sharding_indexed") {
              const found = findCompressor(c.configuration?.codecs || []);
              if (found) return found;
            } else if (c.name && !skip.has(c.name)) {
              return c;
            }
          }
          return null;
        };
        const compCodec = findCompressor(obj.codecs || []);
        return {
          dataType: obj.data_type || "unknown",
          shape: obj.shape || [],
          chunks: obj.chunk_grid?.configuration?.chunk_shape || [],
          compression: { name: compCodec?.name || "unknown", params: compCodec?.configuration || {} },
        };
      }
      const comp = obj.compressor;
      const compName: string = comp?.id || comp?.codec || (comp === null ? "none" : "unknown");
      const compParams: Record<string, unknown> = comp
        ? Object.fromEntries(Object.entries(comp).filter(([k]) => k !== "id" && k !== "codec"))
        : {};
      return {
        dataType: obj.dtype || "unknown",
        shape: obj.shape || [],
        chunks: obj.chunks || [],
        compression: { name: compName, params: compParams },
      };
    }
    // ──────────────────────────────────────────────────────────────────

    let multiscales: any = null;
    let omero: any = null;
    let dataType = "unknown";
    let shape: number[] = [];
    let chunks: number[] = [];
    let compression: { name: string; params: Record<string, unknown> } = { name: "unknown", params: {} };
    let pyramidLayers: any[] = [];
    let baseName = "";

    if (isRemote) {
      // ── Remote store: delegate to the plane server's /info endpoint ──
      // Pyramid (Python) handles zarr v2/v3 and all remote protocols correctly.
      baseName = zarrPath.replace(/\/+$/, "").split("/").pop() || zarrPath;
      try {
        const infoUrl = `http://127.0.0.1:${PLANE_SERVER_PORT}/info?path=${encodeURIComponent(zarrPath)}`;
        const infoRes = await fetch(infoUrl);
        if (infoRes.ok) {
          const info = await infoRes.json() as any;
          // info shape: { axes, shape, resolutionPaths, levelsInfo, channels, dimSizes, dtype }
          const axisNames: string[] = (info.axes || "").split("").filter(Boolean);
          multiscales = {
            datasets: (info.resolutionPaths || []).map((p: string) => ({ path: p })),
          };
          // Build synthetic axes array compatible with the metadata response
          const axisTypes: Record<string, string> = { t: "time", c: "channel", z: "space", y: "space", x: "space" };
          const axisUnits: Record<string, string> = { t: "second", z: "micrometer", y: "micrometer", x: "micrometer" };
          const axesArray = axisNames.map((n: string) => ({
            name: n, type: axisTypes[n] || "space", unit: axisUnits[n],
          }));
          shape = info.shape || [];
          dataType = info.dtype || "unknown";
          if (info.levelsInfo?.length > 0) {
            chunks = info.levelsInfo[0].chunks || [];
            if (info.levelsInfo[0].compression) {
              compression = info.levelsInfo[0].compression;
            }
            pyramidLayers = info.levelsInfo.map((l: any, i: number) => ({
              level: i, shape: l.shape || [], chunks: l.chunks || [], scales: [],
            }));
          }
          // Build channel list from info
          const infoChannels = info.channels || [];
          if (infoChannels.length > 0) {
            omero = {
              channels: infoChannels.map((ch: any) => ({
                label: ch.label,
                color: ch.color || "FFFFFF",
                active: ch.visible !== false,
                window: ch.window || { min: 0, max: 65535, start: 0, end: 65535 },
              })),
            };
          }
          // Patch multiscales with a synthetic axes array so the metadata response uses real axes
          multiscales.axes = axesArray;
        }
      } catch { /* fall through to defaults */ }
    } else {
      // ── Local store: use the filesystem ───────────────────────────
      const resolvedPath = path.resolve(zarrPath);
      if (!fs.existsSync(resolvedPath)) {
        return res.status(404).json({ message: "Path not found" });
      }
      baseName = path.basename(resolvedPath);

      const zattrsPath = path.join(resolvedPath, ".zattrs");
      const rootZarrJsonPath = path.join(resolvedPath, "zarr.json");

      if (fs.existsSync(zattrsPath)) {
        try {
          const zattrs = JSON.parse(fs.readFileSync(zattrsPath, "utf-8"));
          multiscales = zattrs.multiscales?.[0] || null;
          omero = zattrs.omero || null;
        } catch {}
      }

      if (!multiscales && fs.existsSync(rootZarrJsonPath)) {
        try {
          const rootZarrJson = JSON.parse(fs.readFileSync(rootZarrJsonPath, "utf-8"));
          const attrs = rootZarrJson.attributes || rootZarrJson;
          multiscales = attrs.multiscales?.[0] || null;
          omero = omero || attrs.omero || null;
        } catch {}
      }

      if (!multiscales) {
        const subdirs = fs.readdirSync(resolvedPath, { withFileTypes: true })
          .filter(d => d.isDirectory() && /^\d+$/.test(d.name))
          .sort((a, b) => parseInt(a.name) - parseInt(b.name));
        if (subdirs.length > 0) {
          multiscales = { datasets: subdirs.map(d => ({ path: d.name })) };
        }
      }

      if (multiscales?.datasets?.length > 0) {
        const firstDataset = multiscales.datasets[0].path || "0";
        const arrayPath = path.join(resolvedPath, firstDataset);
        const zarrayFile = path.join(arrayPath, ".zarray");
        const zarrJsonFile = path.join(arrayPath, "zarr.json");

        if (fs.existsSync(zarrayFile)) {
          try {
            const zarray = JSON.parse(fs.readFileSync(zarrayFile, "utf-8"));
            ({ dataType, shape, chunks, compression } = parseArrayInfo(zarray, false));
          } catch {}
        } else if (fs.existsSync(zarrJsonFile)) {
          try {
            const zarrJson = JSON.parse(fs.readFileSync(zarrJsonFile, "utf-8"));
            ({ dataType, shape, chunks, compression } = parseArrayInfo(zarrJson, true));
          } catch {}
        }

        pyramidLayers = multiscales.datasets.map((ds: any, i: number) => {
          const dsPath = ds.path || String(i);
          const layerArrayPath = path.join(resolvedPath, dsPath);
          let layerShape: number[] = [];
          let layerChunks: number[] = [];
          const zarrayFile = path.join(layerArrayPath, ".zarray");
          const zarrJsonFile = path.join(layerArrayPath, "zarr.json");
          if (fs.existsSync(zarrayFile)) {
            try {
              const zarray = JSON.parse(fs.readFileSync(zarrayFile, "utf-8"));
              layerShape = zarray.shape || [];
              layerChunks = zarray.chunks || [];
            } catch {}
          } else if (fs.existsSync(zarrJsonFile)) {
            try {
              const zarrJson = JSON.parse(fs.readFileSync(zarrJsonFile, "utf-8"));
              layerShape = zarrJson.shape || [];
              layerChunks = zarrJson.chunk_grid?.configuration?.chunk_shape || [];
            } catch {}
          }
          return { level: i, shape: layerShape, chunks: layerChunks, scales: ds.coordinateTransformations?.[0]?.scale || [] };
        });
      }
    }

    const metadata = {
      name: baseName,
      ngffVersion: multiscales?.version || "0.4",
      resolutionLevels: multiscales?.datasets?.length || pyramidLayers.length || 1,
      dataType,
      axes: multiscales?.axes || [
        { name: "t", type: "time", unit: "second" },
        { name: "c", type: "channel" },
        { name: "z", type: "space", unit: "micrometer" },
        { name: "y", type: "space", unit: "micrometer" },
        { name: "x", type: "space", unit: "micrometer" },
      ],
      shape,
      chunks,
      compression,
      channels: omero?.channels?.map((ch: any, i: number) => ({
        index: i,
        label: ch.label || `Channel ${i}`,
        color: ch.color ? `#${ch.color}` : ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"][i % 4],
        visible: ch.active !== false,
        intensityMin: ch.window?.start || 0,
        intensityMax: ch.window?.end || 65535,
        window: {
          start: ch.window?.start || 0,
          end: ch.window?.end || 65535,
          min: ch.window?.min || 0,
          max: ch.window?.max || 65535,
        },
      })) || [
        { index: 0, label: "Channel 0", color: "#FF0000", visible: true, intensityMin: 0, intensityMax: 65535, window: { start: 0, end: 65535, min: 0, max: 65535 } },
      ],
      pyramidLayers,
      scales: multiscales?.datasets?.[0]?.coordinateTransformations?.[0]?.scale?.map((s: number, i: number) => ({
        axis: multiscales.axes[i]?.name || `dim${i}`,
        value: s,
        unit: multiscales.axes[i]?.unit || "",
      })) || [],
    };

    res.json(metadata);
  });

  app.post("/api/zarr/metadata/update", async (req, res) => {
    const { path: zarrPath, scales } = req.body;
    if (!zarrPath || !scales || !Array.isArray(scales)) {
      return res.status(400).json({ message: "Path and scales array are required" });
    }
    try {
      const response = await fetch(`http://127.0.0.1:${PLANE_SERVER_PORT}/update_scales`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: zarrPath, scales }),
      });
      const result = await response.json() as any;
      if (!response.ok) {
        return res.status(500).json({ message: result.error || "Failed to update metadata" });
      }
      res.json({ success: true, message: "Metadata updated successfully" });
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to update metadata" });
    }
  });

  app.post("/api/zarr/channels/update", async (req, res) => {
    const { path: zarrPath, channels } = req.body;
    if (!zarrPath || !channels || !Array.isArray(channels)) {
      return res.status(400).json({ message: "Path and channels array are required" });
    }
    try {
      const zattrsPath = path.join(zarrPath, ".zattrs");
      let zattrs: any = {};
      try {
        const raw = fs.readFileSync(zattrsPath, "utf-8");
        zattrs = JSON.parse(raw);
      } catch {
        const zarrJsonPath = path.join(zarrPath, "zarr.json");
        try {
          const raw = fs.readFileSync(zarrJsonPath, "utf-8");
          zattrs = JSON.parse(raw);
        } catch {
          return res.status(404).json({ message: "No .zattrs or zarr.json found" });
        }
      }

      const omero = zattrs.omero || {};
      const existingChannels = omero.channels || [];

      const updatedChannels = channels.map((ch: any) => {
        const existing = existingChannels[ch.index] || {};
        const hexToRgb = (hex: string) => {
          const h = hex.replace("#", "");
          return [parseInt(h.substring(0, 2), 16), parseInt(h.substring(2, 4), 16), parseInt(h.substring(4, 6), 16)];
        };
        return {
          ...existing,
          label: ch.label,
          color: ch.color.replace("#", ""),
          active: true,
          window: {
            ...(existing.window || {}),
            start: ch.window.start,
            end: ch.window.end,
            min: ch.window.min,
            max: ch.window.max,
          },
        };
      });

      zattrs.omero = { ...omero, channels: updatedChannels };

      const isZarr3 = fs.existsSync(path.join(zarrPath, "zarr.json"));
      if (isZarr3) {
        const zarrJsonPath = path.join(zarrPath, "zarr.json");
        const zarrJson = JSON.parse(fs.readFileSync(zarrJsonPath, "utf-8"));
        zarrJson.attributes = { ...zarrJson.attributes, omero: zattrs.omero };
        fs.writeFileSync(zarrJsonPath, JSON.stringify(zarrJson, null, 2));
      } else {
        fs.writeFileSync(zattrsPath, JSON.stringify(zattrs, null, 2));
      }

      await invalidatePlaneServerCache(zarrPath);
      res.json({ success: true, message: "Channel metadata updated" });
    } catch (err: any) {
      res.status(500).json({ message: err.message || "Failed to update channel metadata" });
    }
  });

  return httpServer;
}

function runConversion(
  jobId: string,
  config: Record<string, any>,
  broadcast: (jobId: string, msg: Record<string, any>) => void,
) {
  const scriptPath = path.join(SCRIPTS_DIR, "run_conversion.py");
  const configJson = JSON.stringify(config);

  const proc = spawn(PYTHON_EXECUTABLE, [scriptPath, configJson], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env },
  });

  activeProcesses.set(jobId, proc);

  const rl = readline.createInterface({ input: proc.stdout! });

  rl.on("line", async (line) => {
    try {
      const data = JSON.parse(line);

      if (data.type === "log") {
        const message = data.message || "";
        await storage.addJobLog(jobId, message);
        broadcast(jobId, { type: "log", message });
      } else if (data.type === "complete") {
        await storage.updateJob(jobId, { status: "completed", progress: 100, completedAt: new Date().toISOString() });
        broadcast(jobId, { type: "complete" });
      } else if (data.type === "error") {
        const message = data.message || "Unknown error";
        await storage.addJobLog(jobId, `ERROR: ${message}`);
        if (data.traceback) {
          await storage.addJobLog(jobId, data.traceback);
        }
        await storage.updateJob(jobId, { status: "failed" });
        broadcast(jobId, { type: "error", message });
      } else if (data.type === "cancelled") {
        await storage.updateJob(jobId, { status: "cancelled" });
        broadcast(jobId, { type: "error", message: "Conversion cancelled" });
      }
    } catch {
      if (line.trim()) {
        await storage.addJobLog(jobId, line);
        broadcast(jobId, { type: "log", message: line });
      }
    }
  });

  let stderrBuffer = "";
  proc.stderr?.on("data", (chunk) => {
    stderrBuffer += chunk.toString();
    const lines = stderrBuffer.split("\n");
    stderrBuffer = lines.pop() || "";
    for (const errLine of lines) {
      if (errLine.trim()) {
        storage.addJobLog(jobId, errLine);
        broadcast(jobId, { type: "log", message: errLine });
      }
    }
  });

  proc.on("close", async (code) => {
    activeProcesses.delete(jobId);

    if (stderrBuffer.trim()) {
      await storage.addJobLog(jobId, stderrBuffer);
      broadcast(jobId, { type: "log", message: stderrBuffer });
    }

    const job = await storage.getJob(jobId);
    if (job && job.status === "running") {
      if (code === 0) {
        await storage.updateJob(jobId, { status: "completed", progress: 100, completedAt: new Date().toISOString() });
        broadcast(jobId, { type: "complete" });
      } else {
        await storage.updateJob(jobId, { status: "failed" });
        broadcast(jobId, { type: "error", message: `Conversion process exited with code ${code}` });
      }
    }
  });

  proc.on("error", async (err) => {
    activeProcesses.delete(jobId);
    await storage.updateJob(jobId, { status: "failed" });
    broadcast(jobId, { type: "error", message: `Failed to start conversion process: ${err.message}` });
  });
}
