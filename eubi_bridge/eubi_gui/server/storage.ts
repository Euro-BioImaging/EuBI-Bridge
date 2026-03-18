import { randomUUID } from "crypto";

export interface ConversionJobRecord {
  id: string;
  inputPath: string;
  outputPath: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  config: Record<string, any>;
  logs: string[];
  startedAt?: string;
  completedAt?: string;
}

export interface IStorage {
  createJob(config: Record<string, any>): Promise<ConversionJobRecord>;
  getJob(id: string): Promise<ConversionJobRecord | undefined>;
  updateJob(id: string, updates: Partial<ConversionJobRecord>): Promise<ConversionJobRecord | undefined>;
  getAllJobs(): Promise<ConversionJobRecord[]>;
  addJobLog(id: string, message: string): Promise<void>;
}

export class MemStorage implements IStorage {
  private jobs: Map<string, ConversionJobRecord>;

  constructor() {
    this.jobs = new Map();
  }

  async createJob(config: Record<string, any>): Promise<ConversionJobRecord> {
    const id = randomUUID();
    const job: ConversionJobRecord = {
      id,
      inputPath: config.inputPath || "",
      outputPath: config.outputPath || "",
      status: "pending",
      progress: 0,
      config,
      logs: [],
      startedAt: new Date().toISOString(),
    };
    this.jobs.set(id, job);
    return job;
  }

  async getJob(id: string): Promise<ConversionJobRecord | undefined> {
    return this.jobs.get(id);
  }

  async updateJob(id: string, updates: Partial<ConversionJobRecord>): Promise<ConversionJobRecord | undefined> {
    const job = this.jobs.get(id);
    if (!job) return undefined;
    const updated = { ...job, ...updates };
    this.jobs.set(id, updated);
    return updated;
  }

  async getAllJobs(): Promise<ConversionJobRecord[]> {
    return Array.from(this.jobs.values());
  }

  async addJobLog(id: string, message: string): Promise<void> {
    const job = this.jobs.get(id);
    if (job) {
      job.logs.push(message);
    }
  }
}

export const storage = new MemStorage();
