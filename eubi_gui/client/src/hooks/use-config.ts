import { useMutation, useQuery } from "@tanstack/react-query";
import { useConversionStore } from "@/lib/conversion-store";

export interface ConfigPayload {
  cluster?: any;
  reader?: any;
  conversion?: any;
  downscaling?: any;
  metadata?: any;
}

export interface ConfigResponse extends ConfigPayload {
  _configPath?: string;
}

/** Fetch config from the server (loads the JSON file via Python). */
export function useLoadConfig(configPath?: string) {
  const url = configPath
    ? `/api/config?path=${encodeURIComponent(configPath)}`
    : "/api/config";
  return useQuery<ConfigResponse>({
    queryKey: ["config", configPath ?? "default"],
    queryFn: async () => {
      const res = await fetch(url);
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: res.statusText }));
        throw new Error(body.message || "Failed to load config");
      }
      return res.json();
    },
    enabled: false, // only on explicit refetch
    retry: false,
  });
}

/** Save the current store state to the JSON config file. */
export function useSaveConfig() {
  return useMutation<ConfigResponse, Error, { payload: ConfigPayload; configPath?: string }>({
    mutationFn: async ({ payload, configPath }) => {
      const url = configPath
        ? `/api/config?path=${encodeURIComponent(configPath)}`
        : "/api/config";
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: res.statusText }));
        throw new Error(body.message || "Failed to save config");
      }
      return res.json();
    },
  });
}

/** Reset config file to installation defaults. */
export function useResetConfig() {
  return useMutation<ConfigResponse, Error, { configPath?: string }>({
    mutationFn: async ({ configPath }) => {
      const url = configPath
        ? `/api/config/reset?path=${encodeURIComponent(configPath)}`
        : "/api/config/reset";
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: res.statusText }));
        throw new Error(body.message || "Failed to reset config");
      }
      return res.json();
    },
  });
}

/** Apply a ConfigResponse payload into the Zustand store. */
export function useApplyConfig() {
  const { setCluster, setReader, setConversion, setDownscaling, setMetadata } =
    useConversionStore();

  return (data: ConfigResponse) => {
    if (data.cluster)    setCluster(data.cluster);
    if (data.reader)     setReader(data.reader);
    if (data.conversion) setConversion(data.conversion);
    if (data.downscaling) setDownscaling(data.downscaling);
    if (data.metadata)   setMetadata(data.metadata);
  };
}
