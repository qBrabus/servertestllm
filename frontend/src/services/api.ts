import axios from "axios";

const apiClient = axios.create({
  baseURL: "/",
  timeout: 60000
});

const API_KEY_STORAGE = "unified-inference-api-key";

export const getStoredApiKey = () => localStorage.getItem(API_KEY_STORAGE);

export const setStoredApiKey = (value: string | null) => {
  if (value) {
    localStorage.setItem(API_KEY_STORAGE, value);
  } else {
    localStorage.removeItem(API_KEY_STORAGE);
  }
};

apiClient.interceptors.request.use((config) => {
  const key = getStoredApiKey();
  if (key) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${key}`;
  }
  return config;
});

export interface GPUStatus {
  id: number;
  name: string;
  memory_total: number;
  memory_used: number;
  load: number;
  temperature?: number;
}

export interface ModelRuntimeInfo {
  state: "idle" | "loading" | "ready" | "error";
  progress: number;
  status: string;
  details?: Record<string, unknown> | null;
  server?: Record<string, unknown> | null;
  downloaded: boolean;
  last_error?: string | null;
  updated_at?: string | null;
}

export interface ModelStatus {
  identifier: string;
  task: string;
  loaded: boolean;
  description: string;
  format: string;
  params?: Record<string, unknown>;
  runtime?: ModelRuntimeInfo | null;
}

export interface DashboardState {
  gpus: GPUStatus[];
  system: {
    cpu_percent: number;
    memory_percent: number;
  };
  models: Record<string, ModelStatus>;
}

export interface HuggingFaceTokenStatus {
  has_token: boolean;
}

export const fetchDashboard = async (): Promise<DashboardState> => {
  const { data } = await apiClient.get<DashboardState>("/api/admin/status");
  return data;
};

export const fetchHuggingFaceTokenStatus = async (): Promise<HuggingFaceTokenStatus> => {
  const { data } = await apiClient.get<HuggingFaceTokenStatus>("/api/admin/huggingface/token");
  return data;
};

export const updateHuggingFaceToken = async (
  token: string | null
): Promise<HuggingFaceTokenStatus> => {
  const payload = { token: token?.trim() ?? null };
  const { data } = await apiClient.post<HuggingFaceTokenStatus>(
    "/api/admin/huggingface/token",
    payload
  );
  return data;
};

export interface LoadModelPayload {
  key: string;
  gpuDeviceIds?: number[];
}

export const loadModel = async ({
  key,
  gpuDeviceIds
}: LoadModelPayload): Promise<Record<string, ModelStatus>> => {
  const payload = gpuDeviceIds && gpuDeviceIds.length > 0 ? { gpu_device_ids: gpuDeviceIds } : {};
  const { data } = await apiClient.post<{ models: Record<string, ModelStatus> }>(
    `/api/admin/models/${key}/load`,
    payload
  );
  return data.models;
};

export const downloadModel = async (key: string): Promise<Record<string, ModelStatus>> => {
  const { data } = await apiClient.post<{ models: Record<string, ModelStatus> }>(
    `/api/admin/models/${key}/download`
  );
  return data.models;
};

export const unloadModel = async (key: string): Promise<Record<string, ModelStatus>> => {
  const { data } = await apiClient.post<{ models: Record<string, ModelStatus> }>(
    `/api/admin/models/${key}/unload`
  );
  return data.models;
};

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionPayload {
  model?: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
}

export const createChatCompletion = async (payload: ChatCompletionPayload) => {
  const { data } = await apiClient.post("/v1/chat/completions", payload);
  return data;
};

export interface TranscriptionResult {
  text: string;
  sampling_rate: number;
}

export const transcribeAudio = async (file: File): Promise<TranscriptionResult> => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post<TranscriptionResult>("/api/audio/transcribe", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
};

export interface DiarizationResultSegment {
  speaker: string;
  start: number;
  end: number;
}

export const diarizeAudio = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post<{ segments: DiarizationResultSegment[] }>(
    "/api/diarization/process",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" }
    }
  );
  return data;
};
