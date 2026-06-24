export type ModelInfo = {
  params: number;
  n_frames: number;
  img_size: number;
  device: string;
  epoch?: number;
  auc?: number;
  acc?: number;
  error?: string;
}

export type PredictionResult = {
  status: string;
  label: string;
  asd_prob: number;
  td_prob: number;
  confidence: number;
  top_frames: number[];
  frame_weights: number[];
  frame_energies: number[];
  thumbs: string[];
  video_meta: {
    fps: number;
    frames: number;
    width: number;
    height: number;
    duration: number;
    error?: string;
  };
  timing: {
    video_read_ms: number;
    frame_extract_ms: number;
    cnn_encode_ms: number;
    transformer_attn_ms: number;
    lstm_attn_ms: number;
    total_ms: number;
  };
  error?: string;
}

const API_BASE = '/api';

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model_info`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: ${res.statusText}`);
  }
  return res.json();
}

export async function runPrediction(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('video', file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errMsg = `Server error (${res.status})`;
    try { 
      const errData = await res.json(); 
      errMsg = errData.error || errMsg; 
    } catch (_) {
      // Ignore
    }
    throw new Error(errMsg);
  }

  const data = await res.json();
  if (data.error) {
    throw new Error(data.error);
  }

  return data;
}
