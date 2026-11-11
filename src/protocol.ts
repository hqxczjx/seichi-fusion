// 主线程 ↔ Worker 的消息格式
import type { Placement } from './core/align.ts';
import type { Framing, MatchLevel, ModelName, Stage } from './core/pipeline.ts';

export interface RawImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface RunRequest {
  type: 'run';
  id: number;
  anime: RawImage;
  real: RawImage;
}

export interface ResultPayload {
  alpha: Float32Array;
  bounds: { x: number; y: number; w: number; h: number; coverage: number } | null;
  level: MatchLevel;
  inliers: number;
  matches: number;
  score: number;
  placement: Placement;
  framing: Framing;
  /** 内点在实景图上的坐标 [x0,y0,x1,y1,...]，仅用于可视化 */
  points: Float32Array;
  backend: string;
  elapsedMs: number;
}

export type WorkerMessage =
  | { type: 'stage'; id: number; stage: Stage }
  | { type: 'download'; id: number; model: ModelName; loaded: number; total: number }
  | { type: 'result'; id: number; result: ResultPayload }
  | { type: 'error'; id: number; message: string };
