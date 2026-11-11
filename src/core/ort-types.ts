// onnxruntime-web 与 onnxruntime-node 共有的最小接口，便于核心代码在两端复用。

export interface TensorLike {
  data: unknown;
  dims: readonly number[];
}

export interface SessionLike {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>;
}

export interface OrtLike {
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => TensorLike;
}
