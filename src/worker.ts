/// <reference lib="webworker" />
// 所有推理都在这个 Worker 里完成：图片不离开浏览器。
import * as ort from 'onnxruntime-web/webgpu';
import { MODEL_FILES, runPipeline, type ModelName, type Stage } from './core/pipeline.ts';
import type { OrtLike, SessionLike } from './core/ort-types.ts';
import type { RunRequest, WorkerMessage } from './protocol.ts';

const BASE = import.meta.env.BASE_URL;
// 模型从哪里加载：
//  - 默认：本站 /models/（先 npm run models 下载）；
//  - VITE_MODEL_BASE=https://cdn.example.com/models/ ：自己的 CDN / 对象存储；
//  - VITE_MODEL_SOURCE=hf ：直接从 Hugging Face 拉取（GitHub Pages 这类放不下大文件的静态托管用这个），
//    可再用 VITE_HF_ENDPOINT 换成镜像站。
const MODEL_BASE: string = import.meta.env.VITE_MODEL_BASE ?? `${BASE}models/`;
const HF: string = import.meta.env.VITE_HF_ENDPOINT ?? 'https://huggingface.co';
const HF_PATHS: Record<ModelName, string> = {
  matting: 'skytnt/anime-seg/resolve/main/isnetis.onnx',
  extractor: 'bukuroo/ALIKED-LightGlue-ONNX/resolve/main/aliked-n16rot-top2k-640.onnx',
  matcher: 'bukuroo/ALIKED-LightGlue-ONNX/resolve/main/lightglue_for_aliked.onnx',
  features: 'onnx-community/dinov2-with-registers-small/resolve/main/onnx/model.onnx',
};
const modelUrl = (name: ModelName): string =>
  import.meta.env.VITE_MODEL_SOURCE === 'hf'
    ? `${HF}/${HF_PATHS[name]}`
    : new URL(MODEL_BASE + MODEL_FILES[name], self.location.href).href;
const CACHE_NAME = 'seichi-models-v1';

// 只覆盖 .wasm 的地址；胶水代码（.mjs）用打包进来的那份，避免从 public/ 动态 import
ort.env.wasm.wasmPaths = {
  wasm: new URL(`${BASE}ort/ort-wasm-simd-threaded.asyncify.wasm`, self.location.href).href,
};
ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
ort.env.logLevel = 'error';

let currentStage: Stage = 'matting';
const post = (msg: WorkerMessage, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

async function fetchModel(name: ModelName, id: number): Promise<ArrayBuffer> {
  const url = modelUrl(name);
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
  } catch {
    cache = null; // 无痕模式等场景下 Cache API 不可用：每次重新下载
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`模型 ${MODEL_FILES[name]} 下载失败（HTTP ${res.status}）。`);
  }
  // dev server 对不存在的文件会回退到 index.html，这里顺手识别出来
  if ((res.headers.get('content-type') ?? '').includes('text/html')) {
    throw new Error(`找不到模型 ${MODEL_FILES[name]}。请先运行 npm run models 下载模型。`);
  }
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (loaded - lastReport > 2e6 || loaded === total) {
      lastReport = loaded;
      post({ type: 'download', id, model: name, loaded, total });
    }
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  // 下载结束后把提示换回当前阶段（接下来是模型初始化 + 推理）
  post({ type: 'stage', id, stage: currentStage });
  try {
    await cache?.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
  } catch {
    // 配额不足等：不缓存也不影响本次运行
  }
  return buf.buffer;
}

const sessions = new Map<ModelName, Promise<SessionLike>>();
/** 实际用上的推理后端，展示给用户看 */
const backends = new Map<ModelName, string>();

// 有 WebGPU 时这三个模型放到显卡上跑（实测比多线程 WASM 快 2-3 倍，结果一致）；
// 兜底用的语义特征模型输入尺寸不固定且很少用到，留在 WASM 上。
const GPU_MODELS: ReadonlySet<ModelName> = new Set<ModelName>(['matting', 'extractor', 'matcher']);
const hasWebGPU = 'gpu' in navigator;

async function createSession(bytes: ArrayBuffer, provider: 'webgpu' | 'wasm'): Promise<SessionLike> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
  });
  return session as unknown as SessionLike;
}

function getSession(name: ModelName, id: number): Promise<SessionLike> {
  let p = sessions.get(name);
  if (!p) {
    p = (async () => {
      const bytes = await fetchModel(name, id);
      if (!(hasWebGPU && GPU_MODELS.has(name))) {
        backends.set(name, 'wasm');
        return createSession(bytes, 'wasm');
      }
      // WebGPU 可能在建会话或第一次推理时才报错（驱动/算子不支持），任何一步失败都退回 WASM
      let wasm: Promise<SessionLike> | null = null;
      const useWasm = () => {
        backends.set(name, 'wasm');
        return (wasm ??= createSession(bytes, 'wasm'));
      };
      let gpu: SessionLike;
      try {
        gpu = await createSession(bytes, 'webgpu');
        backends.set(name, 'webgpu');
      } catch {
        return useWasm();
      }
      return {
        inputNames: gpu.inputNames,
        outputNames: gpu.outputNames,
        async run(feeds) {
          if (wasm) return (await wasm).run(feeds);
          try {
            return await gpu.run(feeds);
          } catch {
            return (await useWasm()).run(feeds);
          }
        },
      } satisfies SessionLike;
    })();
    // 失败后允许重试
    p.catch(() => sessions.delete(name));
    sessions.set(name, p);
  }
  return p;
}

// 打包后，ONNX Runtime 的 WASM 线程（em-pthread）会重新加载这同一个文件。
// 那些线程的 onmessage 归 Emscripten 管，我们绝不能覆盖，否则推理会永远卡住。
const isOrtThread = (self.name ?? '').startsWith('em-pthread');

const handleRun = async (ev: MessageEvent<RunRequest>) => {
  const { id, anime, real } = ev.data;
  const t0 = performance.now();
  const marks: string[] = [];
  let last = t0;
  try {
    const r = await runPipeline(
      {
        ort: ort as unknown as OrtLike,
        getSession: (name) => getSession(name, id),
        onStage: (stage) => {
          const now = performance.now();
          marks.push(`${stage}@${Math.round(now - last)}ms`);
          last = now;
          currentStage = stage;
          post({ type: 'stage', id, stage });
        },
      },
      anime,
      real,
    );
    console.debug('[seichi] 各阶段开始前耗时', marks.join(' → '));
    const inl = r.align.matches.filter((m) => m.inlier).slice(0, 500);
    const points = new Float32Array(inl.length * 2);
    inl.forEach((m, i) => {
      points[i * 2] = m.bx;
      points[i * 2 + 1] = m.by;
    });
    post(
      {
        type: 'result',
        id,
        result: {
          alpha: r.alpha,
          bounds: r.bounds,
          level: r.level,
          inliers: r.align.inliers,
          matches: r.align.matches.length,
          score: r.align.score,
          placement: r.placement,
          framing: r.framing,
          points,
          backend:
            backends.get('matting') === 'webgpu'
              ? `WebGPU + WASM × ${ort.env.wasm.numThreads} 线程`
              : `WASM × ${ort.env.wasm.numThreads} 线程`,
          elapsedMs: performance.now() - t0,
        },
      },
      [r.alpha.buffer, points.buffer],
    );
  } catch (err) {
    post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
  }
};

if (!isOrtThread) self.onmessage = handleRun;
