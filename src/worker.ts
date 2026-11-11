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
  matting2: 'onnx-community/ormbg-ONNX/resolve/main/onnx/model_fp16.onnx',
  matting3: 'onnx-community/ISNet-ONNX/resolve/main/onnx/model_fp16.onnx',
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

/** 没下完的模型留在内存里：连接断了接着下，整次失败后再点"开始融合"也是接着下 */
interface Partial {
  chunks: Uint8Array[];
  loaded: number;
  total: number;
}
const partials = new Map<string, Partial>();

/** 连续这么多次重连都没有新数据进来，才算下载失败 */
const MAX_IDLE_RETRIES = 6;
/** 这么久收不到任何数据就当作连接卡死，主动断开重连 */
const STALL_MS = 30_000;

class FatalDownloadError extends Error {}

/**
 * 带断线重试 + 断点续传（HTTP Range）的下载。
 * 模型有一两百 MB，慢网络下中途断线很常见，不能每次都从头来。
 */
async function downloadWithResume(url: string, name: ModelName, id: number): Promise<Uint8Array<ArrayBuffer>> {
  let part = partials.get(url);
  if (!part) {
    part = { chunks: [], loaded: 0, total: 0 };
    partials.set(url, part);
  }
  const report = () => post({ type: 'download', id, model: name, loaded: part.loaded, total: part.total });
  let idleRetries = 0;
  let lastError = '';

  for (;;) {
    const before = part.loaded;
    const ctrl = new AbortController();
    let stall: ReturnType<typeof setTimeout> | undefined;
    const kick = () => {
      clearTimeout(stall);
      stall = setTimeout(() => ctrl.abort(), STALL_MS);
    };
    try {
      kick();
      const res = await fetch(url, {
        signal: ctrl.signal,
        headers: part.loaded > 0 ? { Range: `bytes=${part.loaded}-` } : undefined,
      });
      if (res.status === 416 && part.loaded > 0) {
        // 续传位置不被接受（文件变了？）：丢掉已有部分，从头来
        part.chunks = [];
        part.loaded = 0;
        part.total = 0;
        continue;
      }
      if (!res.ok || !res.body) {
        const msg = `模型 ${MODEL_FILES[name]} 下载失败（HTTP ${res.status}）。`;
        // 4xx 重试也没用（408 超时 / 429 限流除外）
        if (res.status >= 400 && res.status < 500 && res.status !== 408 && res.status !== 429) {
          throw new FatalDownloadError(msg);
        }
        throw new Error(msg);
      }
      // dev server 对不存在的文件会回退到 index.html，这里顺手识别出来
      if ((res.headers.get('content-type') ?? '').includes('text/html')) {
        throw new FatalDownloadError(`找不到模型 ${MODEL_FILES[name]}。请先运行 npm run models 下载模型。`);
      }
      if (res.status === 206) {
        // Content-Range: bytes 起-止/总长
        const m = /bytes (\d+)-\d+\/(\d+)/.exec(res.headers.get('content-range') ?? '');
        if (!m || Number(m[1]) !== part.loaded) {
          throw new Error('服务器返回的续传位置不对');
        }
        part.total = Number(m[2]);
      } else {
        // 200：服务器不支持续传（或这是第一次请求），只能从头收
        part.chunks = [];
        part.loaded = 0;
        part.total = Number(res.headers.get('content-length') ?? 0);
      }

      const reader = res.body.getReader();
      let lastReport = part.loaded;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        kick();
        part.chunks.push(value);
        part.loaded += value.length;
        if (part.loaded - lastReport > 5e5) {
          lastReport = part.loaded;
          report();
        }
      }
      if (part.total && part.loaded < part.total) throw new Error('连接提前断开');
      break; // 下完了
    } catch (err) {
      if (err instanceof FatalDownloadError) {
        partials.delete(url);
        throw err;
      }
      lastError = ctrl.signal.aborted ? '长时间收不到数据' : err instanceof Error ? err.message : String(err);
      // 只要这一轮有新数据进来，就不算"原地踏步"
      idleRetries = part.loaded > before ? 0 : idleRetries + 1;
      if (idleRetries >= MAX_IDLE_RETRIES) {
        const mb = (part.loaded / 1e6).toFixed(0);
        throw new Error(
          `模型 ${MODEL_FILES[name]} 下载中断（${lastError}），已下载 ${mb} MB。请检查网络后再点"开始融合"，会从断点继续。`,
        );
      }
      post({ type: 'retry', id, model: name, loaded: part.loaded, total: part.total });
      await new Promise((r) => setTimeout(r, Math.min(15_000, 1000 * 2 ** idleRetries)));
    } finally {
      clearTimeout(stall);
    }
  }

  report();
  const buf = new Uint8Array(part.loaded);
  let off = 0;
  for (const c of part.chunks) {
    buf.set(c, off);
    off += c.length;
  }
  partials.delete(url);
  return buf;
}

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

  const buf = await downloadWithResume(url, name, id);
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

// 有 WebGPU 时这几个模型放到显卡上跑（实测比多线程 WASM 快 2-3 倍，结果一致）；
// 兜底用的语义特征模型输入尺寸不固定且很少用到，留在 WASM 上。
const GPU_MODELS: ReadonlySet<ModelName> = new Set<ModelName>(['matting', 'matting2', 'matting3', 'extractor', 'matcher']);
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
      } catch (err) {
        console.warn(`[seichi] ${name} 无法在 WebGPU 上初始化，改用 WASM：`, err);
        return useWasm();
      }
      return {
        inputNames: gpu.inputNames,
        outputNames: gpu.outputNames,
        async run(feeds) {
          if (wasm) return (await wasm).run(feeds);
          try {
            return await gpu.run(feeds);
          } catch (err) {
            console.warn(`[seichi] ${name} 在 WebGPU 上推理失败，改用 WASM：`, err);
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
        // 两个补充抠图模型是半精度的，只在 WebGPU 上跑；没有 WebGPU 时跳过（少下约 180 MB，深色衣服可能抠不全）
        refineMatting: hasWebGPU,
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
