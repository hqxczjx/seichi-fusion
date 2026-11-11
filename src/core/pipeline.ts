// 整条管线：抠人物 → 场景匹配/对齐 → 给出人物在实景图上的摆放。
// 不依赖 DOM，浏览器 Worker 与 Node 都能跑。
import { estimateAlignment, matchFeatures, placementFromMatrix, type AlignResult, type Placement } from './align.ts';
import { extractFeatures } from './features.ts';
import type { RGBAImage } from './image.ts';
import { matchKeypoints } from './matcher.ts';
import { alphaBounds, mergeAlphas, segmentCharacter, segmentPerson } from './matting.ts';
import type { OrtLike, SessionLike } from './ort-types.ts';

export type ModelName = 'matting' | 'matting2' | 'extractor' | 'matcher' | 'features';

export const MODEL_FILES: Record<ModelName, string> = {
  matting: 'anime-seg.onnx',
  matting2: 'ormbg-fp16.onnx',
  extractor: 'aliked-n16rot-640.onnx',
  matcher: 'lightglue-aliked.onnx',
  features: 'dinov2-reg-small.onnx',
};

export type Stage = 'matting' | 'matching' | 'fallback' | 'done';

/**
 * precise：关键点匹配成功，位置可信；
 * rough：  关键点不够（常见于画风与实景差异大），靠语义特征给出大致位置，建议手动微调；
 * none：   两张图场景不一致，不生成合成图。
 */
export type MatchLevel = 'precise' | 'rough' | 'none';

export interface PipelineResult {
  /** 与动漫图同尺寸的人物 alpha（0-1） */
  alpha: Float32Array;
  /** 人物包围盒（动漫图像素）；没抠到人物时为 null */
  bounds: { x: number; y: number; w: number; h: number; coverage: number } | null;
  level: MatchLevel;
  align: AlignResult;
  /** 人物锚点（包围盒底边中点）在实景图上的摆放 */
  placement: Placement;
  /** 按这个摆放，人物有多少落在实景照片的画面内 */
  framing: Framing;
}

export type Side = 'top' | 'bottom' | 'left' | 'right';

export interface Framing {
  /** 人物落在画面内的比例（按 alpha 加权，0-1）；没抠到人物时为 1 */
  visible: number;
  /** 人物从哪几条边超出了画面 */
  overflow: Side[];
  /**
   * 人物在动漫里本来就被画面底边截断（半身近景），而按场景对齐算出的位置大半落在实景画面之外时，
   * 改为让截断处贴着实景照片的底边摆放。这时为 true。
   */
  snappedToBottom: boolean;
}

/** 人物落在实景画面内的比例低于此值：视为"没拍到人物所在的位置" */
export const MIN_VISIBLE = 0.5;

/**
 * 两张图取景有偏移时，人物在实景里"该站的位置"可能没被拍进去。
 * 把人物的每个像素按 placement 映射到实景图上，统计有多少落在画面内。
 */
export function measureFraming(
  alpha: Float32Array,
  width: number,
  bounds: { x: number; y: number; w: number; h: number } | null,
  placement: Placement,
  realWidth: number,
  realHeight: number,
): Framing {
  if (!bounds) return { visible: 1, overflow: [], snappedToBottom: false };
  const anchorX = bounds.x + bounds.w / 2;
  const anchorY = bounds.y + bounds.h;
  const cos = Math.cos(placement.rotation) * placement.scale;
  const sin = Math.sin(placement.rotation) * placement.scale;
  // 抽样即可：包围盒内最多约 4 万个采样点
  const step = Math.max(1, Math.floor(Math.sqrt((bounds.w * bounds.h) / 40000)));
  // 超出不到 1% 画面尺寸的不算（抠图边缘的零星像素）
  const tolX = realWidth * 0.01;
  const tolY = realHeight * 0.01;
  let total = 0;
  let inside = 0;
  const out: Record<Side, number> = { top: 0, bottom: 0, left: 0, right: 0 };
  for (let y = bounds.y; y < bounds.y + bounds.h; y += step) {
    for (let x = bounds.x; x < bounds.x + bounds.w; x += step) {
      const a = alpha[y * width + x];
      if (a < 0.05) continue;
      const dx = x + 0.5 - anchorX;
      const dy = y + 0.5 - anchorY;
      const rx = placement.x + cos * dx - sin * dy;
      const ry = placement.y + sin * dx + cos * dy;
      total += a;
      if (rx >= 0 && rx < realWidth && ry >= 0 && ry < realHeight) inside += a;
      if (rx < -tolX) out.left += a;
      if (rx >= realWidth + tolX) out.right += a;
      if (ry < -tolY) out.top += a;
      if (ry >= realHeight + tolY) out.bottom += a;
    }
  }
  if (total === 0) return { visible: 1, overflow: [], snappedToBottom: false };
  const overflow = (Object.keys(out) as Side[]).filter((s) => out[s] / total > 0.01);
  return { visible: inside / total, overflow, snappedToBottom: false };
}

export interface PipelineEnv {
  ort: OrtLike;
  /** 按需加载模型（features 只有在兜底时才会被请求） */
  getSession(name: ModelName): Promise<SessionLike>;
  /** 是否启用第二个抠图模型来补回深色衣服（多下载约 90 MB，浏览器里只在有 WebGPU 时开启）。默认开启 */
  refineMatting?: boolean;
  onStage?(stage: Stage): void;
}

export async function runPipeline(env: PipelineEnv, anime: RGBAImage, real: RGBAImage): Promise<PipelineResult> {
  const { ort } = env;

  env.onStage?.('matting');
  let alpha = await segmentCharacter(ort, await env.getSession('matting'), anime);
  if (env.refineMatting !== false) {
    const person = await segmentPerson(ort, await env.getSession('matting2'), anime);
    alpha = mergeAlphas(alpha, person, anime.width, anime.height);
  }
  const bounds = alphaBounds(alpha, anime.width, anime.height);
  const anchor = bounds
    ? { x: bounds.x + bounds.w / 2, y: bounds.y + bounds.h }
    : { x: anime.width / 2, y: anime.height };
  const focus = bounds
    ? { x: bounds.x + bounds.w / 2, y: bounds.y + bounds.h * 0.75 }
    : { x: anime.width / 2, y: anime.height / 2 };

  const onCharacter = (x: number, y: number): boolean => {
    const xi = Math.min(anime.width - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(anime.height - 1, Math.max(0, Math.round(y)));
    return alpha[yi * anime.width + xi] > 0.3;
  };

  // ① 关键点匹配（精确）
  env.onStage?.('matching');
  const kpMatches = (
    await matchKeypoints(ort, await env.getSession('extractor'), await env.getSession('matcher'), anime, real)
  ).filter((m) => !onCharacter(m.ax, m.ay));
  let align = estimateAlignment(kpMatches, {
    threshold: 0.01 * real.width,
    animeWidth: anime.width,
    animeHeight: anime.height,
    realWidth: real.width,
    realHeight: real.height,
    focus,
    minInliers: 30,
    minCoverage: 0.15,
    strongMatches: 90,
  });
  let level: MatchLevel = align.sameScene ? 'precise' : 'none';

  // ② 兜底：语义 patch 特征（粗略，但对画风差异更宽容）
  if (level === 'none') {
    env.onStage?.('fallback');
    const session = await env.getSession('features');
    const fa = await extractFeatures(ort, session, anime);
    const fb = await extractFeatures(ort, session, real);
    const coarse = matchFeatures(fa, fb, { maskA: { alpha, width: anime.width, height: anime.height } });
    let globalSim = 0;
    for (let d = 0; d < fa.dim; d++) globalSim += fa.cls[d] * fb.cls[d];
    const rough = estimateAlignment(coarse, {
      threshold: 1.5 * Math.sqrt(fb.cellW * fb.cellH),
      animeWidth: anime.width,
      animeHeight: anime.height,
      // 这里故意不传实景图尺寸：语义特征的区分力弱，覆盖率仍按整幅动漫画面考核，不随取景偏移放宽
      globalSim,
      minInliers: 25,
      minRatio: 0.2,
      minCoverage: 0.3,
    });
    if (rough.sameScene) {
      align = rough;
      level = 'rough';
    } else {
      // 两种方法都没认出来：保留内点更多的那份统计给界面展示，但不采用其变换
      align = { ...(rough.inliers > align.inliers ? rough : align), sameScene: false };
    }
  }

  // 场景不一致时界面不会生成合成图；这里仍给一个占位摆放，保证返回值完整
  const fallbackScale = real.width / anime.width;
  const placement: Placement =
    level !== 'none' && align.matrix
      ? placementFromMatrix(align.matrix, anchor.x, anchor.y)
      : {
          x: (anchor.x / anime.width) * real.width,
          y: (anchor.y / anime.height) * real.height,
          scale: fallbackScale,
          rotation: 0,
        };

  let framing = measureFraming(alpha, anime.width, bounds, placement, real.width, real.height);

  // 半身近景：人物在动漫里就被底边截断，说明它站在镜头跟前、脚在画面外。
  // 实景照片的下沿往往拍得没动漫那么低，按场景对齐算出来人物会掉到画面下面去。
  // 这种人物的"正确位置"本来就是贴着画面底边，所以把截断处对到实景照片的底边上，水平位置和大小不变。
  // 留 1.5% 的余量：截图常带几像素的边框或黑边
  const cutAtBottom = bounds !== null && bounds.y + bounds.h >= anime.height * 0.985;
  if (level !== 'none' && cutAtBottom && framing.visible < MIN_VISIBLE && placement.y > real.height) {
    placement.y = real.height;
    placement.rotation = 0;
    framing = {
      ...measureFraming(alpha, anime.width, bounds, placement, real.width, real.height),
      snappedToBottom: true,
    };
  }

  env.onStage?.('done');
  return { alpha, bounds, level, align, placement, framing };
}
