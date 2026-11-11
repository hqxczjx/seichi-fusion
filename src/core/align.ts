// 场景对齐：候选匹配点 → RANSAC 估计相似变换 → 仿射精修。
// 得到的 3×3 矩阵把"动漫图像素坐标"映射到"实景图像素坐标"。
import type { PatchFeatures } from './features.ts';

export interface Match {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  score: number;
  inlier: boolean;
}

export interface AlignResult {
  /** 行优先 3×3 矩阵：anime(x,y,1) → real(x,y,1)。未对齐成功时为 null */
  matrix: number[] | null;
  matches: Match[];
  inliers: number;
  /** 内点占互匹配点的比例 */
  inlierRatio: number;
  /** 内点覆盖了动漫画面的多大比例（0-1，按 8×6 粗网格统计） */
  coverage: number;
  /** 动漫画面有多大比例被实景照片拍到了（0-1）。取景偏移越大，这个值越小 */
  overlap: number;
  /** 内点的散布程度：沿最分散方向的标准差 ÷ 动漫图短边。全挤在一个小物体上时接近 0 */
  spread: number;
  /** 全局 CLS 特征余弦相似度 */
  globalSim: number;
  /** 综合场景相似度 0-1 */
  score: number;
  /** 是否判定为同一场景 */
  sameScene: boolean;
}

export interface AlignOptions {
  /** 动漫图上的人物 alpha（原图分辨率），人物覆盖的 patch 不参与匹配 */
  maskA?: { alpha: Float32Array; width: number; height: number };
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 每个 patch 被人物遮挡的比例 */
function patchOcclusion(f: PatchFeatures, mask: NonNullable<AlignOptions['maskA']>): Float32Array {
  const occ = new Float32Array(f.gw * f.gh);
  const cnt = new Float32Array(f.gw * f.gh);
  // 抽样统计即可，没必要逐像素
  const step = Math.max(1, Math.floor(Math.min(f.cellW, f.cellH) / 6));
  for (let y = 0; y < mask.height; y += step) {
    const gy = Math.min(f.gh - 1, Math.floor(y / f.cellH));
    for (let x = 0; x < mask.width; x += step) {
      const gx = Math.min(f.gw - 1, Math.floor(x / f.cellW));
      occ[gy * f.gw + gx] += mask.alpha[y * mask.width + x];
      cnt[gy * f.gw + gx] += 1;
    }
  }
  for (let i = 0; i < occ.length; i++) occ[i] = cnt[i] ? occ[i] / cnt[i] : 0;
  return occ;
}

/** 在 best 周围 3×3 邻域内用 softmax 加权求亚 patch 级位置 */
function softArgmax(
  sim: (idx: number) => number,
  best: number,
  gw: number,
  gh: number,
  temperature: number,
): [number, number] {
  const bx = best % gw;
  const by = (best / gw) | 0;
  const peak = sim(best);
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let dy = -1; dy <= 1; dy++) {
    const y = by + dy;
    if (y < 0 || y >= gh) continue;
    for (let dx = -1; dx <= 1; dx++) {
      const x = bx + dx;
      if (x < 0 || x >= gw) continue;
      const w = Math.exp((sim(y * gw + x) - peak) / temperature);
      sw += w;
      sx += w * x;
      sy += w * y;
    }
  }
  return [sx / sw + 0.5, sy / sw + 0.5];
}

export function matchFeatures(a: PatchFeatures, b: PatchFeatures, opts: AlignOptions = {}): Match[] {
  const na = a.gw * a.gh;
  const nb = b.gw * b.gh;
  const dim = a.dim;
  const occ = opts.maskA ? patchOcclusion(a, opts.maskA) : null;

  // 相似度矩阵（余弦）
  const sim = new Float32Array(na * nb);
  const fa = a.feats;
  const fb = b.feats;
  for (let i = 0; i < na; i++) {
    const ao = i * dim;
    const row = i * nb;
    for (let j = 0; j < nb; j++) {
      const bo = j * dim;
      let s = 0;
      for (let d = 0; d < dim; d++) s += fa[ao + d] * fb[bo + d];
      sim[row + j] = s;
    }
  }

  const bestForA = new Int32Array(na);
  const bestForB = new Int32Array(nb).fill(-1);
  const bestScoreB = new Float32Array(nb).fill(-Infinity);
  for (let i = 0; i < na; i++) {
    let best = -1;
    let bs = -Infinity;
    const row = i * nb;
    const blocked = occ !== null && occ[i] > 0.25;
    for (let j = 0; j < nb; j++) {
      const s = sim[row + j];
      if (s > bs) {
        bs = s;
        best = j;
      }
      if (!blocked && s > bestScoreB[j]) {
        bestScoreB[j] = s;
        bestForB[j] = i;
      }
    }
    bestForA[i] = blocked ? -1 : best;
  }

  const matches: Match[] = [];
  const T = 0.04;
  for (let i = 0; i < na; i++) {
    const j = bestForA[i];
    if (j < 0 || bestForB[j] !== i) continue;
    const [bx, by] = softArgmax((idx) => sim[i * nb + idx], j, b.gw, b.gh, T);
    const [ax, ay] = softArgmax((idx) => sim[idx * nb + j], i, a.gw, a.gh, T);
    matches.push({
      ax: ax * a.cellW,
      ay: ay * a.cellH,
      bx: bx * b.cellW,
      by: by * b.cellH,
      score: sim[i * nb + j],
      inlier: false,
    });
  }
  return matches;
}

type Affine = [number, number, number, number, number, number]; // [a b tx; c d ty]

function applyAffine(m: Affine, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

/** 两对点确定相似变换（缩放+旋转+平移） */
function similarityFrom2(m1: Match, m2: Match): Affine | null {
  const dax = m2.ax - m1.ax;
  const day = m2.ay - m1.ay;
  const dbx = m2.bx - m1.bx;
  const dby = m2.by - m1.by;
  const den = dax * dax + day * day;
  if (den < 1e-6) return null;
  // 复数除法 (db / da)
  const re = (dbx * dax + dby * day) / den;
  const im = (dby * dax - dbx * day) / den;
  return [re, -im, m1.bx - (re * m1.ax - im * m1.ay), im, re, m1.by - (im * m1.ax + re * m1.ay)];
}

/** 最小二乘仿射；给定 focus 时按到 focus 的距离做高斯加权（离人物越近的对应点越重要） */
function fitAffine(ms: Match[], focus?: { x: number; y: number; sigma: number }): Affine | null {
  if (ms.length < 3) return null;
  // 归一化以改善条件数
  let mx = 0;
  let my = 0;
  for (const m of ms) {
    mx += m.ax;
    my += m.ay;
  }
  mx /= ms.length;
  my /= ms.length;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sx = 0;
  let sy = 0;
  let n = 0;
  let bxx = 0;
  let bxy = 0;
  let bx1 = 0;
  let byx = 0;
  let byy = 0;
  let by1 = 0;
  for (const m of ms) {
    const x = m.ax - mx;
    const y = m.ay - my;
    let w = 1;
    if (focus) {
      const d2 = (m.ax - focus.x) ** 2 + (m.ay - focus.y) ** 2;
      w = Math.exp(-d2 / (2 * focus.sigma * focus.sigma)) + 0.05;
    }
    sxx += w * x * x;
    sxy += w * x * y;
    syy += w * y * y;
    sx += w * x;
    sy += w * y;
    n += w;
    bxx += w * x * m.bx;
    bxy += w * y * m.bx;
    bx1 += w * m.bx;
    byx += w * x * m.by;
    byy += w * y * m.by;
    by1 += w * m.by;
  }
  const A = [
    [sxx, sxy, sx],
    [sxy, syy, sy],
    [sx, sy, n],
  ];
  const r1 = solve3(A, [bxx, bxy, bx1]);
  const r2 = solve3(A, [byx, byy, by1]);
  if (!r1 || !r2) return null;
  return [r1[0], r1[1], r1[2] - r1[0] * mx - r1[1] * my, r2[0], r2[1], r2[2] - r2[0] * mx - r2[1] * my];
}

function solve3(A: number[][], b: number[]): number[] | null {
  const m = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < 1e-9) return null;
    [m[c], m[p]] = [m[p], m[c]];
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k = c; k < 4; k++) m[r][k] -= f * m[c][k];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/** 变换是否像一次"正常的巡礼翻拍"：不翻转、缩放与旋转在合理范围、不严重拉伸 */
function plausible(m: Affine): boolean {
  const det = m[0] * m[4] - m[1] * m[3];
  if (det <= 0) return false;
  const scale = Math.sqrt(det);
  if (scale < 0.2 || scale > 5) return false;
  const rot = Math.atan2(m[3] - m[1], m[0] + m[4]);
  if (Math.abs(rot) > (35 * Math.PI) / 180) return false;
  // 各向异性：两个奇异值之比
  const e = (m[0] + m[4]) / 2;
  const f = (m[0] - m[4]) / 2;
  const g = (m[3] + m[1]) / 2;
  const h = (m[3] - m[1]) / 2;
  const q = Math.hypot(e, h);
  const r = Math.hypot(f, g);
  const s1 = q + r;
  const s2 = Math.abs(q - r);
  return s2 > 1e-6 && s1 / s2 < 1.6;
}

/** 覆盖率按重叠区域折算时，重叠率的下限（见 estimateAlignment） */
const MIN_OVERLAP = 0.35;
/**
 * "内点没有挤成一小团"的下限（见 AlignResult.spread），两条通路都要满足。
 * 不同地点也会有一模一样的小物体（比如同款的道口警报机），几十个对应点全落在它上面时
 * spread 约 0.08；回归集里通过的同场景最低是 0.108。
 */
const MIN_SPREAD = 0.1;

export interface EstimateOptions {
  /** 内点判定阈值（实景图像素） */
  threshold: number;
  /** 人物在动漫图上的中心。场景有视差时，优先让人物附近的对应点对齐 */
  focus?: { x: number; y: number };
  /** 判定为同一场景所需的最少内点数 / 最低内点比例 / 最低覆盖率 */
  minInliers?: number;
  minRatio?: number;
  minCoverage?: number;
  /**
   * 候选匹配数达到这个值时，不再要求覆盖率达标（只要求内点别挤成一小团）。
   * 只适用于关键点匹配：匹配器对不相干的两张图本来就给不出多少候选。不设则不启用。
   */
  strongMatches?: number;
  animeWidth: number;
  animeHeight: number;
  /** 实景图尺寸。提供后，覆盖率只在"两张图重叠的那部分画面"里考核（容忍取景偏移） */
  realWidth?: number;
  realHeight?: number;
  globalSim?: number;
  iterations?: number;
  /** 确定性随机种子，便于复现 */
  seed?: number;
}

/** 由一组候选匹配点估计 动漫→实景 的变换，并给出"是否同一场景"的判定。 */
export function estimateAlignment(matches: Match[], opts: EstimateOptions): AlignResult {
  const globalSim = opts.globalSim ?? 0;
  const fail = (): AlignResult => ({
    matrix: null,
    matches,
    inliers: 0,
    inlierRatio: 0,
    coverage: 0,
    overlap: 0,
    spread: 0,
    globalSim,
    score: 0,
    sameScene: false,
  });
  if (matches.length < 6) return fail();

  const rand = mulberry32(opts.seed ?? 20240229);
  const iterations = opts.iterations ?? 4000;
  const thr2 = opts.threshold * opts.threshold;

  const countInliers = (m: Affine): number => {
    let c = 0;
    for (const mt of matches) {
      const [x, y] = applyAffine(m, mt.ax, mt.ay);
      const dx = x - mt.bx;
      const dy = y - mt.by;
      if (dx * dx + dy * dy < thr2) c++;
    }
    return c;
  };

  let best: Affine | null = null;
  let bestCount = 0;
  for (let it = 0; it < iterations; it++) {
    const i = (rand() * matches.length) | 0;
    let j = (rand() * (matches.length - 1)) | 0;
    if (j >= i) j++;
    const m = similarityFrom2(matches[i], matches[j]);
    if (!m || !plausible(m)) continue;
    const c = countInliers(m);
    if (c > bestCount) {
      bestCount = c;
      best = m;
    }
  }
  if (!best || bestCount < 6) return fail();

  // 仿射精修：内点 → 最小二乘 → 重新收集内点，迭代几轮
  let model = best;
  for (let round = 0; round < 4; round++) {
    const inl = matches.filter((mt) => {
      const [x, y] = applyAffine(model, mt.ax, mt.ay);
      return (x - mt.bx) ** 2 + (y - mt.by) ** 2 < thr2;
    });
    const refined = fitAffine(inl);
    if (!refined || !plausible(refined) || countInliers(refined) < countInliers(model)) break;
    model = refined;
  }
  // 场景有视差时没有任何一个 2D 变换能对齐全图：最后按"离人物近"加权再拟合一次
  if (opts.focus) {
    const inl = matches.filter((mt) => {
      const [x, y] = applyAffine(model, mt.ax, mt.ay);
      return (x - mt.bx) ** 2 + (y - mt.by) ** 2 < thr2;
    });
    const sigma = 0.35 * Math.hypot(opts.animeWidth, opts.animeHeight);
    const local = fitAffine(inl, { ...opts.focus, sigma });
    if (local && plausible(local) && countInliers(local) >= 0.8 * countInliers(model)) model = local;
  }

  // 覆盖率：把动漫图划成 8×6 的粗网格，看内点落在多少格里
  const GX = 8;
  const GY = 6;
  let inliers = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  const covered = new Set<number>();
  for (const mt of matches) {
    const [x, y] = applyAffine(model, mt.ax, mt.ay);
    mt.inlier = (x - mt.bx) ** 2 + (y - mt.by) ** 2 < thr2;
    if (mt.inlier) {
      inliers++;
      sumX += mt.ax;
      sumY += mt.ay;
      sumXX += mt.ax * mt.ax;
      sumYY += mt.ay * mt.ay;
      sumXY += mt.ax * mt.ay;
      const cx = Math.min(GX - 1, Math.max(0, Math.floor((mt.ax / opts.animeWidth) * GX)));
      const cy = Math.min(GY - 1, Math.max(0, Math.floor((mt.ay / opts.animeHeight) * GY)));
      covered.add(cy * GX + cx);
    }
  }
  const coverage = covered.size / (GX * GY);
  const inlierRatio = inliers / matches.length;

  // 重叠率：动漫画面的粗网格里，有多少格的中心被映射到了实景照片的画面内
  let overlap = 1;
  if (opts.realWidth && opts.realHeight) {
    let inside = 0;
    for (let gy = 0; gy < GY; gy++) {
      for (let gx = 0; gx < GX; gx++) {
        const [x, y] = applyAffine(model, ((gx + 0.5) / GX) * opts.animeWidth, ((gy + 0.5) / GY) * opts.animeHeight);
        if (x >= 0 && x < opts.realWidth && y >= 0 && y < opts.realHeight) inside++;
      }
    }
    overlap = inside / (GX * GY);
  }

  const minInliers = opts.minInliers ?? 30;
  const minRatio = opts.minRatio ?? 0;
  const minCoverage = opts.minCoverage ?? 0.15;
  // 实景只拍到动漫画面的一部分时，对应点自然只会出现在重叠的那部分里，
  // 所以覆盖率按重叠区域折算；MIN_OVERLAP 给折算设了下限，重叠太小时不会因此变得容易通过。
  const coverageInOverlap = coverage / Math.max(overlap, MIN_OVERLAP);

  // 散布程度：内点坐标协方差矩阵的最大特征值开方（最分散方向上的标准差）
  let spread = 0;
  if (inliers >= 2) {
    const vx = sumXX / inliers - (sumX / inliers) ** 2;
    const vy = sumYY / inliers - (sumY / inliers) ** 2;
    const vxy = sumXY / inliers - (sumX / inliers) * (sumY / inliers);
    const major = (vx + vy) / 2 + Math.sqrt(((vx - vy) / 2) ** 2 + vxy * vxy);
    spread = Math.sqrt(Math.max(0, major)) / Math.min(opts.animeWidth, opts.animeHeight);
  }

  // 前提：内点没有挤在同一个小物体上（notClustered）。在此之上两条通路任一满足即可：
  //  A. 内点在重叠区域里分布得足够广；
  //  B. 候选匹配本身就很多（匹配器已经很确信两张图有重叠）。
  //     B 用来接住"画面大半是天空/路面，对应点只能集中在一根电线杆上"这类取景。
  // notClustered 必须同时约束 A：覆盖率按重叠折算后，3 个网格就可能达标，
  // 而 3 个网格正好装得下一个两地都有的同款物体。
  const notClustered = spread >= MIN_SPREAD;
  const wideEnough = coverageInOverlap >= minCoverage;
  const strongEvidence = opts.strongMatches !== undefined && matches.length >= opts.strongMatches;
  const sameScene =
    inliers >= minInliers && inlierRatio >= minRatio && notClustered && (wideEnough || strongEvidence);
  const score = Math.min(
    1,
    0.6 * Math.min(1, inliers / (minInliers * 2.5)) + 0.4 * Math.min(1, coverageInOverlap / (minCoverage * 2)),
  );

  return {
    matrix: [model[0], model[1], model[2], model[3], model[4], model[5], 0, 0, 1],
    matches,
    inliers,
    inlierRatio,
    coverage,
    overlap,
    spread,
    globalSim,
    score,
    sameScene,
  };
}

export interface Placement {
  /** 人物锚点（包围盒底边中点）在实景图上的位置 */
  x: number;
  y: number;
  /** 等比缩放倍数（动漫像素 → 实景像素） */
  scale: number;
  /** 旋转（弧度） */
  rotation: number;
}

/**
 * 把场景的仿射变换折算成"人物怎么摆"：位置跟着场景走，
 * 但人物本身只做等比缩放 + 旋转，不做拉伸/错切，避免人物变形。
 */
export function placementFromMatrix(matrix: number[], anchorX: number, anchorY: number): Placement {
  const [a, b, tx, c, d, ty] = matrix;
  return {
    x: a * anchorX + b * anchorY + tx,
    y: c * anchorX + d * anchorY + ty,
    scale: Math.sqrt(Math.abs(a * d - b * c)),
    rotation: Math.atan2(c - b, a + d),
  };
}
