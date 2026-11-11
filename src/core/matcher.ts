// 关键点匹配：ALIKED 提取关键点 + LightGlue 匹配。亚像素精度，是对齐的主力。
import type { Match } from './align.ts';
import { rgbaToRgbFloat, type RGBAImage } from './image.ts';
import type { OrtLike, SessionLike } from './ort-types.ts';

const SIZE = 640; // 该 ONNX 导出固定 640×640 输入
const DESC = 128;

interface Keypoints {
  /** 归一化到 [-1,1] 的坐标（相对 640 画布），LightGlue 直接使用 */
  kpts: Float32Array;
  desc: Float32Array;
  n: number;
  /** 画布像素 → 原图像素 的倍数 */
  toOriginal: number;
}

async function extract(ort: OrtLike, session: SessionLike, img: RGBAImage): Promise<Keypoints> {
  // 等比缩放后贴在画布左上角，其余补零
  const scale = SIZE / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const rgb = rgbaToRgbFloat(img, w, h);
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y * w + x) * 3;
      const di = y * SIZE + x;
      input[di] = rgb[si];
      input[plane + di] = rgb[si + 1];
      input[2 * plane + di] = rgb[si + 2];
    }
  }
  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]),
  });
  const k = out['keypoints'].data as Float32Array;
  const d = out['descriptors'].data as Float32Array;
  const s = out['scores'].data as Float32Array;

  // 丢掉落在补边区域里的点
  const keep: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const px = ((k[i * 2] + 1) / 2) * SIZE;
    const py = ((k[i * 2 + 1] + 1) / 2) * SIZE;
    if (s[i] > 0 && px < w && py < h) keep.push(i);
  }
  const kpts = new Float32Array(keep.length * 2);
  const desc = new Float32Array(keep.length * DESC);
  keep.forEach((i, q) => {
    kpts[q * 2] = k[i * 2];
    kpts[q * 2 + 1] = k[i * 2 + 1];
    desc.set(d.subarray(i * DESC, (i + 1) * DESC), q * DESC);
  });
  return { kpts, desc, n: keep.length, toOriginal: 1 / scale };
}

export async function matchKeypoints(
  ort: OrtLike,
  extractor: SessionLike,
  matcher: SessionLike,
  a: RGBAImage,
  b: RGBAImage,
): Promise<Match[]> {
  const ka = await extract(ort, extractor, a);
  const kb = await extract(ort, extractor, b);
  if (ka.n < 8 || kb.n < 8) return [];

  const out = await matcher.run({
    kpts0: new ort.Tensor('float32', ka.kpts, [1, ka.n, 2]),
    kpts1: new ort.Tensor('float32', kb.kpts, [1, kb.n, 2]),
    desc0: new ort.Tensor('float32', ka.desc, [1, ka.n, DESC]),
    desc1: new ort.Tensor('float32', kb.desc, [1, kb.n, DESC]),
  });
  const idx = out['matches0'].data as BigInt64Array;
  const scores = out['mscores0'].data as Float32Array;

  const px = (v: number, k: Keypoints) => ((v + 1) / 2) * SIZE * k.toOriginal;
  const matches: Match[] = [];
  for (let i = 0; i < scores.length; i++) {
    const ia = Number(idx[i * 2]);
    const ib = Number(idx[i * 2 + 1]);
    matches.push({
      ax: px(ka.kpts[ia * 2], ka),
      ay: px(ka.kpts[ia * 2 + 1], ka),
      bx: px(kb.kpts[ib * 2], kb),
      by: px(kb.kpts[ib * 2 + 1], kb),
      score: scores[i],
      inlier: false,
    });
  }
  return matches;
}
