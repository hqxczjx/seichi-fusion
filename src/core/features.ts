// 场景特征：DINOv2 (with registers) 的 patch 特征。
// 自监督 ViT 的 patch 特征对"画风"不敏感、对"是什么东西/在哪"敏感，
// 所以能跨越二次元/三次元的域差做对应。
import { rgbaToRgbFloat, type RGBAImage } from './image.ts';
import type { OrtLike, SessionLike } from './ort-types.ts';

const PATCH = 14;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export interface PatchFeatures {
  /** gw*gh 个 L2 归一化的 patch 特征，按行优先排列 */
  feats: Float32Array;
  dim: number;
  gw: number;
  gh: number;
  /** 每个 patch 在原图坐标系中的宽高（像素） */
  cellW: number;
  cellH: number;
  /** L2 归一化的全局 CLS 特征 */
  cls: Float32Array;
}

export async function extractFeatures(
  ort: OrtLike,
  session: SessionLike,
  img: RGBAImage,
  longSide = 644,
): Promise<PatchFeatures> {
  const scale = longSide / Math.max(img.width, img.height);
  const gw = Math.max(4, Math.round((img.width * scale) / PATCH));
  const gh = Math.max(4, Math.round((img.height * scale) / PATCH));
  const w = gw * PATCH;
  const h = gh * PATCH;

  const rgb = rgbaToRgbFloat(img, w, h);
  const plane = w * h;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = (rgb[i * 3] - MEAN[0]) / STD[0];
    input[plane + i] = (rgb[i * 3 + 1] - MEAN[1]) / STD[1];
    input[2 * plane + i] = (rgb[i * 3 + 2] - MEAN[2]) / STD[2];
  }

  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, h, w]),
  });
  const hidden = out['last_hidden_state'] ?? out[session.outputNames[0]];
  const [, tokens, dim] = hidden.dims;
  const data = hidden.data as Float32Array;
  const n = gw * gh;
  // token 排列：[CLS, (registers...), patches]，patch 永远在末尾
  const skip = tokens - n;
  if (skip < 1) throw new Error(`特征 token 数异常：${tokens}，期望至少 ${n + 1}`);

  const feats = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const src = (skip + i) * dim;
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += data[src + d] * data[src + d];
    norm = 1 / Math.sqrt(norm + 1e-12);
    for (let d = 0; d < dim; d++) feats[i * dim + d] = data[src + d] * norm;
  }
  const cls = new Float32Array(dim);
  let cn = 0;
  for (let d = 0; d < dim; d++) cn += data[d] * data[d];
  cn = 1 / Math.sqrt(cn + 1e-12);
  for (let d = 0; d < dim; d++) cls[d] = data[d] * cn;

  return { feats, dim, gw, gh, cellW: img.width / gw, cellH: img.height / gh, cls };
}
