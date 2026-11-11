// 动漫人物抠图：skytnt/anime-seg (ISNet)。输出与原图同尺寸的 alpha（0-1）。
import { resizeFloat, rgbaToRgbFloat, type RGBAImage } from './image.ts';
import type { OrtLike, SessionLike } from './ort-types.ts';

export async function segmentCharacter(
  ort: OrtLike,
  session: SessionLike,
  img: RGBAImage,
  size = 1024,
): Promise<Float32Array> {
  const { width: w0, height: h0 } = img;
  // 等比缩放到 size×size 内并居中补零（与 anime-seg 官方推理一致）
  const [h, w] =
    h0 > w0
      ? [size, Math.max(1, Math.floor((size * w0) / h0))]
      : [Math.max(1, Math.floor((size * h0) / w0)), size];
  const top = (size - h) >> 1;
  const left = (size - w) >> 1;
  const rgb = rgbaToRgbFloat(img, w, h);

  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y * w + x) * 3;
      const di = (y + top) * size + (x + left);
      input[di] = rgb[si];
      input[plane + di] = rgb[si + 1];
      input[2 * plane + di] = rgb[si + 2];
    }
  }

  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) };
  const out = await session.run(feeds);
  const mask = out[session.outputNames[0]].data as Float32Array;

  // 裁掉补边，再缩放回原图分辨率
  const cropped = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = (y + top) * size + left;
    cropped.set(mask.subarray(row, row + w), y * w);
  }
  const alpha = resizeFloat(cropped, w, h, 1, w0, h0);
  for (let i = 0; i < alpha.length; i++) alpha[i] = alpha[i] < 0 ? 0 : alpha[i] > 1 ? 1 : alpha[i];
  return alpha;
}

/**
 * 补充抠图模型（都是 ISNet 结构，输入输出格式相同，只是训练数据不同）：
 *  - ormbg：用人像数据训练；
 *  - isnet-general：通用显著物体分割。
 * anime-seg 在"深色衣服 + 深色或低对比背景"时会把黑外套、黑裤子抠丢，这两个模型各能补回一部分；
 * 但它们单独用都会漏人或多抠背景，所以只拿来补洞，见 mergeDarkRegions。
 * mean 是输入归一化的均值：ormbg 为 0，isnet-general 为 0.5。
 */
export async function segmentSalient(
  ort: OrtLike,
  session: SessionLike,
  img: RGBAImage,
  mean: number,
  size = 1024,
): Promise<Float32Array> {
  // 整幅图直接拉伸成正方形
  const rgb = rgbaToRgbFloat(img, size, size);
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = rgb[i * 3] - mean;
    input[plane + i] = rgb[i * 3 + 1] - mean;
    input[2 * plane + i] = rgb[i * 3 + 2] - mean;
  }
  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]),
  });
  const raw = out[session.outputNames[0]].data as Float32Array;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < plane; i++) {
    if (raw[i] < min) min = raw[i];
    if (raw[i] > max) max = raw[i];
  }
  const mask = new Float32Array(plane);
  // 画面里没有前景时输出整体很低，这时不能做 min-max 拉伸，否则噪声会被放大成"前景"
  const range = max > 0.5 ? max - min : 1;
  for (let i = 0; i < plane; i++) mask[i] = (raw[i] - min) / (range + 1e-9);
  const alpha = resizeFloat(mask, size, size, 1, img.width, img.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = alpha[i] < 0 ? 0 : alpha[i] > 1 ? 1 : alpha[i];
  return alpha;
}

/**
 * 用补充模型的结果给 anime-seg 补洞。补进来的像素必须同时满足三条，缺一不可：
 *  1. 是深色的——要补的就是被漏掉的深色衣物；浅色的地面、柱子即使被补充模型认成前景也不采纳；
 *  2. 与 anime-seg 认定的人物连在一起；
 *  3. 所在的那一块与人物有足够长的接触边——裤子接着上衣、外套裹着身体，接触边都很长；
 *     而人物手边恰好挨着的深色物件（灯笼、栏杆）只在一两个点上相碰。
 */
export function mergeDarkRegions(img: RGBAImage, primary: Float32Array, extras: Float32Array[]): Float32Array {
  const { width, height, data } = img;
  const n = width * height;
  const TH = 0.3;

  // 候选像素：补充模型认为是前景 × 深色程度
  const add = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (primary[i] > 0.5) continue;
    let e = 0;
    for (const x of extras) if (x[i] > e) e = x[i];
    if (e <= 0.15) continue;
    const luma = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
    const dark = luma <= 0.28 ? 1 : luma >= 0.42 ? 0 : (0.42 - luma) / 0.14;
    // 补充模型在这些区域往往只给出半透明的置信度，放大一些，不然补回来的衣服是半透明的
    add[i] = Math.min(1, e * 1.8) * dark;
  }

  // 把候选像素分成连通块，统计每块的面积和与人物相邻的像素数
  const label = new Int32Array(n);
  const stack = new Int32Array(n);
  const keep: boolean[] = [false];
  let next = 1;
  const touchesPrimary = (i: number): boolean => {
    const x = i % width;
    return (
      (x > 0 && primary[i - 1] > 0.5) ||
      (x < width - 1 && primary[i + 1] > 0.5) ||
      (i >= width && primary[i - width] > 0.5) ||
      (i < n - width && primary[i + width] > 0.5)
    );
  };
  for (let start = 0; start < n; start++) {
    if (label[start] !== 0 || add[start] <= TH) continue;
    let top = 0;
    stack[top++] = start;
    label[start] = next;
    let area = 0;
    let touch = 0;
    while (top > 0) {
      const i = stack[--top];
      area++;
      if (touchesPrimary(i)) touch++;
      const x = i % width;
      if (x > 0 && label[i - 1] === 0 && add[i - 1] > TH) {
        label[i - 1] = next;
        stack[top++] = i - 1;
      }
      if (x < width - 1 && label[i + 1] === 0 && add[i + 1] > TH) {
        label[i + 1] = next;
        stack[top++] = i + 1;
      }
      if (i >= width && label[i - width] === 0 && add[i - width] > TH) {
        label[i - width] = next;
        stack[top++] = i - width;
      }
      if (i < n - width && label[i + width] === 0 && add[i + width] > TH) {
        label[i + width] = next;
        stack[top++] = i + width;
      }
    }
    // 接触边长度要与这一块的"尺寸"（面积开方）成比例
    keep.push(touch >= TOUCH_RATIO * Math.sqrt(area));
    next++;
  }

  const out = Float32Array.from(primary);
  for (let i = 0; i < n; i++) {
    if (label[i] !== 0 && keep[label[i]] && add[i] > out[i]) out[i] = add[i];
  }
  return out;
}

/** 新增的一块区域，与人物相邻的像素数至少要达到 √面积 的这个倍数 */
const TOUCH_RATIO = 0.3;

/** alpha 中超过阈值的像素的包围盒；没有前景时返回 null。 */
export function alphaBounds(
  alpha: Float32Array,
  width: number,
  height: number,
  threshold = 0.05,
): { x: number; y: number; w: number; h: number; coverage: number } | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  let sum = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = alpha[y * width + x];
      sum += a;
      if (a > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, coverage: sum / (width * height) };
}
