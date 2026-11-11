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
 * 第二个抠图模型（ormbg，和 anime-seg 同为 ISNet 结构，但用人像数据训练）。
 * anime-seg 在"深色衣服 + 深色或低对比背景"时会把黑外套、黑裤子抠丢，ormbg 常能补回来；
 * 反过来 ormbg 单独用会漏掉一些动漫人物，所以两者配合使用，见 mergeAlphas。
 */
export async function segmentPerson(
  ort: OrtLike,
  session: SessionLike,
  img: RGBAImage,
  size = 1024,
): Promise<Float32Array> {
  // ormbg 的预处理：整幅图直接拉伸成正方形，像素值 0-1，不做均值方差归一化
  const rgb = rgbaToRgbFloat(img, size, size);
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = rgb[i * 3];
    input[plane + i] = rgb[i * 3 + 1];
    input[2 * plane + i] = rgb[i * 3 + 2];
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
 * 合并两个模型的结果：以 anime-seg 为主，ormbg 只用来"补洞"。
 * ormbg 给出的每一块连通区域，只有和 anime-seg 认定的人物有足够重叠才采纳——
 * 这样能补回同一个人身上被漏掉的深色衣服，又不会把 ormbg 误认的路人、物件带进来。
 */
export function mergeAlphas(primary: Float32Array, extra: Float32Array, width: number, height: number): Float32Array {
  const n = width * height;
  const label = new Int32Array(n); // 0 = 未访问/背景
  const stack = new Int32Array(n);
  const keep: boolean[] = [false];
  let next = 1;
  for (let start = 0; start < n; start++) {
    if (label[start] !== 0 || extra[start] <= 0.5) continue;
    let top = 0;
    stack[top++] = start;
    label[start] = next;
    let area = 0;
    let shared = 0;
    while (top > 0) {
      const i = stack[--top];
      area++;
      if (primary[i] > 0.5) shared++;
      const x = i % width;
      if (x > 0 && label[i - 1] === 0 && extra[i - 1] > 0.5) {
        label[i - 1] = next;
        stack[top++] = i - 1;
      }
      if (x < width - 1 && label[i + 1] === 0 && extra[i + 1] > 0.5) {
        label[i + 1] = next;
        stack[top++] = i + 1;
      }
      if (i >= width && label[i - width] === 0 && extra[i - width] > 0.5) {
        label[i - width] = next;
        stack[top++] = i - width;
      }
      if (i < n - width && label[i + width] === 0 && extra[i + width] > 0.5) {
        label[i + width] = next;
        stack[top++] = i + width;
      }
    }
    keep.push(shared >= 0.15 * area);
    next++;
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = label[i] !== 0 && keep[label[i]] ? extra[i] : 0;
    out[i] = primary[i] > e ? primary[i] : e;
  }
  return out;
}

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
