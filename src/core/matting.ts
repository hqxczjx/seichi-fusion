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
