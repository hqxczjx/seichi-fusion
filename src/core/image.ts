// 与运行环境无关的图像工具（浏览器 Worker 与 Node 测试脚本共用）。

export interface RGBAImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

/**
 * 可分离三角滤波缩放（缩小时自动加宽滤波核以抗锯齿）。
 * src 为按像素交错存放的浮点数据，ch 为通道数。
 */
export function resizeFloat(
  src: Float32Array,
  sw: number,
  sh: number,
  ch: number,
  dw: number,
  dh: number,
): Float32Array {
  if (sw === dw && sh === dh) return src;
  const tmp = resampleAxis(src, sw, sh, ch, dw, true);
  return resampleAxis(tmp, dw, sh, ch, dh, false);
}

function resampleAxis(
  src: Float32Array,
  sw: number,
  sh: number,
  ch: number,
  dn: number,
  horizontal: boolean,
): Float32Array {
  const sn = horizontal ? sw : sh;
  const ow = horizontal ? dn : sw;
  const oh = horizontal ? sh : dn;
  const out = new Float32Array(ow * oh * ch);
  const scale = sn / dn;
  const support = Math.max(1, scale);

  // 预计算每个输出坐标的采样起点与权重
  const starts = new Int32Array(dn);
  const counts = new Int32Array(dn);
  const maxTaps = Math.ceil(support * 2) + 2;
  const weights = new Float32Array(dn * maxTaps);
  for (let i = 0; i < dn; i++) {
    const center = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(center - support));
    const hi = Math.min(sn - 1, Math.ceil(center + support));
    let sum = 0;
    let n = 0;
    for (let x = lo; x <= hi && n < maxTaps; x++, n++) {
      const w = Math.max(0, 1 - Math.abs(x + 0.5 - center) / support);
      weights[i * maxTaps + n] = w;
      sum += w;
    }
    if (sum === 0) {
      weights[i * maxTaps] = 1;
      sum = 1;
      n = Math.max(n, 1);
    }
    for (let k = 0; k < n; k++) weights[i * maxTaps + k] /= sum;
    starts[i] = lo;
    counts[i] = n;
  }

  if (horizontal) {
    for (let y = 0; y < sh; y++) {
      const rowIn = y * sw * ch;
      const rowOut = y * ow * ch;
      for (let i = 0; i < dn; i++) {
        const s = starts[i];
        const n = counts[i];
        const wo = i * maxTaps;
        for (let c = 0; c < ch; c++) {
          let acc = 0;
          for (let k = 0; k < n; k++) acc += src[rowIn + (s + k) * ch + c] * weights[wo + k];
          out[rowOut + i * ch + c] = acc;
        }
      }
    }
  } else {
    const stride = sw * ch;
    for (let i = 0; i < dn; i++) {
      const s = starts[i];
      const n = counts[i];
      const wo = i * maxTaps;
      const rowOut = i * stride;
      for (let k = 0; k < n; k++) {
        const w = weights[wo + k];
        const rowIn = (s + k) * stride;
        for (let x = 0; x < stride; x++) out[rowOut + x] += src[rowIn + x] * w;
      }
    }
  }
  return out;
}

/** RGBA(0-255) → 交错 RGB 浮点(0-1)，可选同时缩放。 */
export function rgbaToRgbFloat(img: RGBAImage, dw = img.width, dh = img.height): Float32Array {
  const { data, width, height } = img;
  const rgb = new Float32Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height; i++, j += 3) {
    rgb[j] = data[i * 4] / 255;
    rgb[j + 1] = data[i * 4 + 1] / 255;
    rgb[j + 2] = data[i * 4 + 2] / 255;
  }
  return resizeFloat(rgb, width, height, 3, dw, dh);
}
