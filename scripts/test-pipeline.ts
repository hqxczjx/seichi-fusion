// 命令行验证整条管线（不经过浏览器）：
//   node scripts/test-pipeline.ts <动漫图> <实景图> <输出前缀>
// 输出 <前缀>-cutout.png / -composite.jpg / -matches.jpg，并打印匹配统计。
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import type { RGBAImage } from '../src/core/image.ts';
import { MODEL_FILES, runPipeline, type ModelName } from '../src/core/pipeline.ts';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const [animePath, realPath, outPrefix = 'out'] = process.argv.slice(2);
if (!animePath || !realPath) {
  console.error('用法: node scripts/test-pipeline.ts <动漫图> <实景图> [输出前缀]');
  process.exit(1);
}

async function load(path: string): Promise<RGBAImage> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data.buffer, data.byteOffset, data.length), width: info.width, height: info.height };
}

const [anime, real] = await Promise.all([load(animePath), load(realPath)]);
const sessions = new Map<ModelName, Promise<unknown>>();
const timeline: [string, number][] = [['start', performance.now()]];
const { alpha, bounds, level, align: result, placement, framing } = await runPipeline(
  {
    ort,
    getSession(name) {
      if (!sessions.has(name)) {
        sessions.set(name, ort.InferenceSession.create(join(ROOT, 'public/models', MODEL_FILES[name])));
      }
      return sessions.get(name) as never;
    },
    onStage: (stage) => timeline.push([stage, performance.now()]),
  },
  anime,
  real,
);

console.log(
  JSON.stringify(
    {
      anime: `${anime.width}x${anime.height}`,
      real: `${real.width}x${real.height}`,
      character: bounds,
      level,
      matches: result.matches.length,
      inliers: result.inliers,
      inlierRatio: +result.inlierRatio.toFixed(3),
      coverage: +result.coverage.toFixed(3),
      overlap: +result.overlap.toFixed(3),
      spread: +result.spread.toFixed(3),
      score: +result.score.toFixed(3),
      placement: Object.fromEntries(Object.entries(placement).map(([k, v]) => [k, +v.toFixed(3)])),
      framing: { visible: +framing.visible.toFixed(3), overflow: framing.overflow, snappedToBottom: framing.snappedToBottom },
      ms: Object.fromEntries(timeline.slice(1).map(([name, t], i) => [name, Math.round(t - timeline[i][1])])),
    },
    null,
    1,
  ),
);

// --- 抠图结果 ---
const cut = Buffer.alloc(anime.width * anime.height * 4);
for (let i = 0; i < anime.width * anime.height; i++) {
  cut[i * 4] = anime.data[i * 4];
  cut[i * 4 + 1] = anime.data[i * 4 + 1];
  cut[i * 4 + 2] = anime.data[i * 4 + 2];
  cut[i * 4 + 3] = Math.round(alpha[i] * 255);
}
await sharp(cut, { raw: { width: anime.width, height: anime.height, channels: 4 } })
  .png()
  .toFile(`${outPrefix}-cutout.png`);

// --- 合成（逆向仿射 + 双线性采样）---
{
  // 人物只做等比缩放 + 旋转，锚点（包围盒底边中点）落在 placement 给出的位置
  const ancX = bounds ? bounds.x + bounds.w / 2 : anime.width / 2;
  const ancY = bounds ? bounds.y + bounds.h : anime.height;
  const cos = Math.cos(placement.rotation) * placement.scale;
  const sin = Math.sin(placement.rotation) * placement.scale;
  const [a, b, c, d] = [cos, -sin, sin, cos];
  const tx = placement.x - (a * ancX + b * ancY);
  const ty = placement.y - (c * ancX + d * ancY);
  const det = a * d - b * c;
  const out = Buffer.from(real.data);
  for (let y = 0; y < real.height; y++) {
    for (let x = 0; x < real.width; x++) {
      const u = (d * (x - tx) - b * (y - ty)) / det - 0.5;
      const v = (-c * (x - tx) + a * (y - ty)) / det - 0.5;
      const u0 = Math.floor(u);
      const v0 = Math.floor(v);
      if (u0 < 0 || v0 < 0 || u0 >= anime.width - 1 || v0 >= anime.height - 1) continue;
      const fu = u - u0;
      const fv = v - v0;
      let r = 0;
      let g = 0;
      let bl = 0;
      let al = 0;
      for (let k = 0; k < 4; k++) {
        const px = u0 + (k & 1);
        const py = v0 + (k >> 1);
        const w = (k & 1 ? fu : 1 - fu) * (k >> 1 ? fv : 1 - fv);
        const i = py * anime.width + px;
        const pa = alpha[i] * w;
        r += anime.data[i * 4] * pa;
        g += anime.data[i * 4 + 1] * pa;
        bl += anime.data[i * 4 + 2] * pa;
        al += pa;
      }
      if (al < 1e-3) continue;
      const o = (y * real.width + x) * 4;
      out[o] = r + out[o] * (1 - al);
      out[o + 1] = g + out[o + 1] * (1 - al);
      out[o + 2] = bl + out[o + 2] * (1 - al);
    }
  }
  await sharp(out, { raw: { width: real.width, height: real.height, channels: 4 } })
    .jpeg({ quality: 88 })
    .toFile(`${outPrefix}-composite.jpg`);
}

// --- 匹配连线可视化 ---
{
  const H = 540;
  const wa = Math.round((anime.width * H) / anime.height);
  const wb = Math.round((real.width * H) / real.height);
  const sa = H / anime.height;
  const sb = H / real.height;
  const lines = result.matches
    .filter((m) => m.inlier)
    .map(
      (m, i) =>
        `<line x1="${(m.ax * sa).toFixed(1)}" y1="${(m.ay * sa).toFixed(1)}" x2="${(wa + m.bx * sb).toFixed(1)}" y2="${(m.by * sb).toFixed(1)}" stroke="hsl(${(i * 47) % 360} 90% 55%)" stroke-width="1" opacity="0.7"/>`,
    )
    .join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${wa + wb}" height="${H}">${lines}</svg>`;
  await sharp({ create: { width: wa + wb, height: H, channels: 3, background: '#000' } })
    .composite([
      { input: await sharp(animePath).rotate().resize(wa, H, { fit: 'fill' }).toBuffer(), left: 0, top: 0 },
      { input: await sharp(realPath).rotate().resize(wb, H, { fit: 'fill' }).toBuffer(), left: wa, top: 0 },
      { input: Buffer.from(svg), left: 0, top: 0 },
    ])
    .jpeg({ quality: 80 })
    .toFile(`${outPrefix}-matches.jpg`);
}
