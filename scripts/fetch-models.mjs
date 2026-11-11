// 下载推理所需的 ONNX 模型到 public/models/（约 410 MB，只需执行一次）。
// 国内网络可设置镜像：HF_ENDPOINT=https://hf-mirror.com npm run models
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HF = process.env.HF_ENDPOINT ?? 'https://huggingface.co';
const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');

const MODELS = [
  // 动漫人物分割（ISNet，Apache-2.0）
  { file: 'anime-seg.onnx', url: `${HF}/skytnt/anime-seg/resolve/main/isnetis.onnx` },
  // 补充抠图模型（ormbg，Apache-2.0，半精度）：补回 anime-seg 漏掉的深色衣服
  { file: 'ormbg-fp16.onnx', url: `${HF}/onnx-community/ormbg-ONNX/resolve/main/onnx/model_fp16.onnx` },
  // 关键点提取 + 匹配（ALIKED BSD-3 / LightGlue Apache-2.0，ONNX 导出来自 bukuroo/ALIKED-LightGlue-ONNX）
  { file: 'aliked-n16rot-640.onnx', url: `${HF}/bukuroo/ALIKED-LightGlue-ONNX/resolve/main/aliked-n16rot-top2k-640.onnx` },
  { file: 'lightglue-aliked.onnx', url: `${HF}/bukuroo/ALIKED-LightGlue-ONNX/resolve/main/lightglue_for_aliked.onnx` },
  // 兜底用的语义特征（DINOv2 ViT-S/14 with registers，Apache-2.0），仅在关键点匹配失败时才会加载
  { file: 'dinov2-reg-small.onnx', url: `${HF}/onnx-community/dinov2-with-registers-small/resolve/main/onnx/model.onnx` },
];

mkdirSync(OUT, { recursive: true });
for (const { file, url } of MODELS) {
  const dest = join(OUT, file);
  if (existsSync(dest) && statSync(dest).size > 1e6) {
    console.log(`✓ ${file} 已存在，跳过`);
    continue;
  }
  console.log(`↓ ${file}  ←  ${url}`);
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`下载失败 ${res.status} ${url}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  let done = 0;
  let lastPct = -1;
  const body = Readable.fromWeb(res.body);
  body.on('data', (chunk) => {
    done += chunk.length;
    const pct = total ? Math.floor((done / total) * 100) : -1;
    if (pct !== lastPct && pct % 10 === 0) {
      lastPct = pct;
      console.log(`  ${pct}%  (${(done / 1e6).toFixed(0)} MB)`);
    }
  });
  await pipeline(body, createWriteStream(dest + '.part'));
  renameSync(dest + '.part', dest);
  console.log(`✓ ${file}  ${(done / 1e6).toFixed(1)} MB`);
}
