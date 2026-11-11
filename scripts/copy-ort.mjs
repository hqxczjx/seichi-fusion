// 把 onnxruntime-web 的 wasm 运行时复制到 public/ort/，由页面自行托管（不依赖 CDN）。
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules/onnxruntime-web/dist');
const dst = join(root, 'public/ort');
mkdirSync(dst, { recursive: true });
for (const file of ['ort-wasm-simd-threaded.asyncify.wasm']) {
  const from = join(src, file);
  const to = join(dst, file);
  if (!existsSync(to) || statSync(to).size !== statSync(from).size) copyFileSync(from, to);
}
