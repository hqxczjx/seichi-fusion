import type { Placement } from './core/align.ts';
import { MIN_VISIBLE, type ModelName, type Stage } from './core/pipeline.ts';
import type { RawImage, ResultPayload, RunRequest, WorkerMessage } from './protocol.ts';

/** 处理与输出的最长边上限（像素）。再大对效果没有帮助，只会更慢更占内存 */
const MAX_SIDE = 2560;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const runBtn = $<HTMLButtonElement>('run');
const sampleBtn = $<HTMLButtonElement>('sample');
const statusBox = $<HTMLElement>('status');
const statusText = $<HTMLElement>('status-text');
const bar = $<HTMLElement>('bar');
const barFill = $<HTMLElement>('bar-fill');
const resultBox = $<HTMLElement>('result');
const verdict = $<HTMLElement>('verdict');
const canvas = $<HTMLCanvasElement>('canvas');
const meta = $<HTMLElement>('meta');
const composite = $<HTMLElement>('composite');
const cScale = $<HTMLInputElement>('c-scale');
const cRot = $<HTMLInputElement>('c-rot');
const cTone = $<HTMLInputElement>('c-tone');
const cShadow = $<HTMLInputElement>('c-shadow');
const cPoints = $<HTMLInputElement>('c-points');

/** 低于此值（但不低于 MIN_VISIBLE）：照常生成，但提示人物被画面边缘裁掉了一部分 */
const FULL_VISIBLE = 0.95;
const SIDE_LABEL = { top: '上方', bottom: '下方', left: '左侧', right: '右侧' } as const;

interface Loaded {
  canvas: HTMLCanvasElement;
  image: ImageData;
}

const inputs: { anime: Loaded | null; real: Loaded | null } = { anime: null, real: null };

interface Character {
  /** 抠出来的人物（裁到包围盒，未做色调调整） */
  base: ImageData;
  /** 实际绘制用的画布（已做色调调整） */
  canvas: HTMLCanvasElement;
  /** 锚点（包围盒底边中点）在 canvas 里的坐标 */
  anchorX: number;
  anchorY: number;
  /** 人物是否被画面底边截断（半身像不画脚下阴影） */
  cutOff: boolean;
  /** 色调迁移参数：动漫背景 → 实景 的逐通道增益与偏移 */
  gain: [number, number, number];
  bias: [number, number, number];
}

let character: Character | null = null;
let result: ResultPayload | null = null;
let placement: Placement = { x: 0, y: 0, scale: 1, rotation: 0 };
let autoPlacement: Placement = placement;

// ---------- 图片载入 ----------

async function loadImage(source: Blob): Promise<Loaded> {
  const bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' });
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(bitmap.width * scale));
  c.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, c.width, c.height);
  bitmap.close();
  return { canvas: c, image: ctx.getImageData(0, 0, c.width, c.height) };
}

async function setInput(kind: 'anime' | 'real', source: Blob) {
  const drop = $<HTMLLabelElement>(`drop-${kind}`);
  try {
    inputs[kind] = await loadImage(source);
  } catch {
    showError('这张图片打不开，请换一张 JPG / PNG / WebP 图片。');
    return;
  }
  const img = drop.querySelector('img')!;
  URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(source);
  img.hidden = false;
  drop.querySelector<HTMLElement>('.drop-hint')!.hidden = true;
  runBtn.disabled = !(inputs.anime && inputs.real) || busy;
  if (!busy) statusBox.hidden = true;
}

for (const kind of ['anime', 'real'] as const) {
  const drop = $<HTMLLabelElement>(`drop-${kind}`);
  const input = drop.querySelector('input')!;
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) void setInput(kind, file);
    input.value = '';
  });
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    const file = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith('image/'));
    if (file) void setInput(kind, file);
  });
}

sampleBtn.addEventListener('click', async () => {
  sampleBtn.disabled = true;
  try {
    const base = `${import.meta.env.BASE_URL}samples/`;
    const [a, r] = await Promise.all([fetch(`${base}anime.jpg`), fetch(`${base}real.jpg`)]);
    if (!a.ok || !r.ok) throw new Error('sample');
    await setInput('anime', await a.blob());
    await setInput('real', await r.blob());
  } catch {
    showError('示例图片加载失败。');
  } finally {
    sampleBtn.disabled = false;
  }
});

// ---------- 与 Worker 通信 ----------

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
let busy = false;
let runId = 0;

const STAGE_TEXT: Record<Stage, string> = {
  matting: '正在抠出动漫人物…',
  matching: '正在比对两张图的场景…',
  fallback: '关键点不够，改用语义特征做粗略对齐…',
  done: '完成',
};
const MODEL_LABEL: Record<ModelName, string> = {
  matting: '人物抠图模型',
  matting2: '人物抠图补充模型（1/2）',
  matting3: '人物抠图补充模型（2/2）',
  extractor: '关键点模型',
  matcher: '场景匹配模型',
  features: '语义特征模型',
};

function showStatus(text: string) {
  statusBox.hidden = false;
  statusBox.classList.remove('error');
  statusText.textContent = text;
  bar.hidden = true;
}

function showError(text: string) {
  statusBox.hidden = false;
  statusBox.classList.add('error');
  statusText.textContent = text;
  bar.hidden = true;
}

runBtn.addEventListener('click', () => {
  if (!inputs.anime || !inputs.real || busy) return;
  busy = true;
  runBtn.disabled = true;
  resultBox.hidden = true;
  showStatus('准备中…');
  const raw = (l: Loaded): RawImage => ({ data: l.image.data, width: l.image.width, height: l.image.height });
  const req: RunRequest = { type: 'run', id: ++runId, anime: raw(inputs.anime), real: raw(inputs.real) };
  worker.postMessage(req);
});

worker.onmessage = (ev: MessageEvent<WorkerMessage>) => {
  const msg = ev.data;
  if (msg.id !== runId) return;
  switch (msg.type) {
    case 'stage':
      if (msg.stage !== 'done') showStatus(STAGE_TEXT[msg.stage]);
      break;
    case 'download': {
      const mb = (n: number) => (n / 1e6).toFixed(0);
      statusBox.hidden = false;
      statusText.textContent = msg.total
        ? `下载${MODEL_LABEL[msg.model]}（${mb(msg.loaded)} / ${mb(msg.total)} MB，仅首次需要）`
        : `下载${MODEL_LABEL[msg.model]}（${mb(msg.loaded)} MB）`;
      bar.hidden = !msg.total;
      barFill.style.width = msg.total ? `${(msg.loaded / msg.total) * 100}%` : '0';
      break;
    }
    case 'retry': {
      const mb = (n: number) => (n / 1e6).toFixed(0);
      statusBox.hidden = false;
      statusText.textContent = `下载${MODEL_LABEL[msg.model]}时连接中断，正在从 ${mb(msg.loaded)} MB 处继续…`;
      break;
    }
    case 'result':
      busy = false;
      runBtn.disabled = false;
      statusBox.hidden = true;
      showResult(msg.result);
      break;
    case 'error':
      busy = false;
      runBtn.disabled = false;
      showError(`处理失败：${msg.message}`);
      break;
  }
};
worker.onerror = (e) => {
  busy = false;
  runBtn.disabled = !(inputs.anime && inputs.real);
  showError(`处理失败：${e.message || '推理线程出错'}`);
};

// ---------- 结果与合成 ----------

/** 逐通道均值与标准差（抽样），skip 返回 true 的像素不计入 */
function channelStats(img: ImageData, skip?: (index: number) => boolean) {
  const mean = [0, 0, 0];
  const sq = [0, 0, 0];
  let n = 0;
  const step = Math.max(1, Math.floor(Math.sqrt((img.width * img.height) / 40000)));
  for (let y = 0; y < img.height; y += step) {
    for (let x = 0; x < img.width; x += step) {
      const i = y * img.width + x;
      if (skip?.(i)) continue;
      for (let c = 0; c < 3; c++) {
        const v = img.data[i * 4 + c];
        mean[c] += v;
        sq[c] += v * v;
      }
      n++;
    }
  }
  if (n === 0) return { mean: [128, 128, 128], std: [60, 60, 60] };
  return {
    mean: mean.map((m) => m / n),
    std: sq.map((s, c) => Math.sqrt(Math.max(1, s / n - (mean[c] / n) ** 2))),
  };
}

function buildCharacter(anime: ImageData, real: ImageData, r: ResultPayload): Character | null {
  const b = r.bounds;
  if (!b || b.coverage < 0.002) return null;
  const pad = 2;
  const x0 = Math.max(0, b.x - pad);
  const y0 = Math.max(0, b.y - pad);
  const x1 = Math.min(anime.width, b.x + b.w + pad);
  const y1 = Math.min(anime.height, b.y + b.h + pad);
  const w = x1 - x0;
  const h = y1 - y0;
  const base = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y + y0) * anime.width + (x + x0);
      const di = (y * w + x) * 4;
      base.data[di] = anime.data[si * 4];
      base.data[di + 1] = anime.data[si * 4 + 1];
      base.data[di + 2] = anime.data[si * 4 + 2];
      base.data[di + 3] = Math.round(r.alpha[si] * 255);
    }
  }

  // 两个场景的光照差异 ≈ 动漫背景与实景照片的整体色彩差异；把同样的变化施加到人物上
  const sa = channelStats(anime, (i) => r.alpha[i] > 0.1);
  const sr = channelStats(real);
  const gain = [0, 1, 2].map((c) => Math.min(1.5, Math.max(0.67, sr.std[c] / sa.std[c]))) as [number, number, number];
  const bias = [0, 1, 2].map((c) => sr.mean[c] - sa.mean[c] * gain[c]) as [number, number, number];

  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return {
    base,
    canvas: c,
    anchorX: b.x + b.w / 2 - x0,
    anchorY: b.y + b.h - y0,
    cutOff: b.y + b.h >= anime.height * 0.985,
    gain,
    bias,
  };
}

function applyTone(ch: Character, strength: number) {
  const out = new ImageData(new Uint8ClampedArray(ch.base.data), ch.base.width, ch.base.height);
  if (strength > 0) {
    const d = out.data;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue;
      for (let c = 0; c < 3; c++) {
        const v = d[i + c];
        d[i + c] = v + strength * (v * ch.gain[c] + ch.bias[c] - v);
      }
    }
  }
  ch.canvas.getContext('2d')!.putImageData(out, 0, 0);
}

function render() {
  const real = inputs.real;
  if (!real) return;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(real.canvas, 0, 0);

  if (character) {
    const p = placement;
    const shadow = Number(cShadow.value);
    if (shadow > 0 && !character.cutOff) {
      const rx = character.canvas.width * p.scale * 0.42;
      const ry = rx * 0.22;
      ctx.save();
      ctx.translate(p.x, p.y - ry * 0.25);
      ctx.rotate(p.rotation);
      ctx.scale(1, ry / rx);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
      g.addColorStop(0, `rgba(0,0,0,${0.55 * shadow})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, rx, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rotation);
    ctx.scale(p.scale, p.scale);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(character.canvas, -character.anchorX, -character.anchorY);
    ctx.restore();
  }

  if (cPoints.checked && result) {
    const r = Math.max(3, canvas.width / 400);
    ctx.fillStyle = 'rgba(92,214,255,0.9)';
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = r / 3;
    for (let i = 0; i < result.points.length; i += 2) {
      ctx.beginPath();
      ctx.arc(result.points[i], result.points[i + 1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function syncControls() {
  cScale.value = String(Math.log2(placement.scale / autoPlacement.scale));
  cRot.value = String((placement.rotation * 180) / Math.PI);
}

function showResult(r: ResultPayload) {
  const { anime, real } = inputs;
  if (!anime || !real) return;
  result = r;
  character = buildCharacter(anime.image, real.image, r);
  autoPlacement = { ...r.placement };
  placement = { ...r.placement };
  canvas.width = real.canvas.width;
  canvas.height = real.canvas.height;

  verdict.className = 'verdict';
  const say = (cls: string, title: string, detail: string) => {
    verdict.classList.add(cls);
    verdict.replaceChildren(
      Object.assign(document.createElement('strong'), { textContent: title }),
      Object.assign(document.createElement('span'), { textContent: detail }),
    );
  };
  // 只有"场景一致、抠到了人物、且人物所在位置被实景拍到了"才生成合成图；否则只给出提示
  const matched = r.level !== 'none';
  const sideText = r.framing.overflow.map((s) => SIDE_LABEL[s]).join('、');
  const outPct = Math.round((1 - r.framing.visible) * 100);
  if (matched && character && r.framing.visible < MIN_VISIBLE) {
    character = null;
    say(
      'none',
      '实景照片没有拍到人物所在的位置',
      `场景是一致的，但两张图取景有偏移：人物该站的位置有 ${outPct}% 在实景照片的画面${sideText ? `${sideText}` : ''}之外，所以没有生成合成图。请换一张把这个位置拍进去的照片。`,
    );
  } else if (!matched) {
    character = null;
    say(
      'none',
      '两张图的场景不一致',
      '没有在两张图之间找到足够多一致的对应点，所以没有生成合成图。请确认动漫截图和实景照片是同一个地点、相近的角度。',
    );
  } else if (!character) {
    say('none', '场景一致，但没有在动漫图里找到人物', '请换一张画面里有人物的截图。');
  } else if (r.framing.snappedToBottom) {
    say(
      r.level === 'precise' ? 'precise' : 'rough',
      r.level === 'precise' ? '场景一致' : '场景大致一致',
      `${r.level === 'precise' ? `找到 ${r.inliers} 处一致的对应点。` : ''}人物是贴着画面底边的近景，实景照片的下沿没拍到那么低，所以让人物贴着实景照片的底边摆放，可以拖动微调。`,
    );
  } else if (r.framing.visible < FULL_VISIBLE) {
    say(
      'rough',
      '场景一致，但人物有一部分在画面外',
      `两张图取景有偏移：人物有 ${outPct}% 超出了实景照片的${sideText || '边缘'}，合成图里这部分会被裁掉。`,
    );
  } else if (r.level === 'precise') {
    say('precise', '场景一致', `找到 ${r.inliers} 处一致的对应点，人物已放到实景中的相同位置。`);
  } else {
    say('rough', '场景大致一致', '两张图画风差异较大，只做了粗略对齐，建议拖动人物微调位置和大小。');
  }
  composite.hidden = !character;

  if (character) applyTone(character, Number(cTone.value));
  syncControls();
  meta.textContent = character
    ? `用时 ${(r.elapsedMs / 1000).toFixed(1)} 秒 · ${r.backend} · 输出 ${canvas.width}×${canvas.height}`
    : `用时 ${(r.elapsedMs / 1000).toFixed(1)} 秒 · ${r.backend}`;
  resultBox.hidden = false;
  if (character) render();
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- 交互 ----------

cScale.addEventListener('input', () => {
  placement.scale = autoPlacement.scale * 2 ** Number(cScale.value);
  render();
});
cRot.addEventListener('input', () => {
  placement.rotation = (Number(cRot.value) * Math.PI) / 180;
  render();
});
cTone.addEventListener('input', () => {
  if (character) applyTone(character, Number(cTone.value));
  render();
});
cShadow.addEventListener('input', render);
cPoints.addEventListener('change', render);

$('reset').addEventListener('click', () => {
  placement = { ...autoPlacement };
  syncControls();
  render();
});

let drag: { id: number; x: number; y: number } | null = null;
const toCanvas = (e: PointerEvent | WheelEvent) => {
  const rect = canvas.getBoundingClientRect();
  return { x: ((e.clientX - rect.left) * canvas.width) / rect.width, y: ((e.clientY - rect.top) * canvas.height) / rect.height };
};
canvas.addEventListener('pointerdown', (e) => {
  if (!character) return;
  drag = { id: e.pointerId, ...toCanvas(e) };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add('dragging');
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  const p = toCanvas(e);
  placement.x += p.x - drag.x;
  placement.y += p.y - drag.y;
  drag.x = p.x;
  drag.y = p.y;
  render();
});
const endDrag = (e: PointerEvent) => {
  if (drag?.id !== e.pointerId) return;
  drag = null;
  canvas.classList.remove('dragging');
};
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener(
  'wheel',
  (e) => {
    if (!character) return;
    e.preventDefault();
    const min = autoPlacement.scale * 2 ** Number(cScale.min);
    const max = autoPlacement.scale * 2 ** Number(cScale.max);
    placement.scale = Math.min(max, Math.max(min, placement.scale * Math.exp(-e.deltaY * 0.0015)));
    syncControls();
    render();
  },
  { passive: false },
);

$('download').addEventListener('click', () => {
  const showPoints = cPoints.checked;
  cPoints.checked = false; // 导出的图里不带调试用的对应点
  render();
  canvas.toBlob(
    (blob) => {
      cPoints.checked = showPoints;
      render();
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'seichi-fusion.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    },
    'image/png',
  );
});
