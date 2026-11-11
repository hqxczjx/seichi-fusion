# 次元融合 · 圣地巡礼合成器

上传动漫里的**一幕**和这一幕取景地的**实景照片**。系统先判断两张图是不是同一个场景：一致，就把动漫人物抠出来放到实景照片里的相同位置；不一致，直接提示、不生成。

所有计算都在浏览器里完成（ONNX Runtime Web，WebGPU / WASM），图片不会上传到服务器。

## 运行

```bash
npm install
npm run models   # 下载模型到 public/models/（约 320 MB，只需一次）
npm run dev
```

国内网络下载模型可以用镜像：`HF_ENDPOINT=https://hf-mirror.com npm run models`。

不经过浏览器、直接在命令行验证算法：

```bash
node scripts/test-pipeline.ts public/samples/anime.jpg public/samples/real.jpg out
# 生成 out-cutout.png（抠图）/ out-composite.jpg（合成）/ out-matches.jpg（对应点连线）
```

## 原理

| 步骤 | 做法 | 模型 |
| --- | --- | --- |
| 抠人物 | 动漫专用的显著性分割，输出逐像素 alpha | [anime-seg](https://huggingface.co/skytnt/anime-seg)（ISNet，176 MB） |
| 场景匹配 | 两张图各取 2000 个关键点并互相匹配，人物身上的点不参与 | [ALIKED + LightGlue](https://huggingface.co/bukuroo/ALIKED-LightGlue-ONNX)（6 + 46 MB） |
| 估计对齐 | RANSAC 求"动漫 → 实景"的相似变换，再做仿射精修；有视差时让人物附近的对应点权重更高 | 纯 TypeScript，见 `src/core/align.ts` |
| 兜底 | 关键点对不上时（画风与实景差异大），改用语义 patch 特征做粗略对齐 | [DINOv2-small](https://huggingface.co/onnx-community/dinov2-with-registers-small)（88 MB，用到时才下载） |
| 合成 | 人物只做等比缩放 + 旋转（不拉伸），按两张图的整体色彩差异做色调迁移，加脚下阴影 | Canvas 2D |

匹配结果分三档：

- **场景一致**：一致的对应点 ≥ 30 个，且没有挤在同一个小物体上（不同地点也会有同款的道口警报机、路牌），并且满足下面任意一条，位置可信，生成合成图：
  - 对应点分布得足够广。只在两张图重叠的那部分画面里考核，所以实景照片只拍到动漫画面的一半也能通过；
  - 候选匹配 ≥ 90 个。用来接住"画面大半是天空或路面，对应点只能集中在一根电线杆上"的取景；
- **场景大致一致**：只有语义特征对得上，生成合成图，但位置是粗略的，需要手动微调；
- **场景不一致**：只提示，不生成合成图。

场景一致之后还会检查取景偏移：把人物按算出的位置放到实景图上，看有多少落在画面内（`measureFraming`）。

- 落在画面内 ≥ 95%：正常生成；
- 50%–95%：照常生成，并提示人物有多少、从哪一侧被画面裁掉；
- 不足 50%：提示"实景照片没有拍到人物所在的位置"，不生成。

## 目录

```
src/core/       与运行环境无关的算法（浏览器 Worker 和 Node 共用）
src/worker.ts   模型下载/缓存与推理
src/main.ts     界面、合成画布、交互
scripts/        模型下载、命令行测试
public/samples  示例图片（来源见 CREDITS.md）
```

## 已知局限

- **还没有用真正的动漫截图验证过。** 开发时用的"动漫图"是把实景照片做平涂化处理后贴上维基娘合成的（避免把有版权的截图放进仓库）。真实动画背景是手绘的，关键点匹配的成功率会低于这里的测试结果，届时更多会落到"大致一致 + 手动微调"，也可能把本该一致的画面判成不一致。判定阈值（`src/core/pipeline.ts` 里的 `minInliers`、`strongMatches` 等）是在同一个地点的 3 张照片、2 张无关照片及其裁剪组合（25 组同场景、78 组不同场景）上校准的，样本很窄。
- 两张图机位差得远（比如换了个方向拍）时不会判为同一场景。
- 人物被前景遮挡的关系不会还原：人物永远画在实景照片的最上层。
- 一张图里有多个人物时会一起抠出、整体摆放，暂不支持单独挑选。
- 首次使用要下载约 230 MB 模型；抠图模型可以转成 fp16 减半，尚未做。

## 部署

`npm run build` 产出纯静态站点（`dist/`）。

### GitHub Pages

仓库自带 `.github/workflows/deploy.yml`：推送到 `main` 后自动构建并发布到 `https://<用户名>.github.io/<仓库名>/`（仓库 Settings → Pages 的 Source 需选 GitHub Actions）。线上构建有两点和本地不同：

- 模型文件超过 GitHub 的单文件上限，不进仓库，页面直接从 Hugging Face 下载（`VITE_MODEL_SOURCE=hf`）。访问不了 Hugging Face 的网络可以在 workflow 里加 `VITE_HF_ENDPOINT=https://hf-mirror.com`。
- GitHub Pages 不能自定义响应头，WASM 只能单线程。有 WebGPU 的浏览器影响不大（主要模型都在显卡上跑）；没有 WebGPU 时会明显变慢。

本地模拟这个环境：`VITE_MODEL_SOURCE=hf VITE_NO_ISOLATION=1 VITE_BASE=/seichi-fusion/ npm run dev`。

### 自己的服务器

1. 返回 `Cross-Origin-Opener-Policy: same-origin` 和 `Cross-Origin-Embedder-Policy: require-corp`，WASM 才能多线程运行。
2. 模型文件较大，建议放到 CDN / 对象存储，构建时用 `VITE_MODEL_BASE=https://your-cdn/models/` 指过去（需要允许跨域，并带 `Cross-Origin-Resource-Policy: cross-origin`）。

## 许可提示

anime-seg、LightGlue、DINOv2 为 Apache-2.0，ALIKED 为 BSD-3。ALIKED/LightGlue 的 ONNX 文件来自社区导出，上线前建议从官方权重自行导出一份。用户上传的动漫截图版权归原作权利人所有。
