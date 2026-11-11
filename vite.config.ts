import { defineConfig } from 'vite';

// 跨源隔离：让 onnxruntime-web 能用多线程 WASM（SharedArrayBuffer）。
// 线上部署时静态服务器也要返回这两个响应头，否则会自动退回单线程（能用但更慢）。
// GitHub Pages 不能自定义响应头，所以那里是单线程 WASM + WebGPU。
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};
// VITE_NO_ISOLATION=1：本地模拟"没有这两个响应头"的托管环境
const headers = process.env.VITE_NO_ISOLATION ? {} : isolation;

export default defineConfig({
  // 部署在子路径下时（如 GitHub Pages 的 /仓库名/）用 VITE_BASE 指定
  base: process.env.VITE_BASE ?? '/',
  server: { headers },
  preview: { headers },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
