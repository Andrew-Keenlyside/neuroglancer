import { chromium } from "playwright";
const [,, port, store, segsArg, pos, scale, outPng, waitMs] = process.argv;
const segments = segsArg === "-" ? [] : segsArg.split(",");
const state = {
  dimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] },
  position: pos.split(",").map(Number),
  projectionScale: Number(scale),
  layers: [{ type: "segmentation", name: "zv", source: `http://127.0.0.1:${port}/${store}/|zarr-vectors:`, segments }],
  layout: "3d",
};
const url = `http://127.0.0.1:8081/#!${encodeURIComponent(JSON.stringify(state))}`;
const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const reqs = [];
page.on("response", (r) => { const u = r.url(); if (u.includes(`:${port}/`)) reqs.push(`${r.status()} ${u.split(`:${port}`)[1]}`); });
await page.goto(url, { waitUntil: "load", timeout: 60000 });
await page.waitForTimeout(Number(waitMs ?? 25000));
try { await page.screenshot({ path: outPng, timeout: 90000 }); } catch (e) { console.log("screenshot failed:", e.message.split("\n")[0]); }
const chunks0 = reqs; const chunks = reqs.filter((r) => r.includes("/c/") || r.includes("/vertices/"));
console.log(`total store requests: ${reqs.length}, chunk-ish: ${chunks.length}`);
console.log(chunks.slice(0, 6).join("\n") || "(NO CHUNK REQUESTS)");
await browser.close();
