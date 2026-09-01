// Does hovering a zarr-vectors skeleton resolve a NODE, or only its object?
import { chromium } from "playwright";
const [,, port, store, segs, pos, scale] = process.argv;
const state = {
  dimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] },
  position: pos.split(",").map(Number),
  projectionScale: Number(scale),
  layers: [{ type: "segmentation", name: "zv", source: `http://127.0.0.1:${port}/${store}/|zarr-vectors:`, segments: segs.split(",") }],
  layout: "3d",
};
const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.goto(`http://127.0.0.1:8081/#!${encodeURIComponent(JSON.stringify(state))}`, { waitUntil: "load", timeout: 60000 });
await page.waitForTimeout(20000);

// Nodes are only pickable when the layer draws POINTS as well as lines --
// hovering a line yields a "segment-edge" pick, which carries no node id. The
// edit tool flips this itself (setSpatialSkeletonModesToLinesAndPoints); do the
// same here so the probe measures the same thing the tool would see.
await page.evaluate(() => {
  const layer = window.viewer.layerManager.managedLayers[0].layer;
  const o = layer.displayState.skeletonRenderingOptions;
  o.params2d.mode.value = 1;
  o.params3d.mode.value = 1;
});
await page.waitForTimeout(6000);

let hit = null, tried = 0;
outer:
for (let y = 250; y <= 600 && !hit; y += 12) {
  for (let x = 380; x <= 900; x += 12) {
    await page.mouse.move(x, y);
    tried++;
    const picked = await page.evaluate(() => {
      const ms = window.viewer?.mouseState;
      const p = ms?.pickedSpatialSkeleton;
      if (!p) return null;
      return { nodeId: p.nodeId ?? null, segmentId: p.segmentId ?? null,
               segmentIdU64: p.segmentIdU64 != null ? String(p.segmentIdU64) : null };
    });
    if (picked && (picked.nodeId != null || picked.segmentId != null)) { hit = { x, y, ...picked }; break outer; }
  }
}
console.log(`probed ${tried} positions`);
console.log("pick:", JSON.stringify(hit));
await browser.close();
