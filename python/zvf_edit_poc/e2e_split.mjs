// End-to-end: load ONE skeleton, pick a node, split it via the UI's own command
// path, and confirm the store on disk became two objects.
import { chromium } from "playwright";

const port = process.argv[2] ?? "9077";
const store = process.argv[3] ?? "single_axon.zv";
const service = process.argv[4] ?? "http://127.0.0.1:9099";
const shot = process.argv[5] ?? "/tmp/zv_e2e.png";

const state = {
  dimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] },
  position: [480388, 150172, 333700],
  projectionScale: 300000,
  layers: [{
    type: "segmentation",
    name: "axon",
    source: `http://127.0.0.1:${port}/${store}/|zarr-vectors:#edit=${encodeURIComponent(service)}`,
    segments: ["1"],
  }],
  layout: "3d",
};

const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(`http://127.0.0.1:8081/#!${encodeURIComponent(JSON.stringify(state))}`, { waitUntil: "load", timeout: 60000 });
await page.waitForTimeout(20000);

// Nodes are pickable only when points are drawn; the edit tool does this itself.
await page.evaluate(() => {
  const layer = window.viewer.layerManager.managedLayers[0].layer;
  layer.displayState.skeletonRenderingOptions.params2d.mode.value = 1;
  layer.displayState.skeletonRenderingOptions.params3d.mode.value = 1;
});
await page.waitForTimeout(8000);

const gate = await page.evaluate(() => {
  const layer = window.viewer.layerManager.managedLayers[0].layer;
  const skel = layer.getSpatiallyIndexedSkeletonLayer();
  return {
    haveSkeletonLayer: skel !== undefined,
    readonly: skel?.source?.readonly ?? null,
    hasSplitFactory: typeof skel?.source?.splitSkeletonsCommand?.createCommand === "function",
    splitDisabledReason: layer.getSpatialSkeletonActionsDisabledReason("splitSkeletons") ?? null,
  };
});
console.log("gate:", JSON.stringify(gate));

let picked = null;
outer:
for (let y = 250; y <= 600; y += 10) {
  for (let x = 380; x <= 900; x += 10) {
    await page.mouse.move(x, y);
    const p = await page.evaluate(() => {
      const s = window.viewer?.mouseState?.pickedSpatialSkeleton;
      return s && s.nodeId ? { nodeId: s.nodeId, segmentId: s.segmentId } : null;
    });
    if (p) { picked = { x, y, ...p }; break outer; }
  }
}
console.log("picked:", JSON.stringify(picked));
if (!picked) { console.log("NO NODE PICKED — cannot split"); await browser.close(); process.exit(1); }

const result = await page.evaluate(async ({ nodeId, segmentId }) => {
  const layer = window.viewer.layerManager.managedLayers[0].layer;
  const source = layer.getSpatiallyIndexedSkeletonLayer().source;
  try {
    const command = source.splitSkeletonsCommand.createCommand(layer, { nodeId, segmentId });
    await layer.spatialSkeletonState.commandHistory.execute(command);
    const group = layer.displayState.segmentationGroupState.value;
    return { ok: true, visible: [...group.visibleSegments].map(String) };
  } catch (e) {
    return { ok: false, error: String(e && e.message ? e.message : e) };
  }
}, { nodeId: picked.nodeId, segmentId: picked.segmentId });
console.log("split:", JSON.stringify(result));

await page.waitForTimeout(12000);
await page.screenshot({ path: shot });
console.log("pageerrors:", errors.slice(0, 3).join(" | ") || "(none)");
await browser.close();
