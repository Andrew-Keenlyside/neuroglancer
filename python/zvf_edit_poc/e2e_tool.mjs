// The real user gesture: activate the Split tool, Ctrl+right-click a node.
import { chromium } from "playwright";
const [,, port, store, service, shot] = process.argv;
const state = {
  dimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] },
  position: [480388, 150172, 333700],
  projectionScale: 300000,
  layers: [{
    type: "segmentation", name: "axon",
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

await page.waitForTimeout(22000);

// Activate the tool the way the Skeleton tab's button does, and draw points so
// nodes are pickable (the tool does this itself via
// setSpatialSkeletonModesToLinesAndPoints).
const tool = await page.evaluate(() => {
  try {
    const layer = window.viewer.layerManager.managedLayers[0].layer;
    layer.displayState.skeletonRenderingOptions.params2d.mode.value = 1;
    layer.displayState.skeletonRenderingOptions.params3d.mode.value = 1;
    // Tools are bound to a key and then activated, which is what the Skeleton
    // tab's buttons do via ToolBindingWidget. Bind and activate directly here.
    layer.toolBinder.setJson("keyq", "spatialSkeletonSplitMode");
    // `activate` lives on the viewer-wide binder; the per-layer one only holds
    // the bindings.
    // Bind only; activation happens by actually pressing the key below, which
    // is the gesture a user makes.
    let t = null;
    try { t = layer.tool?.value?.toJSON?.() ?? null; } catch (e) { t = `err:${e.message}`; }
    let m = null;
    try { m = layer.displayState.skeletonRenderingOptions.params3d.mode.value; } catch (e) { m = `err:${e.message}`; }
    return { tool: t, mode3d: m };
  } catch (e) { return { error: String(e && e.message) }; }
});
console.log("tool state:", JSON.stringify(tool));
await page.waitForTimeout(8000);

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
if (!picked) { console.log("NO NODE PICKED"); await browser.close(); process.exit(1); }

// Press the bound key to activate the Split tool, then Ctrl+right-click the node.
await page.mouse.move(picked.x, picked.y);
await page.keyboard.press("q");
await page.waitForTimeout(1500);
const activeTool = await page.evaluate(() => {
  const b = window.viewer.layerManager.managedLayers[0].layer.toolBinder.globalBinder;
  return b.activeTool_?.tool?.toJSON?.() ?? b.activeTool?.tool?.toJSON?.() ?? null;
});
console.log("active tool:", JSON.stringify(activeTool));
await page.keyboard.down("Control");
await page.mouse.move(picked.x, picked.y);
await page.mouse.down({ button: "right" });
await page.mouse.up({ button: "right" });
await page.keyboard.up("Control");
await page.waitForTimeout(15000);

const after = await page.evaluate(() => {
  const layer = window.viewer.layerManager.managedLayers[0].layer;
  const g = layer.displayState.segmentationGroupState.value;
  return { visible: [...g.visibleSegments].map(String), body: document.body.innerText.slice(0, 300) };
});
console.log("visible after:", JSON.stringify(after.visible));
await page.screenshot({ path: shot });
console.log("pageerrors:", errors.slice(0, 2).join(" | ") || "(none)");
await browser.close();
