/**
 * @license
 * Copyright 2019 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import "#src/widget/render_scale_widget.css";

import { debounce, throttle } from "lodash-es";
import type { UserLayer } from "#src/layer/index.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import {
  getRenderScaleFromHistogramOffset,
  getRenderScaleHistogramOffset,
} from "#src/render_scale_statistics.js";
import type { TrackableValueInterface } from "#src/trackable_value.js";
import { WatchableValue } from "#src/trackable_value.js";
import { serializeColor } from "#src/util/color.js";
import { hsvToRgb } from "#src/util/colorspace.js";
import { RefCounted } from "#src/util/disposable.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import {
  EventActionMap,
  registerActionListener,
} from "#src/util/event_action_map.js";
import { vec3 } from "#src/util/geom.js";
import { clampToInterval } from "#src/util/lerp.js";
import { MouseEventBinder } from "#src/util/mouse_bindings.js";
import { numberToStringFixed } from "#src/util/number_to_string.js";
import { formatScaleWithUnitAsString } from "#src/util/si_units.js";
import type { LayerControlFactory } from "#src/widget/layer_control.js";

const updateInterval = 200;

const inputEventMap = EventActionMap.fromObject({
  mousedown0: { action: "set" },
  wheel: { action: "adjust-via-wheel" },
  dblclick0: { action: "reset" },
});

function formatPixelNumber(x: number) {
  if (x < 1 || x > 1024) {
    const exponent = Math.log2(x) | 0;
    const coeff = x / 2 ** exponent;
    return `${numberToStringFixed(coeff, 1)}p${exponent}`;
  }
  return Math.round(x) + "";
}

export interface RenderScaleWidgetOptions {
  histogram: RenderScaleHistogram;
  target: TrackableValueInterface<number>;
}

export class RenderScaleWidget extends RefCounted {
  label = document.createElement("div");
  element = document.createElement("div");
  canvas = document.createElement("canvas");
  legend = document.createElement("div");
  legendRenderScale = document.createElement("div");
  legendSpatialScale = document.createElement("div");
  legendChunks = document.createElement("div");
  protected unitOfTarget = "px";
  private ctx = this.canvas.getContext("2d")!;
  hoverTarget = new WatchableValue<[number, number] | undefined>(undefined);
  private throttledUpdateView = this.registerCancellable(
    throttle(() => this.debouncedUpdateView(), updateInterval, {
      leading: true,
      trailing: true,
    }),
  );
  private debouncedUpdateView = this.registerCancellable(
    debounce(() => this.updateView(), 0),
  );

  adjustViaWheel(event: WheelEvent) {
    const deltaY = this.getWheelMoveValue(event);
    if (deltaY === 0) {
      return;
    }
    this.hoverTarget.value = undefined;
    const logScaleMax = Math.round(
      this.histogram.logScaleOrigin +
        this.histogram.numBins * this.histogram.binSize,
    );
    const targetValue = clampToInterval(
      [2 ** this.histogram.logScaleOrigin, 2 ** (logScaleMax - 1)],
      this.target.value * 2 ** Math.sign(deltaY),
    ) as number;
    this.target.value = targetValue;
    event.preventDefault();
  }

  constructor(
    public histogram: RenderScaleHistogram,
    public target: TrackableValueInterface<number>,
  ) {
    super();
    const {
      canvas,
      label,
      element,
      legend,
      legendRenderScale,
      legendSpatialScale,
      legendChunks,
    } = this;
    label.className = "neuroglancer-render-scale-widget-prompt";
    element.className = "neuroglancer-render-scale-widget";
    element.title = inputEventMap.describe();
    legend.className = "neuroglancer-render-scale-widget-legend";
    element.appendChild(label);
    element.appendChild(canvas);
    element.appendChild(legend);
    legendRenderScale.title = "Target resolution of data in screen pixels";
    legendChunks.title = "Number of chunks rendered";
    legend.appendChild(legendRenderScale);
    legend.appendChild(legendChunks);
    legend.appendChild(legendSpatialScale);
    this.registerDisposer(histogram.changed.add(this.throttledUpdateView));
    this.registerDisposer(
      histogram.visibility.changed.add(this.debouncedUpdateView),
    );
    this.registerDisposer(target.changed.add(this.debouncedUpdateView));
    this.registerDisposer(new MouseEventBinder(canvas, inputEventMap));
    this.registerDisposer(target.changed.add(this.debouncedUpdateView));
    this.registerDisposer(
      this.hoverTarget.changed.add(this.debouncedUpdateView),
    );

    const getTargetValue = (event: MouseEvent) => {
      const axis = this.resolutionAxis;
      if (this.byResolution && axis !== undefined) {
        const log =
          axis.lo + (event.offsetX / canvas.width) * (axis.hi - axis.lo);
        return 2 ** log / axis.pixelSize;
      }
      const [start, end] = this.binWindow;
      const position = start + (event.offsetX / canvas.width) * (end - start);
      return getRenderScaleFromHistogramOffset(
        position,
        this.histogram.logScaleOrigin,
        this.histogram.binSize,
      );
    };
    this.registerEventListener(canvas, "pointermove", (event: MouseEvent) => {
      this.hoverTarget.value = [getTargetValue(event), event.offsetY];
    });

    this.registerEventListener(canvas, "pointerleave", () => {
      this.hoverTarget.value = undefined;
    });

    this.registerDisposer(
      registerActionListener<MouseEvent>(canvas, "set", (actionEvent) => {
        this.target.value = getTargetValue(actionEvent.detail);
      }),
    );

    this.registerDisposer(
      registerActionListener<WheelEvent>(
        canvas,
        "adjust-via-wheel",
        (actionEvent) => {
          this.adjustViaWheel(actionEvent.detail);
        },
      ),
    );

    this.registerDisposer(
      registerActionListener(canvas, "reset", (event) => {
        this.reset();
        event.preventDefault();
      }),
    );
    const resizeObserver = new ResizeObserver(() => this.debouncedUpdateView());
    resizeObserver.observe(canvas);
    this.registerDisposer(() => resizeObserver.disconnect());
    this.updateView();
  }

  getWheelMoveValue(event: WheelEvent) {
    return event.deltaY;
  }

  reset() {
    this.hoverTarget.value = undefined;
    this.target.reset();
  }

  /**
   * The bins shown, [start, end): the levels' range and the target, with a
   * margin, so the levels spread across the width.  Kept while what is
   * drawn stays inside, so it does not jump about.
   */
  binWindow: [number, number] = [0, 0];

  /** The last levels drawn, kept to show (greyed) while the layer is hidden. */
  private lastData: Uint32Array | undefined;
  private lastScales: Map<number, number> | undefined;

  private updateBinWindow(lo: number, hi: number) {
    const { numBins } = this.histogram;
    const [start, end] = this.binWindow;
    const span = Math.max(2, hi - lo);
    const margin = Math.max(1, span * 0.25);
    const wanted: [number, number] = [
      Math.max(0, lo - margin),
      Math.min(numBins, hi + margin),
    ];
    const inside = lo >= start && hi <= end;
    const tooWide = end - start > (wanted[1] - wanted[0]) * 2;
    if (!inside || tooWide) this.binWindow = wanted;
  }

  /**
   * Draws the levels by resolution (see `byResolution`); false when there
   * are no levels with a finite spatial scale or no pixel size yet, for the
   * render-scale layout to draw instead.
   */
  private updateViewByResolution(): boolean {
    const { ctx, canvas, histogram } = this;
    const { numBins, spatialScales, value: data, pixelSize } = histogram;
    const visible = histogram.visibility.visible;
    // Present and not-present chunks per level, this frame.
    const counts = new Map<number, [number, number]>();
    if (visible) {
      for (const [scale, row] of spatialScales) {
        let present = 0;
        let notPresent = 0;
        for (let bin = 0; bin < numBins; ++bin) {
          present += data[row * numBins * 2 + bin];
          notPresent += data[row * numBins * 2 + numBins + bin];
        }
        counts.set(scale, [present, notPresent]);
      }
    }
    const levels = Array.from(histogram.levels).sort((a, b) => a - b);
    if (levels.length === 0 || pixelSize === undefined) return false;

    const width = (canvas.width = canvas.offsetWidth);
    const height = (canvas.height = canvas.offsetHeight);
    ctx.clearRect(0, 0, width, height);
    const targetValue = this.target.value;
    const hoverValue = this.hoverTarget.value;

    // The axis: the levels' range, with half a level's step each side.
    const logs = levels.map((s) => Math.log2(s));
    const step =
      levels.length > 1
        ? (logs[logs.length - 1] - logs[0]) / (levels.length - 1)
        : 1;
    const lo = logs[0] - Math.max(step, 0.5) / 2;
    const hi = logs[logs.length - 1] + Math.max(step, 0.5) / 2;
    this.resolutionAxis = { lo, hi, pixelSize };
    const toX = (log: number) => ((log - lo) / (hi - lo)) * width;
    const barWidth = Math.max(3, Math.min(24, (width / levels.length) * 0.6));

    let maxCount = 1;
    let totalPresent = 0;
    let totalNotPresent = 0;
    for (const [scale, [present, notPresent]] of counts) {
      if (!Number.isFinite(scale)) continue;
      maxCount = Math.max(maxCount, present + notPresent);
      totalPresent += present;
      totalNotPresent += notPresent;
    }
    const yScale = (height - 2) / Math.log(1 + maxCount);

    // The level under the mouse.
    let hoverLevel: number | undefined;
    if (hoverValue !== undefined) {
      const x = toX(Math.log2(hoverValue[0] * pixelSize));
      let best = barWidth;
      for (let i = 0; i < levels.length; ++i) {
        const d = Math.abs(toX(logs[i]) - x);
        if (d <= best) {
          best = d;
          hoverLevel = levels[i];
        }
      }
    }

    const tempColor = vec3.create();
    for (let i = 0; i < levels.length; ++i) {
      const level = levels[i];
      const [present, notPresent] = counts.get(level) ?? [0, 0];
      const total = present + notPresent;
      const x = Math.round(toX(logs[i]) - barWidth / 2);
      if (total === 0) {
        // A level not drawn now: a stub, so every level shows.
        ctx.fillStyle = level === hoverLevel ? "#888" : "#555";
        ctx.fillRect(x, height - 3, Math.round(barWidth), 3);
        continue;
      }
      const hue = (((Math.log2(level) * 0.1) % 1) + 1) % 1;
      const saturation = level === hoverLevel ? 0.5 : 1;
      const top = height - Math.log(1 + total) * yScale;
      const split = top + (notPresent / total) * (height - top);
      hsvToRgb(tempColor, hue, saturation, 0.5);
      ctx.fillStyle = serializeColor(tempColor);
      ctx.fillRect(
        x,
        Math.round(top),
        Math.round(barWidth),
        Math.round(split - top),
      );
      hsvToRgb(tempColor, hue, saturation, 1);
      ctx.fillStyle = serializeColor(tempColor);
      ctx.fillRect(
        x,
        Math.round(split),
        Math.round(barWidth),
        height - Math.round(split),
      );
    }

    const line = (value: number, colour: string) => {
      const x = Math.min(
        width - 1,
        Math.max(0, toX(Math.log2(value * pixelSize))),
      );
      ctx.fillStyle = colour;
      ctx.fillRect(Math.floor(x), 0, 1, height);
    };
    line(targetValue, "#fff");
    if (hoverValue !== undefined) line(hoverValue[0], "#888");

    if (!visible) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("not drawn", width / 2, height / 2);
    }

    const shown = hoverValue === undefined ? targetValue : hoverValue[0];
    this.legendRenderScale.textContent = `${formatPixelNumber(shown)} ${this.unitOfTarget}`;
    const spatial = hoverLevel ?? shown * pixelSize;
    this.legendSpatialScale.textContent = formatScaleWithUnitAsString(
      spatial,
      "m",
      {
        precision: 2,
        elide1: false,
      },
    );
    const [present, notPresent] =
      hoverLevel !== undefined
        ? (counts.get(hoverLevel) ?? [0, 0])
        : [totalPresent, totalNotPresent - histogram.fakeChunkCount];
    this.legendChunks.textContent = `${present}/${present + notPresent}`;
    return true;
  }

  /**
   * Whether the levels are laid out by resolution: each level (spatial
   * scale) keeps its place across the width however the view zooms, every
   * level is shown (those not drawn now as stubs), and the target (the
   * resolution the view asks for: the target render scale times the size of
   * a screen pixel) is the line that moves.  Otherwise, as Neuroglancer
   * draws it, by render scale.
   */
  protected byResolution = true;

  /** The resolution layout's axis: log2 spatial scale at the left and right edges, and the pixel size. */
  private resolutionAxis:
    | { lo: number; hi: number; pixelSize: number }
    | undefined;

  updateView() {
    if (this.byResolution && this.updateViewByResolution()) return;
    this.resolutionAxis = undefined;
    const { ctx } = this;
    const { canvas } = this;
    const width = (canvas.width = canvas.offsetWidth);
    const height = (canvas.height = canvas.offsetHeight);
    const targetValue = this.target.value;
    const hoverValue = this.hoverTarget.value;

    {
      const { legendRenderScale } = this;
      const value = hoverValue === undefined ? targetValue : hoverValue[0];
      const valueString = formatPixelNumber(value);
      legendRenderScale.textContent = valueString + " " + this.unitOfTarget;
    }

    const { numBins } = this.histogram;

    ctx.clearRect(0, 0, width, height);

    const { histogram } = this;
    // histogram.begin(this.frameNumberCounter.frameNumber);
    let { value: histogramData, spatialScales } = histogram;

    if (!histogram.visibility.visible) {
      histogramData.fill(0);
    }

    // Which bins hold anything; while none do (the layer is hidden or not
    // drawn), the last levels seen are shown greyed, and the target can
    // still be set against them.
    let lo = Infinity;
    let hi = -Infinity;
    const occupied = (data: Uint32Array, rows: number) => {
      lo = Infinity;
      hi = -Infinity;
      for (let bin = 0; bin < numBins; ++bin) {
        for (let row = 0; row < rows; ++row) {
          const index = row * numBins * 2 + bin;
          if (data[index] + data[index + numBins] > 0) {
            lo = Math.min(lo, bin);
            hi = Math.max(hi, bin + 1);
            break;
          }
        }
      }
    };
    occupied(histogramData, spatialScales.size);
    const inactive = lo === Infinity;
    if (!inactive) {
      this.lastData = histogramData.slice();
      this.lastScales = new Map(spatialScales);
    } else if (this.lastData !== undefined && this.lastScales !== undefined) {
      histogramData = this.lastData;
      spatialScales = this.lastScales;
      occupied(histogramData, spatialScales.size);
    }
    const targetBin = getRenderScaleHistogramOffset(
      targetValue,
      this.histogram.logScaleOrigin,
      this.histogram.binSize,
    );
    if (lo === Infinity) {
      lo = targetBin - 2;
      hi = targetBin + 2;
    }
    this.updateBinWindow(Math.min(lo, targetBin), Math.max(hi, targetBin + 1));
    const [windowStart, windowEnd] = this.binWindow;

    function binToCanvasX(bin: number) {
      return ((bin - windowStart) * width) / (windowEnd - windowStart);
    }

    const sortedSpatialScales = Array.from(spatialScales.keys());
    sortedSpatialScales.sort();

    const tempColor = vec3.create();

    let maxCount = 1;
    const numRows = spatialScales.size;
    let totalPresent = 0;
    let totalNotPresent = 0;
    for (let bin = 0; bin < numBins; ++bin) {
      let count = 0;
      for (let row = 0; row < numRows; ++row) {
        const index = row * numBins * 2 + bin;
        const presentCount = histogramData[index];
        const notPresentCount = histogramData[index + numBins];
        totalPresent += presentCount;
        totalNotPresent += notPresentCount;
        count += presentCount + notPresentCount;
      }
      maxCount = Math.max(count, maxCount);
    }
    totalNotPresent -= histogram.fakeChunkCount;

    const maxBarHeight = height;

    const yScale = maxBarHeight / Math.log(1 + maxCount);

    function countToCanvasY(count: number) {
      return height - Math.log(1 + count) * yScale;
    }

    let hoverSpatialScale: number | undefined = undefined;
    if (hoverValue !== undefined) {
      const i = Math.floor(
        getRenderScaleHistogramOffset(
          hoverValue[0],
          this.histogram.logScaleOrigin,
          this.histogram.binSize,
        ),
      );
      if (i >= 0 && i < numBins) {
        let sum = 0;
        const hoverY = hoverValue[1];
        for (
          let spatialScaleIndex = numRows - 1;
          spatialScaleIndex >= 0;
          --spatialScaleIndex
        ) {
          const spatialScale = sortedSpatialScales[spatialScaleIndex];
          const row = spatialScales.get(spatialScale)!;
          const index = 2 * row * numBins + i;
          const count = histogramData[index] + histogramData[index + numBins];
          if (count === 0) continue;
          const yStart = Math.round(countToCanvasY(sum));
          sum += count;
          const yEnd = Math.round(countToCanvasY(sum));
          if (yEnd <= hoverY && hoverY <= yStart) {
            hoverSpatialScale = spatialScale;
            break;
          }
        }
      }
    }
    if (hoverSpatialScale !== undefined) {
      totalPresent = 0;
      totalNotPresent = 0;
      const row = spatialScales.get(hoverSpatialScale)!;
      const baseIndex = 2 * row * numBins;
      for (let bin = 0; bin < numBins; ++bin) {
        const index = baseIndex + bin;
        totalPresent += histogramData[index];
        totalNotPresent += histogramData[index + numBins];
      }
      if (Number.isFinite(hoverSpatialScale)) {
        this.legendSpatialScale.textContent = formatScaleWithUnitAsString(
          hoverSpatialScale,
          "m",
          { precision: 2, elide1: false },
        );
      } else {
        this.legendSpatialScale.textContent = "unknown";
      }
    } else {
      this.legendSpatialScale.textContent = "";
    }

    this.legendChunks.textContent = `${totalPresent}/${
      totalPresent + totalNotPresent
    }`;

    const spatialScaleColors = sortedSpatialScales.map((spatialScale) => {
      const saturation = inactive
        ? 0
        : spatialScale === hoverSpatialScale
          ? 0.5
          : 1;
      let hue;
      if (Number.isFinite(spatialScale)) {
        hue = (((Math.log2(spatialScale) * 0.1) % 1) + 1) % 1;
      } else {
        hue = 0;
      }
      hsvToRgb(tempColor, hue, saturation, 1);
      const presentColor = serializeColor(tempColor);
      hsvToRgb(tempColor, hue, saturation, 0.5);
      const notPresentColor = serializeColor(tempColor);
      return [presentColor, notPresentColor];
    });

    for (let i = 0; i < numBins; ++i) {
      let sum = 0;
      for (
        let spatialScaleIndex = numRows - 1;
        spatialScaleIndex >= 0;
        --spatialScaleIndex
      ) {
        const spatialScale = sortedSpatialScales[spatialScaleIndex];
        const row = spatialScales.get(spatialScale)!;
        const index = row * numBins * 2 + i;
        const presentCount = histogramData[index];
        const notPresentCount = histogramData[index + numBins];
        const count = presentCount + notPresentCount;
        if (count === 0) continue;
        const xStart = Math.round(binToCanvasX(i));
        const xEnd = Math.round(binToCanvasX(i + 1));
        const yStart = Math.round(countToCanvasY(sum));
        sum += count;
        const yEnd = Math.round(countToCanvasY(sum));
        const ySplit = (presentCount * yEnd + notPresentCount * yStart) / count;
        ctx.fillStyle = spatialScaleColors[spatialScaleIndex][1];
        ctx.fillRect(xStart, yEnd, xEnd - xStart, ySplit - yEnd);
        ctx.fillStyle = spatialScaleColors[spatialScaleIndex][0];
        ctx.fillRect(xStart, ySplit, xEnd - xStart, yStart - ySplit);
      }
    }

    if (inactive) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        this.lastData === undefined ? "inactive: not drawn" : "inactive",
        width / 2,
        height / 2,
      );
    }

    {
      const value = targetValue;
      ctx.fillStyle = "#fff";
      const startOffset = binToCanvasX(
        getRenderScaleHistogramOffset(
          value,
          this.histogram.logScaleOrigin,
          this.histogram.binSize,
        ),
      );
      const lineWidth = 1;
      ctx.fillRect(Math.floor(startOffset), 0, lineWidth, height);
    }

    if (hoverValue !== undefined) {
      const value = hoverValue[0];
      ctx.fillStyle = "#888";
      const startOffset = binToCanvasX(
        getRenderScaleHistogramOffset(
          value,
          this.histogram.logScaleOrigin,
          this.histogram.binSize,
        ),
      );
      const lineWidth = 1;
      ctx.fillRect(Math.floor(startOffset), 0, lineWidth, height);
    }
  }
}

export class VolumeRenderingRenderScaleWidget extends RenderScaleWidget {
  protected unitOfTarget = "samples";
  protected byResolution = false;

  getWheelMoveValue(event: WheelEvent) {
    return -event.deltaY;
  }
}

const TOOL_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:shift+wheel": { action: "adjust-via-wheel" },
  "at:shift+dblclick0": { action: "reset" },
});

export function renderScaleLayerControl<
  LayerType extends UserLayer,
  WidgetType extends RenderScaleWidget,
>(
  getter: (layer: LayerType) => RenderScaleWidgetOptions,
  widgetClass: new (
    histogram: RenderScaleHistogram,
    target: TrackableValueInterface<number>,
  ) => WidgetType = RenderScaleWidget as new (
    histogram: RenderScaleHistogram,
    target: TrackableValueInterface<number>,
  ) => WidgetType,
): LayerControlFactory<LayerType, RenderScaleWidget> {
  return {
    makeControl: (layer, context) => {
      const { histogram, target } = getter(layer);
      const control = context.registerDisposer(
        new widgetClass(histogram, target),
      );
      return { control, controlElement: control.element };
    },
    activateTool: (activation, control) => {
      activation.bindInputEventMap(TOOL_INPUT_EVENT_MAP);
      activation.bindAction(
        "adjust-via-wheel",
        (event: ActionEvent<WheelEvent>) => {
          event.stopPropagation();
          event.preventDefault();
          control.adjustViaWheel(event.detail);
        },
      );
      activation.bindAction("reset", (event: ActionEvent<WheelEvent>) => {
        event.stopPropagation();
        event.preventDefault();
        control.reset();
      });
    },
  };
}
