/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * @file The editor for ONE attribute predicate, shared by the Filter tab's
 * per-group list and the "By attribute" staging panel.
 *
 * Shared because the two are the same control in two places: what a staged
 * predicate does to the live preview is exactly what a committed one does to
 * its group, and a user who learns the control in one place must not meet a
 * different one in the other.
 *
 * Which control it is depends on the DATA, not on the predicate: a two-valued
 * integer column ({@link isFlagAttr}) gets a checkbox, anything else gets a
 * min/max pair. An attribute nobody has measured yet gets a placeholder and a
 * measurement request -- see {@link AttrStatsCache} for why measuring is lazy.
 */

import type { AttrStats } from "#src/datasource/zarr-vectors/attribute_catalog.js";
import {
  flagFilter,
  flagFilterValue,
  isFlagAttr,
} from "#src/datasource/zarr-vectors/attribute_catalog.js";
import {
  fieldWatchable,
  labelled,
} from "#src/datasource/zarr-vectors/filter_widgets.js";
import type { RoiAttrFilter } from "#src/datasource/zarr-vectors/roi.js";
import type { RefCounted } from "#src/util/disposable.js";
import type { NullarySignal } from "#src/util/signal.js";
import { makeDeleteButton } from "#src/widget/delete_button.js";
import { RangeWidget } from "#src/widget/range.js";

/** How many slider steps to divide an attribute's observed range into. */
const RANGE_STEPS = 200;

export interface AttrFilterControlOptions {
  /** The predicate as of build time; only its name/scope are read directly. */
  readonly filter: RoiAttrFilter;
  /** Its measured distribution, or `undefined` while unmeasured. */
  readonly stats: AttrStats | undefined;
  /** Owns the widgets' disposers (the caller's per-rebuild context). */
  readonly context: RefCounted;
  /** Fires whenever the underlying store changes; drives widget re-reads. */
  readonly changed: NullarySignal;
  /** Live read of the current predicate (never a captured copy — see `fieldWatchable`). */
  readonly read: () => RoiAttrFilter;
  /** Commit an edited predicate. */
  readonly write: (filter: RoiAttrFilter) => void;
  /** Remove this predicate entirely; omit to render no delete button. */
  readonly remove?: () => void;
  /** Ask for the measurement that is missing (called only when `stats` is undefined). */
  readonly requestStats?: () => void;
}

/**
 * Build one predicate's editor.
 *
 * A flag is one line -- `<name> is [x]` -- because that is the whole control. A
 * range needs three: the two bounds are each a slider plus a number box, and
 * side by side in a side panel the upper one is squeezed to nothing, so the name
 * takes a line of its own and each bound gets a full-width line under it.
 */
export function makeAttrFilterControl(
  options: AttrFilterControlOptions,
): HTMLElement {
  const { filter, stats, context, changed, read, write } = options;
  const row = document.createElement("div");
  row.classList.add("neuroglancer-streamline-filter-attr-row");

  const remove = options.remove;
  const deleteButton =
    remove === undefined
      ? undefined
      : makeDeleteButton({
          title: `Remove the ${filter.name} filter`,
          onClick: () => remove(),
        });

  // The one-line variants keep their control and delete button on the same
  // line; only a range's bounds claim a line each.
  const inline = () =>
    row.classList.add("neuroglancer-streamline-filter-attr-row-inline");

  if (stats === undefined) {
    options.requestStats?.();
    const measuring = document.createElement("span");
    measuring.textContent = "measuring…";
    inline();
    row.appendChild(labelled(filter.name, measuring));
    if (deleteButton !== undefined) row.appendChild(deleteButton);
    return row;
  }

  if (isFlagAttr(stats)) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = flagFilterValue(read(), stats);
    context.registerEventListener(checkbox, "change", () => {
      write(flagFilter(stats, checkbox.checked));
    });
    inline();
    row.appendChild(labelled(`${filter.name} is`, checkbox));
    if (deleteButton !== undefined) row.appendChild(deleteButton);
    return row;
  }

  const head = document.createElement("div");
  head.classList.add("neuroglancer-streamline-filter-attr-head");
  const name = document.createElement("span");
  name.classList.add("neuroglancer-streamline-filter-attr-name");
  name.textContent = filter.name;
  head.appendChild(name);
  if (deleteButton !== undefined) head.appendChild(deleteButton);
  row.appendChild(head);

  const span = stats.max - stats.min;
  const step = span > 0 ? Math.max(span / RANGE_STEPS, 1e-6) : 1;
  const set = (min: number, max: number) => write({ ...read(), min, max });
  const lo = context.registerDisposer(
    new RangeWidget(
      fieldWatchable(
        changed,
        () => read().min,
        (v) => set(v, read().max),
      ),
      { min: stats.min, max: stats.max, step },
    ),
  );
  const hi = context.registerDisposer(
    new RangeWidget(
      fieldWatchable(
        changed,
        () => read().max,
        (v) => set(read().min, v),
      ),
      { min: stats.min, max: stats.max, step },
    ),
  );
  row.appendChild(labelled("≥", lo.element));
  row.appendChild(labelled("≤", hi.element));
  return row;
}
