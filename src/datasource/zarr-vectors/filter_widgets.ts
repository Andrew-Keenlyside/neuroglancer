/**
 * @license
 * Copyright 2026 Google Inc.
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

/**
 * Small DOM helpers shared by the zarr-vectors Filter and Export tabs.
 *
 * Extracted so the two tabs cannot drift into subtly different field markup:
 * both style their rows off `neuroglancer-streamline-filter-field`, so a change
 * to one tab's labelling would otherwise silently misalign the other.
 */

/** Wrap a control in a `<label>` with leading text. */
export function labelled(
  text: string,
  control: HTMLElement,
  className?: string,
): HTMLElement {
  const label = document.createElement("label");
  label.classList.add("neuroglancer-streamline-filter-field");
  if (className !== undefined) label.classList.add(className);
  const span = document.createElement("span");
  span.textContent = text;
  label.appendChild(span);
  label.appendChild(control);
  return label;
}

/**
 * A `<select>` over numeric-valued options.
 *
 * Numeric because every caller is backed by a `const enum` (`RoiOperator`,
 * `RoiPredicate`), whose values survive a round trip through the DOM's string
 * `value` only via `Number`.
 */
export function makeSelect<T extends number>(
  options: { value: T; label: string }[],
  current: T,
  onChange: (v: T) => void,
): HTMLSelectElement {
  const select = document.createElement("select");
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = String(opt.value);
    el.textContent = opt.label;
    if (opt.value === current) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => onChange(Number(select.value) as T));
  return select;
}

/** The same, over string-valued options (formats, destinations). */
export function makeStringSelect(
  options: { value: string; label: string }[],
  current: string,
  onChange: (v: string) => void,
): HTMLSelectElement {
  const select = document.createElement("select");
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.label;
    if (opt.value === current) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => onChange(select.value));
  return select;
}
