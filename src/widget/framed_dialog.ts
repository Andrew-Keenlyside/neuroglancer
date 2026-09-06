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
 * An Overlay with a titled header, a scrolling body and a footer -- the frame
 * the fork's dialogs share. Kept out of overlay.ts so that file stays identical
 * to upstream.
 */

import svg_close from "ikonate/icons/close.svg?raw";
import { Overlay } from "#src/overlay.js";
import "#src/widget/framed_dialog.css";
import { makeIcon } from "#src/widget/icon.js";

export class FramedDialog extends Overlay {
  header: HTMLDivElement;
  headerTitle: HTMLSpanElement;
  closeMenuIcon: HTMLElement;
  closeButton: HTMLButtonElement;
  body: HTMLDivElement;
  footer: HTMLDivElement;
  constructor(
    title: string = "Dialog",
    closeText: string = "Close",
    extraClassPrefix?: string,
  ) {
    super();
    this.content.classList.add("neuroglancer-framed-dialog");

    const header = (this.header = document.createElement("div"));
    const closeMenuIcon = (this.closeMenuIcon = makeIcon({ svg: svg_close }));
    closeMenuIcon.addEventListener("click", () => this.close());
    closeMenuIcon.classList.add("neuroglancer-framed-dialog-close-icon");
    const headerTitle = (this.headerTitle = document.createElement("span"));
    headerTitle.textContent = title;
    headerTitle.classList.add("neuroglancer-framed-dialog-title");
    header.classList.add("neuroglancer-framed-dialog-header");
    header.appendChild(headerTitle);
    header.appendChild(closeMenuIcon);
    this.content.appendChild(header);

    const body = (this.body = document.createElement("div"));
    body.classList.add("neuroglancer-framed-dialog-body");
    this.content.appendChild(body);

    const footer = (this.footer = document.createElement("div"));
    footer.classList.add("neuroglancer-framed-dialog-footer");
    const closeFooterButton = (this.closeButton =
      document.createElement("button"));
    closeFooterButton.textContent = closeText;
    closeFooterButton.classList.add("neuroglancer-framed-dialog-close-button");
    closeFooterButton.addEventListener("click", () => this.close());
    footer.appendChild(closeFooterButton);
    this.content.appendChild(this.footer);

    if (extraClassPrefix !== undefined) {
      this.content.classList.add(`${extraClassPrefix}`);
      this.header.classList.add(`${extraClassPrefix}-header`);
      this.headerTitle.classList.add(`${extraClassPrefix}-title`);
      this.closeMenuIcon.classList.add(`${extraClassPrefix}-close-icon`);
      this.body.classList.add(`${extraClassPrefix}-body`);
      this.footer.classList.add(`${extraClassPrefix}-footer`);
      this.closeButton.classList.add(`${extraClassPrefix}-close-button`);
    }
  }
}
