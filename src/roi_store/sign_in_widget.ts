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
 * @file Top-row chip showing who is signed in to the ROI group store.
 *
 * Sign-in is only needed to SAVE a group — browsing and loading are anonymous
 * — so this is a status affordance and an escape hatch for switching account,
 * not a gate the user must pass before using the viewer.
 */

import type { RoiStoreAuth } from "#src/roi_store/credentials.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren } from "#src/util/dom.js";
import { makeIcon } from "#src/widget/icon.js";

export class RoiStoreSignInWidget extends RefCounted {
  element = document.createElement("div");

  constructor(private auth: RoiStoreAuth) {
    super();
    this.element.style.display = "flex";
    this.element.style.alignItems = "center";
    this.element.style.marginRight = "5px";
    this.registerDisposer(this.auth.changed.add(() => this.updateView()));
    this.updateView();
  }

  private updateView() {
    removeChildren(this.element);
    // Keyed on `signedIn`, not `email`: the middleauth provider is signed in
    // without exposing an email, so an email test would show it as signed out.
    if (!this.auth.signedIn) {
      this.element.appendChild(
        makeIcon({
          text: "Sign in",
          title: "Sign in to save ROI groups to the shared store",
          onClick: () => {
            // getCredentialsWithStatus drives its own status-bar UI, including
            // the failure message and retry button. Still log: if signIn throws
            // before reaching that UI (e.g. no credentials provider, or a
            // blocked popup) the click would otherwise look like a no-op.
            this.auth.signIn().catch((e) => {
              console.error("[roi-store] sign-in failed:", e);
            });
          },
        }),
      );
      return;
    }

    const { email } = this.auth;
    const label = document.createElement("span");
    label.textContent = email ?? "Signed in";
    label.title =
      email === undefined
        ? "Signed in to the ROI group store"
        : `Signed in to the ROI group store as ${email}`;
    label.style.fontFamily = "sans-serif";
    label.style.fontSize = "10pt";
    label.style.marginRight = "5px";
    this.element.appendChild(label);

    this.element.appendChild(
      makeIcon({
        text: "Sign out",
        title: "Forget this ROI group store sign-in",
        onClick: () => this.auth.signOut(),
      }),
    );
  }

  disposed() {
    this.element.remove();
    super.disposed();
  }
}
