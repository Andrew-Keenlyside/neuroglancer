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
 * @file The "Guide" tab: how to use the ROI streamline filter.
 *
 * Sits beside the Filter tab and appears under the same condition, so the
 * explanation is one click from the controls it describes.  Content is static
 * prose built from a small declarative structure — the point is that adding a
 * section means adding an entry to `SECTIONS`, not writing DOM code.
 *
 * Terms here must match the on-screen labels in `streamline_filter_tab.ts`
 * exactly; a guide that renames things is worse than no guide.
 */

import "#src/datasource/zarr-vectors/streamline_guide_tab.css";

import { roiStoreEnabled } from "#src/roi_store/config.js";
import { Tab } from "#src/widget/tab_view.js";

interface GuideSection {
  heading: string;
  /** Paragraphs of prose. */
  body?: string[];
  /** Ordered steps, rendered as a numbered list. */
  steps?: string[];
  /** Term/definition pairs, for the controls in a panel. */
  terms?: { term: string; definition: string }[];
  /** Omit the section entirely unless this is true. */
  when?: boolean;
}

const SECTIONS: GuideSection[] = [
  {
    heading: "What a group is",
    body: [
      "A GROUP is one named, coloured dissection: an ordered list of regions " +
        "that a streamline is tested against. A tract belongs to the group if " +
        "it passes that group's regions. Passing tracts are drawn in the " +
        "group's colour; everything else is ghosted back.",
      "Groups are independent. Several can be active at once, each with its " +
        "own colour, visibility and opacity, so you can compare dissections " +
        "side by side in one view.",
    ],
  },
  {
    heading: "Making a dissection",
    steps: [
      "Turn on Active at the top of the Filter tab. Nothing is filtered until " +
        "it is on.",
      "Press + New group, then name it and pick a colour.",
      "Add a region with + Sphere, + Box or + Plane…, and position it with the " +
        "centre and size sliders. Regions are placed from the sliders, not by " +
        "dragging in the viewport.",
      "Set how each region combines with the ones above it: Include, Or or " +
        "Exclude.",
      "Add more regions to narrow the dissection. The tract count under the " +
        "panel updates as you go.",
    ],
  },
  {
    heading: "How regions combine",
    body: [
      "Regions are evaluated top to bottom, each applied to the verdict so " +
        "far. The first region in the list starts the verdict, so its own " +
        "setting has no effect.",
    ],
    terms: [
      {
        term: "Include",
        definition:
          "Keep only tracts that also pass this region. This is how you " +
          "narrow a bundle.",
      },
      {
        term: "Or",
        definition:
          "Also keep tracts that pass this region, widening the selection.",
      },
      {
        term: "Exclude",
        definition:
          "Drop tracts that pass this region — the standard way to remove a " +
          "neighbouring bundle that keeps contaminating the result.",
      },
    ],
  },
  {
    heading: "Crosses vs Point inside",
    body: [
      "Each region also chooses what counts as passing. This matters more " +
        "than it looks.",
    ],
    terms: [
      {
        term: "Crosses",
        definition:
          "The tract's path enters the region. Tested against the line " +
          "itself, so it does not depend on how finely the tract is sampled. " +
          "Prefer this.",
      },
      {
        term: "Point inside",
        definition:
          "One of the tract's stored points lies inside the region. A tract " +
          "can step straight over a small region without landing a point in " +
          "it and be missed. Offered for consistency with results from tools " +
          "that test this way; enlarging the region to compensate is a " +
          "symptom, not a fix.",
      },
    ],
  },
  {
    heading: "Display",
    terms: [
      {
        term: "Colour by group",
        definition:
          "Recolour passing tracts with their group's colour. Turn it off to " +
          "keep the tractogram's own colouring and use the filter only to " +
          "select.",
      },
      {
        term: "Opacity",
        definition: "Per-group opacity for that group's passing tracts.",
      },
      {
        term: "High detail",
        definition:
          "Re-fetch this group's passing tracts at full resolution. Slower, " +
          "and worth turning on once a dissection is settled rather than " +
          "while you are still moving regions.",
      },
      {
        term: "Hide regions in 2-d",
        definition:
          "Stop drawing the region outlines in the cross-section views, when " +
          "they get in the way of the anatomy.",
      },
    ],
  },
  {
    heading: "Splitting a group into its own layer",
    body: [
      "⇗ moves a group into a sibling layer on the same data, which gives it " +
        "independent visibility, ordering and render controls. ⇙ moves it " +
        "back. The underlying tracts are not re-fetched — both layers share " +
        "the loaded data — and the dissection moves rather than being copied, " +
        "so the work does not double.",
      "+ New group as layer does the same thing for a fresh, empty group.",
    ],
  },
  {
    heading: "Saving and sharing",
    when: roiStoreEnabled,
    body: [
      "Save to store on a group writes it to a shared library, so it survives " +
        "closing the tab and can be opened by someone else. Saving the same " +
        "group again updates that entry rather than leaving a second copy.",
      "From store, at the top of this tab, lists the dissections already " +
        "saved for this dataset. Tick one to bring it in and show it; untick " +
        "to hide it again. The list refreshes itself whenever anything is " +
        "saved, and reads “none found” until the first one is.",
      "A ticked group behaves like any other from then on — its own colour, " +
        "opacity, high-detail toggle and ⇗ expand-to-layer all apply, and you " +
        "can edit its regions. Unticking only hides it, so those edits are " +
        "not lost; use the group's delete button to remove it outright.",
      "Browsing and loading need no sign-in. Only saving and deleting do — " +
        "the first save prompts for a Google account, and the top bar then " +
        "shows who you are signed in as.",
      "Region coordinates are stored in the tractogram's own space, so a " +
        "saved group only lines up against the data it was drawn on. From " +
        "store therefore lists only groups saved against this data source. " +
        "Browse saved… shows everything, flagging entries from elsewhere " +
        "without blocking them: the same tractography is often served from " +
        "more than one address, and only you can tell.",
      "Each saved group also records the viewer URL it was drawn in, so the " +
        "view can be recovered, and the account that saved it.",
    ],
  },
  {
    heading: "If the filter seems wrong",
    terms: [
      {
        term: "Nothing is shown",
        definition:
          "Check Active is on, that the group is visible, and that its first " +
          "region actually intersects the bundle — an Include chain starting " +
          "from an empty region can never recover.",
      },
      {
        term: "Too many tracts",
        definition:
          "Add an Exclude region over the bundle you want gone, rather than " +
          "shrinking the Include regions until the true bundle thins out too.",
      },
      {
        term: "Tracts flicker while navigating",
        definition:
          "Detail is streamed by view. Let the view settle, or turn on High " +
          "detail for the group you care about.",
      },
    ],
  },
];

export class StreamlineGuideTab extends Tab {
  constructor() {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-streamline-guide-tab");
    for (const section of SECTIONS) {
      if (section.when === false) continue;
      element.appendChild(makeSection(section));
    }
  }
}

function makeSection(section: GuideSection): HTMLElement {
  const el = document.createElement("section");
  el.classList.add("neuroglancer-streamline-guide-section");

  const heading = document.createElement("h3");
  heading.textContent = section.heading;
  el.appendChild(heading);

  for (const paragraph of section.body ?? []) {
    const p = document.createElement("p");
    p.textContent = paragraph;
    el.appendChild(p);
  }

  if (section.steps !== undefined) {
    const ol = document.createElement("ol");
    for (const step of section.steps) {
      const li = document.createElement("li");
      li.textContent = step;
      ol.appendChild(li);
    }
    el.appendChild(ol);
  }

  if (section.terms !== undefined) {
    const dl = document.createElement("dl");
    for (const { term, definition } of section.terms) {
      const dt = document.createElement("dt");
      dt.textContent = term;
      dl.appendChild(dt);
      const dd = document.createElement("dd");
      dd.textContent = definition;
      dl.appendChild(dd);
    }
    el.appendChild(dl);
  }
  return el;
}
