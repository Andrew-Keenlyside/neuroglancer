/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * The `#attributes=a,b,c` URL fragment, which names the per-vertex attributes a
 * zarr-vectors layer should expose.
 *
 * Parsing and formatting live together, and in their own module, because they
 * have to be exact inverses: the formatted URL is what gets saved into a layer's
 * JSON, so a name that does not survive the round trip turns a saved link into a
 * load error the next time it is opened.
 */

const PREFIX = "attributes=";

/**
 * Parse the fragment's attribute list, or `undefined` when there is no
 * fragment. Throws when the fragment is something else entirely -- silently
 * ignoring it would drop a selection the user asked for.
 */
export function parseAttributesFragment(
  fragment: string | undefined,
): string[] | undefined {
  if (!fragment) return undefined;
  if (!fragment.startsWith(PREFIX)) {
    throw new Error(
      "the only supported fragment is `#attributes=<comma-separated names>`",
    );
  }
  // Split BEFORE decoding: a name containing a percent-encoded comma (`%2C`)
  // is one attribute, not two, and decoding first would split it.
  return fragment
    .slice(PREFIX.length)
    .split(",")
    .map((part) => {
      const trimmed = part.trim();
      try {
        return decodeURIComponent(trimmed);
      } catch {
        // A stray `%` is not an escape; take the name literally rather than
        // failing the whole load over it.
        return trimmed;
      }
    })
    .filter((name) => name.length > 0);
}

/** Format an attribute list back into a fragment `parseAttributesFragment` reads. */
export function formatAttributesFragment(
  attributes: readonly string[] | undefined,
): string {
  if (attributes === undefined) return "";
  return `#${PREFIX}${attributes.map((n) => encodeURIComponent(n)).join(",")}`;
}
