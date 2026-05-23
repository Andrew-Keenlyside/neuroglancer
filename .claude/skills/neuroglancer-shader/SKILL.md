---
name: neuroglancer-shader
description: Write custom GLSL shader code for Neuroglancer layers. Handles image/volume, annotation, skeleton, and mesh layers. Creates interactive visualizations with UI controls for brightness, contrast, colormaps, thresholds, and more. Use when asked to write, modify, or explain a Neuroglancer shader.
---

# Neuroglancer Shader Writing Guide

Neuroglancer renders all data using GLSL ES 3.0 fragment shaders. Each layer type exposes a different API. This guide covers how to write correct, interactive shaders for all layer types.

---

## Step 1: Gather Information

Before writing any code, ask the user for missing information from this list. Skip questions irrelevant to the layer type.

| Question | Why it matters |
|---|---|
| **Layer type**: image, annotation, skeleton, or single mesh? | Determines all available functions |
| **Data type**: uint8, uint16, uint32, int8, int16, int32, float32? | Affects which conversion functions work |
| **Number of channels**: single or multi-channel? How many? | Determines `getDataValue()` call patterns |
| **Value range**: typical min and max values? | Initializes `invlerp` range/window |
| **Visualization goal**: grayscale, colormap, false color, mask overlay, etc.? | Shapes the whole shader |
| **For annotations**: property names and their types/ranges? | Required for `prop_name()` calls |
| **For skeletons**: vertex attribute names and what they represent? | Required for `prop_name()` calls |
| **Volume rendering**: work for both 2D slices and 3D volume? | Needs `VOLUME_RENDERING` branch |
| **Colormap preference**: viridis, plasma, inferno, magma, coolwarm, RdBu, jet, cubehelix? | Determines colormap to embed |

---

## Step 2: Design Principles

**UI-first**: Every parameter the user might want to adjust must be a `#uicontrol`. Avoid hardcoded constants for:
- Brightness / contrast → use `invlerp` or `slider`
- Colormap range → use `invlerp` `range` and `window`
- Colors → use `color` picker
- Thresholds → use `slider`
- Channel selection → use `int slider`
- Opacity → use `float slider` or `invlerp`

**Use `invlerp` over `toNormalized`**: `invlerp` exposes a histogram UI the user can adjust interactively. `toNormalized` maps the full type range with no UI.

**Use `transferFunction` for complex opacity**: When mapping data to both color and opacity via control points, use `transferFunction`.

**Signed integer warning**: `int8_t`, `int16_t`, `int32_t` have **no `toNormalized()`**. Use `invlerp` (which handles them correctly), or convert manually: `float(toRaw(v)) / 32767.0`.

**Embed colormaps inline**: `colormapViridis`, `colormapPlasma`, etc. are NOT built-in. Copy the function from the Colormap Library section below into the shader before `void main()`.

---

## Layer Types

### 1. Image / Volume Layers

**Documentation**: `src/sliceview/image_layer_rendering.md`, `src/volume_rendering/README.md`

#### Data Access

```glsl
T getDataValue(int channelIndex...);             // nearest-neighbor
T getInterpolatedDataValue(int channelIndex...); // trilinear interpolation
```

`T` is one of: `float`, `uint8_t`, `uint16_t`, `uint32_t`, `int8_t`, `int16_t`, `int32_t`, `uint64_t`.

These are GLSL structs (not native types), so access their `.value` field only when needed:
```glsl
struct uint8_t  { highp uint value; };
struct uint16_t { highp uint value; };
struct uint32_t { highp uint value; };
struct int8_t   { highp int value; };
struct int16_t  { highp int value; };
struct int32_t  { highp int value; };
struct uint64_t { highp uvec2 value; };
// float is a native GLSL type - no .value needed
```

#### Type Conversion

```glsl
// Maps full integer range to [0, 1]. ONLY for unsigned types and float:
float toNormalized(float x)    // identity
float toNormalized(uint8_t x)  // divides by 255
float toNormalized(uint16_t x) // divides by 65535
float toNormalized(uint32_t x) // divides by 4294967295

// Returns raw integer value. Works for ALL integer types:
highp uint toRaw(uint8_t x)
highp uint toRaw(uint16_t x)
highp uint toRaw(uint32_t x)
highp int  toRaw(int8_t x)   // WARNING: no toNormalized for signed types
highp int  toRaw(int16_t x)  // WARNING: no toNormalized for signed types
highp int  toRaw(int32_t x)  // WARNING: no toNormalized for signed types
float      toRaw(float x)    // identity
```

**Signed integer normalization** — use `invlerp` (best) or manual:
```glsl
// Manual int16 normalization to [0,1]:
float v = float(toRaw(getDataValue())) / 32767.0 * 0.5 + 0.5;
```

#### Emit Functions

```glsl
void emitGrayscale(float x);     // x in [0,1] → grayscale
void emitRGB(vec3 rgb);          // components in [0,1]
void emitRGBA(vec4 rgba);        // alpha multiplied by layer's opacity setting
void emitTransparent();          // fully transparent pixel
void emitIntensity(float x);     // MAX/MIN projection intensity (no-op in slice view)
```

#### Volume Rendering

```glsl
// Compile-time constant: false in 2D slice view, true in 3D volume mode
#define VOLUME_RENDERING false  // or true

// In volume rendering mode, alpha is corrected for sampling density.
// Use small per-step alpha (0.005–0.05). Do not use 1.0 in volume mode.
if (VOLUME_RENDERING) {
  emitRGBA(vec4(color, 0.02));  // small alpha accumulates across ray
} else {
  emitRGB(color);
}

// For MAX/MIN projection, explicitly set intensity (defaults to last invlerp called):
emitIntensity(v);  // float in [0,1]
```

#### Built-in Colormaps (always available in image layers)

```glsl
vec3 colormapJet(float x)        // classic rainbow (not perceptually uniform)
vec3 colormapCubehelix(float x)  // perceptually uniform spiral, dark to bright
```

#### Default Shader

```glsl
#uicontrol invlerp normalized
void main() {
  emitGrayscale(normalized());
}
```

#### Image Layer Examples

**Grayscale with histogram control** (default):
```glsl
#uicontrol invlerp normalized
void main() {
  emitGrayscale(normalized());
}
```

**Viridis colormap with adjustable range**:
```glsl
#uicontrol invlerp normalized(range=[0, 1000], window=[0, 4000])

// paste colormapViridis from Colormap Library below

void main() {
  emitRGB(colormapViridis(normalized()));
}
```

**Multi-channel RGB composite** (three channels → R, G, B):
```glsl
#uicontrol invlerp red(channel=0)
#uicontrol invlerp green(channel=1)
#uicontrol invlerp blue(channel=2)
void main() {
  emitRGB(vec3(red(), green(), blue()));
}
```

**Probability / mask overlay** (single channel as semitransparent color):
```glsl
#uicontrol invlerp alpha(range=[0, 255])
#uicontrol vec3 color color(default="red")
void main() {
  emitRGBA(vec4(color, alpha()));
}
```

**Threshold with transparent background**:
```glsl
#uicontrol invlerp normalized
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.5, step=0.01)
#uicontrol vec3 color color(default="cyan")
void main() {
  float v = normalized();
  if (v < threshold) {
    emitTransparent();
  } else {
    emitRGB(color * v);
  }
}
```

**Per-value categorical coloring**:
```glsl
#uicontrol vec3 color0 color(default="gray")
#uicontrol vec3 color1 color(default="red")
#uicontrol vec3 color2 color(default="green")
#uicontrol vec3 color3 color(default="blue")
void main() {
  float v = toRaw(getDataValue());
  vec3 color = color0;
  if (v == 1.0) color = color1;
  if (v == 2.0) color = color2;
  if (v == 3.0) color = color3;
  emitRGB(color);
}
```

**Transfer function** (full color + opacity control via UI):
```glsl
#uicontrol transferFunction tf(
  window=[0, 65535],
  controlPoints=[[0.0, "#000000", 0.0], [32768.0, "#0055ff", 0.5], [65535.0, "#ffffff", 1.0]],
  channel=0,
  defaultColor="#ffffff"
)
void main() {
  emitRGBA(tf());
}
```

**Volume rendering with opacity** (works in both 2D and 3D):
```glsl
#uicontrol invlerp normalized
#uicontrol float opacity slider(min=0.0, max=0.1, default=0.02, step=0.001)

// paste colormapViridis here

void main() {
  float v = normalized();
  if (VOLUME_RENDERING) {
    emitRGBA(vec4(colormapViridis(v), v * opacity));
  } else {
    emitRGB(colormapViridis(v));
  }
}
```

**MAX/MIN projection with explicit intensity**:
```glsl
#uicontrol invlerp normalized

// paste colormapPlasma here

void main() {
  float v = normalized();
  emitRGBA(vec4(colormapPlasma(v), 1.0));
  emitIntensity(v);
}
```

**Diverging colormap for signed data** (coolwarm centered at zero):
```glsl
#uicontrol invlerp normalized(range=[-500, 500], window=[-1000, 1000], clamp=false)

// paste colormapCoolwarm here

void main() {
  // invlerp maps -500→0.0, 0→0.5, 500→1.0 (perfect for diverging)
  emitRGB(colormapCoolwarm(clamp(normalized(), 0.0, 1.0)));
}
```

---

### 2. Annotation Layers

**Documentation**: `src/annotation/rendering.md`

The same shader applies to all annotation types (points, lines, polylines, bounding boxes, ellipsoids). Type-specific functions have no effect when rendering other types.

#### Property Access

```glsl
// Syntax: prop_<propertyName>()
// Return types map from property type:
//   float32  → float
//   uint8/16/32 → highp uint
//   int8/16/32  → highp int
//   bool     → highp uint (0 or 1)
//   rgb      → vec3  (normalized [0,1])
//   rgba     → vec4  (normalized [0,1])

float      prop_score();       // float32 property named "score"
highp uint prop_label();       // uint property named "label"
highp int  prop_depth();       // int property named "depth"
vec3       prop_rgb_color();   // rgb property named "rgb_color"
```

#### Common API

```glsl
const bool PROJECTION_VIEW;   // true = 3D projection view, false = cross-section

// Hide this annotation entirely:
discard;                       // use as statement (NOT discard())

// Sets color for all annotation types at once:
void setColor(vec4 rgba);
void setColor(vec3 rgb);       // alpha = 1.0

// Returns the UI-configured annotation color:
vec3 defaultColor();           // if not called, color selector won't appear in UI
```

#### Point Annotations

```glsl
void setPointMarkerSize(float diameterInScreenPixels);      // default: 5px
void setPointMarkerBorderWidth(float widthInScreenPixels);  // default: 1px
void setPointMarkerColor(vec4 rgba);
void setPointMarkerColor(vec3 rgb);
void setPointMarkerBorderColor(vec4 rgba);
void setPointMarkerBorderColor(vec3 rgb);
```

#### Line Annotations

```glsl
void setLineColor(vec4 rgba);
void setLineColor(vec3 rgb);
void setLineColor(vec4 startColor, vec4 endColor);  // color gradient along line
void setLineColor(vec3 startColor, vec3 endColor);
void setLineWidth(float widthInScreenPixels);         // default: 1px
void setEndpointMarkerColor(vec4 rgba);
void setEndpointMarkerColor(vec3 rgb);
void setEndpointMarkerColor(vec4 startColor, vec4 endColor);
void setEndpointMarkerSize(float diameter);
void setEndpointMarkerSize(float startDiameter, float endDiameter);
void setEndpointMarkerBorderWidth(float width);
void setEndpointMarkerBorderWidth(float startWidth, float endWidth);
void setEndpointMarkerBorderColor(vec4 rgba);
void setEndpointMarkerBorderColor(vec3 rgb);
```

#### Polyline Annotations

Polylines follow the same API as lines; use `setPoly` prefix to target polylines only:
```glsl
void setPolyLineColor(vec4 startColor, vec4 endColor);
void setPolyLineWidth(float width);
void setPolyEndpointMarkerColor(vec4 startColor, vec4 endColor);
void setPolyEndpointMarkerSize(float startSize, float endSize);
void setPolyEndpointMarkerBorderColor(vec4 startColor, vec4 endColor);
void setPolyEndpointMarkerBorderWidth(float startWidth, float endWidth);
```

#### Bounding Box Annotations

```glsl
void setBoundingBoxBorderColor(vec4 rgba);
void setBoundingBoxBorderColor(vec3 rgb);
void setBoundingBoxBorderWidth(float widthInScreenPixels);  // default: 1px
void setBoundingBoxFillColor(vec4 rgba);   // cross-section view only
void setBoundingBoxFillColor(vec3 rgb);
```

#### Ellipsoid Annotations

```glsl
void setEllipsoidFillColor(vec4 rgba);
void setEllipsoidFillColor(vec3 rgb);
```

#### Default Shader

```glsl
void main() {
  setColor(defaultColor());
}
```

#### Annotation Layer Examples

**Color by float property using viridis**:
```glsl
#uicontrol invlerp colorScale(property="score", range=[0, 1])

// paste colormapViridis here

void main() {
  setColor(vec4(colormapViridis(colorScale()), 1.0));
}
```

**Hide low-confidence points with threshold**:
```glsl
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.5, step=0.01)
#uicontrol invlerp colorScale(property="score", range=[0, 1])

// paste colormapViridis here

void main() {
  if (prop_score() < threshold) discard;
  setColor(vec4(colormapViridis(colorScale()), 1.0));
}
```

**Scale point size by property value**:
```glsl
#uicontrol invlerp sizeScale(property="weight", range=[0, 100])
#uicontrol float maxSize slider(min=2.0, max=30.0, default=12.0, step=0.5)
#uicontrol vec3 color color(default="yellow")
void main() {
  setPointMarkerSize(sizeScale() * maxSize);
  setPointMarkerColor(vec3(color));
  setPointMarkerBorderColor(vec4(0, 0, 0, 1));
}
```

**Line with gradient (shallow → deep)**:
```glsl
#uicontrol invlerp depthScale(property="depth", range=[0, 1000])
void main() {
  float v = depthScale();
  setLineColor(
    vec4(0.2, 0.2, 1.0, 1.0),       // start: blue (shallow)
    vec4(1.0, 0.2, 0.2, 1.0)        // end: red (deep)
  );
}
```

**Categorical coloring by integer label**:
```glsl
#uicontrol vec3 color1   color(default="red")
#uicontrol vec3 color2   color(default="green")
#uicontrol vec3 color3   color(default="blue")
#uicontrol vec3 colorOther color(default="gray")
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.05)
void main() {
  uint label = prop_label();
  vec3 c = colorOther;
  if (label == 1u)      c = color1;
  else if (label == 2u) c = color2;
  else if (label == 3u) c = color3;
  setColor(vec4(c, opacity));
}
```

**Different appearance in 2D vs 3D**:
```glsl
void main() {
  if (PROJECTION_VIEW) {
    setColor(vec4(1.0, 0.5, 0.0, 0.6));  // semi-transparent orange in 3D
  } else {
    setColor(vec4(1.0, 1.0, 0.0, 1.0));  // solid yellow in cross-section
  }
}
```

---

### 3. Skeleton Layers

**Source file**: `src/skeleton/frontend.ts`

Skeletons render as lines (edges) and circles (nodes) using the same `void main()`. Vertex attributes from the skeleton data are exposed per-vertex as `prop_name()`.

#### API

```glsl
vec4 segmentColor();         // Neuroglancer-assigned color for this segment (RGBA)
void emitDefault();          // use the segment's assigned color
void emitRGB(vec3 color);    // override color (keeps segment alpha)
void emitRGBA(vec4 color);   // full override

// Vertex attribute access (name matches skeleton data attribute):
float prop_radius();          // example: float attribute named "radius"
float prop_synapse_density(); // example: float attribute named "synapse_density"
// Types depend on skeleton data format
```

**Built-in colormaps**: `colormapJet` and `colormapCubehelix` are available.

#### Default Shader

```glsl
void main() {
  emitDefault();
}
```

#### Skeleton Layer Examples

**Color by vertex attribute with viridis**:
```glsl
#uicontrol invlerp colorScale(range=[0, 100])

// paste colormapViridis here

void main() {
  emitRGB(colormapViridis(colorScale(prop_synapse_density())));
}
```

**Blend between segment color and attribute color**:
```glsl
#uicontrol invlerp attrNorm(range=[0, 1])
#uicontrol float blend slider(min=0.0, max=1.0, default=0.5, step=0.05)

// paste colormapViridis here

void main() {
  vec4 seg = segmentColor();
  vec3 attr = colormapViridis(attrNorm(prop_myAttribute()));
  emitRGB(mix(seg.rgb, attr, blend));
}
```

**Diverging colormap on vertex attribute**:
```glsl
#uicontrol invlerp normalized(range=[-1, 1], clamp=true)

// paste colormapCoolwarm here

void main() {
  float t = normalized(prop_signed_value());
  emitRGB(colormapCoolwarm(t));
}
```

**Threshold-based highlight**:
```glsl
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.5, step=0.01)
#uicontrol vec3 highlightColor color(default="orange")
void main() {
  if (prop_score() > threshold) {
    emitRGB(highlightColor);
  } else {
    emitDefault();
  }
}
```

---

### 4. Single Mesh Layers

**Source file**: `src/single_mesh/frontend.ts`

Meshes render with per-vertex lighting applied automatically. Vertex attributes from the mesh file are available as GLSL variables directly (without `prop_` prefix).

#### API

```glsl
void emitGray();                    // default white/gray with lighting
void emitRGB(vec3 color);           // color with lighting applied
void emitRGBA(vec4 color);          // with transparency + lighting
void emitPremultipliedRGBA(vec4 c); // pre-multiplied alpha
```

#### Vertex Attribute Access

Attribute names come from the mesh file. They are exposed as variables, NOT as `prop_*()` functions:
```glsl
// If mesh has attribute named "curvature":
float curvatureValue = curvature;

// If mesh has attribute named "thickness":
float t = thickness;
```

#### Default Shader

```glsl
void main() {
  emitGray();
}
```

#### Single Mesh Examples

**Colormap by vertex attribute**:
```glsl
#uicontrol invlerp colorScale(range=[0, 1])

// paste colormapViridis here

void main() {
  emitRGB(colormapViridis(colorScale(myAttribute)));
}
```

**Semi-transparent with color control**:
```glsl
#uicontrol vec3 color color(default="cyan")
#uicontrol float opacity slider(min=0.0, max=1.0, default=0.7, step=0.01)
void main() {
  emitRGBA(vec4(color, opacity));
}
```

---

## UI Controls Reference

All controls are declared with `#uicontrol` before `void main()`.

### `slider` — Numeric slider

```glsl
#uicontrol float brightness slider(min=-1, max=1, default=0, step=0.01)
#uicontrol int   channel   slider(min=0, max=3, default=0)
#uicontrol uint  level     slider(min=0, max=255, default=128)
```

- `min` and `max`: required
- `step`: optional (defaults to 1 for int/uint, `(max-min)/100` for float)
- `default`: optional (defaults to `min` for int/uint, `(min+max)/2` for float)

### `color` — Color picker

```glsl
#uicontrol vec3 color color(default="red")
#uicontrol vec3 bg    color(default="#1a2b3c")
```

- Type must be `vec3` (RGB in `[0,1]`)
- Default is a CSS color string (quoted); defaults to `"white"` if omitted

### `checkbox` — Boolean toggle (compile-time constant)

```glsl
#uicontrol bool invert    checkbox(default=false)
#uicontrol bool showMask  checkbox(default=true)
```

- Type must be `bool`
- Toggling **recompiles** the shader (it's a compile-time constant)
- Best for major behavioral switches, not per-frame adjustments

### `invlerp` — Inverse linear interpolation with histogram

Maps a data interval to `[0, 1]` and shows an ECDF histogram in the UI.

```glsl
// Image layer:
#uicontrol invlerp normalized(range=[0, 255], window=[0, 1000], channel=0, clamp=true)

// Multi-dimensional channel (e.g., time=0, channel=1):
#uicontrol invlerp ch1(range=[0, 255], channel=[0, 1])

// Annotation layer:
#uicontrol invlerp colorScale(property="score", range=[0.0, 1.0], window=[0.0, 2.0])
```

Parameters:
- `range`: data interval mapped to `[0, 1]` — adjustable in UI. Can be inverted: `range=[100, 0]` maps 100→0, 0→1.
- `window`: histogram display range — adjustable in UI. Defaults to `range`.
- `channel`: channel index or array for histogram (image layers).
- `property`: property name for ECDF (annotation layers).
- `clamp`: whether to clamp result to `[0, 1]`. Default `true`. **Not adjustable in UI**.

Generated functions:
```glsl
float normalized(T value);  // normalize an explicit value
float normalized();         // normalize current channel/property automatically
```

### `transferFunction` — Color + opacity via control points

Maps data values to RGBA using a series of control points interpolated linearly.

```glsl
#uicontrol transferFunction tf(
  window=[0, 65535],
  controlPoints=[[0.0, "#000000", 0.0], [32768.0, "#0055ff", 0.5], [65535.0, "#ffffff", 1.0]],
  channel=0,
  defaultColor="#ffffff"
)
```

Parameters:
- `window`: display range for the UI. Required if control points don't span the full range. **Cannot be inverted**.
- `controlPoints`: array of `[inputValue, "#rrggbb", opacity]`. Data before first point → transparent. Data after last point → color of last point.
- `channel`: channel for histogram.
- `defaultColor`: color for new control points added via UI.

Generated functions:
```glsl
vec4 tf(T value);  // map explicit value to RGBA
vec4 tf();         // map current channel value to RGBA
```

---

## Colormap Library

Copy these GLSL functions into your shader before `void main()`.

**All functions**: take `float x` in `[0, 1]`, return `vec3` RGB.

### Built-in (always available — no copy needed)

```glsl
vec3 colormapJet(float x)        // classic rainbow — not perceptually uniform
vec3 colormapCubehelix(float x)  // perceptually uniform dark-to-bright spiral
```

---

### Sequential Colormaps

#### Viridis — perceptually uniform, colorblind-safe (recommended default)
Purple → blue → green → yellow

```glsl
vec3 colormapViridis(float x) {
  x = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 0.5139045538019999);
  vec3 c2 = vec3(-0.1554846426062665, 0.214847559468114, 0.2882845726573711);
  vec3 c3 = vec3(4.421483672780069, -4.815752998712279, -1.523304699551617);
  vec3 c4 = vec3(-6.449900613484578, 6.814218890839987, 1.586319730987697);
  vec3 c5 = vec3(4.985027369390448, -5.374467529984653, -1.049898449823647);
  vec3 c6 = vec3(-1.630030898948953, 2.019384888944737, 0.3665667028843458);
  return clamp(c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6))))), 0.0, 1.0);
}
```

#### Plasma — high contrast, perceptually uniform
Purple → pink → yellow

```glsl
vec3 colormapPlasma(float x) {
  x = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
  vec3 c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
  vec3 c2 = vec3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
  vec3 c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158);
  vec3 c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
  vec3 c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655740221);
  vec3 c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828);
  return clamp(c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6))))), 0.0, 1.0);
}
```

#### Inferno — perceptually uniform, print-safe
Black → purple → orange → bright yellow-white

```glsl
vec3 colormapInferno(float x) {
  x = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(0.0002189403691192265, 0.001651004631528365, -0.01948089843709184);
  vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  vec3 c3 = vec3(-41.70399613249965, 17.43639888205313, 44.35414519872813);
  vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return clamp(c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6))))), 0.0, 1.0);
}
```

#### Magma — perceptually uniform
Black → purple → pink → white

```glsl
vec3 colormapMagma(float x) {
  x = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
  vec3 c1 = vec3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
  vec3 c2 = vec3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
  vec3 c3 = vec3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
  vec3 c4 = vec3(52.17613981234068, -27.94360607168351, 12.94416944238394);
  vec3 c5 = vec3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
  vec3 c6 = vec3(18.65570506591883, -11.48977351915498, -5.601961508734096);
  return clamp(c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6))))), 0.0, 1.0);
}
```

---

### Diverging Colormaps

Diverging colormaps are for data that has a meaningful center value (e.g., 0 for signed data, or a baseline for differences). Center → 0.5 input.

#### Coolwarm — blue → white → red
Best for correlation coefficients, signed differences, z-scores

```glsl
vec3 colormapCoolwarm(float x) {
  // x=0.0 → cool blue,  x=0.5 → neutral gray-white,  x=1.0 → warm red
  x = clamp(x, 0.0, 1.0);
  vec3 blue  = vec3(0.229, 0.298, 0.754);
  vec3 white = vec3(0.865, 0.865, 0.865);
  vec3 red   = vec3(0.706, 0.016, 0.150);
  return x < 0.5
    ? mix(blue, white, x * 2.0)
    : mix(white, red, (x - 0.5) * 2.0);
}
```

#### RdBu — red → white → blue
Same as Coolwarm reversed; common in climate science and neuroscience

```glsl
vec3 colormapRdBu(float x) {
  // x=0.0 → red,  x=0.5 → white,  x=1.0 → blue
  x = clamp(x, 0.0, 1.0);
  vec3 red   = vec3(0.647, 0.000, 0.149);
  vec3 white = vec3(0.969, 0.969, 0.969);
  vec3 blue  = vec3(0.192, 0.212, 0.584);
  return x < 0.5
    ? mix(red, white, x * 2.0)
    : mix(white, blue, (x - 0.5) * 2.0);
}
```

#### Using Diverging Colormaps

**With `invlerp` (recommended — gives histogram UI):**
```glsl
// invlerp maps symmetrically: range[-v, +v] → [0, 1], center 0 → 0.5
#uicontrol invlerp normalized(range=[-100, 100], window=[-200, 200], clamp=true)

// paste colormapCoolwarm here

void main() {
  emitRGB(colormapCoolwarm(normalized()));
}
```

**With adjustable center and half-range sliders (symmetric):**
```glsl
#uicontrol float center    slider(min=-1000, max=1000, default=0, step=1)
#uicontrol float halfRange slider(min=1, max=2000, default=500, step=1)

// paste colormapCoolwarm here

void main() {
  float raw = toRaw(getDataValue());
  float t = (float(raw) - center) / (2.0 * halfRange) + 0.5;
  emitRGB(colormapCoolwarm(clamp(t, 0.0, 1.0)));
}
```

**With independent cool min, midpoint, and hot max (asymmetric):**

Use this when the data range above the midpoint differs from the range below it — for example, a skewed distribution, or when you want to emphasize one side more than the other.

```glsl
#uicontrol float midpoint slider(min=-1000, max=1000, default=0,    step=1)
#uicontrol float coolMin  slider(min=-2000, max=0,    default=-500, step=1)
#uicontrol float hotMax   slider(min=0,     max=2000, default=500,  step=1)

// paste colormapCoolwarm here

void main() {
  float raw = float(toRaw(getDataValue()));
  float t;
  if (raw <= midpoint) {
    // Map [coolMin, midpoint] → [0.0, 0.5]
    t = 0.5 * (raw - coolMin) / max(midpoint - coolMin, 1e-6);
  } else {
    // Map [midpoint, hotMax] → [0.5, 1.0]
    t = 0.5 + 0.5 * (raw - midpoint) / max(hotMax - midpoint, 1e-6);
  }
  emitRGB(colormapCoolwarm(clamp(t, 0.0, 1.0)));
}
```

The `max(..., 1e-6)` guards against division by zero if the user collapses a range to zero. Adjust the `slider` `min`/`max`/`default` values to match the actual data range.

---

## Common Pitfalls

### `discard;` not `discard()`
In annotation shaders, use `discard;` as a statement:
```glsl
if (prop_score() < threshold) discard;  // correct
if (prop_score() < threshold) discard(); // WRONG - compile error
```

### Signed integers have no `toNormalized()`
`int8_t`, `int16_t`, `int32_t` lack `toNormalized()`. Use `invlerp` with an explicit range (it handles signed types), or normalize manually:
```glsl
// WRONG for int16:
float v = toNormalized(getDataValue());

// Correct option 1 — use invlerp:
// #uicontrol invlerp normalized(range=[-32768, 32767])
// float v = normalized();

// Correct option 2 — manual:
float v = float(toRaw(getDataValue())) / 32767.0 * 0.5 + 0.5;
```

### Alpha in volume rendering mode
In volume rendering, alpha is opacity-corrected per step. Use small values:
```glsl
// WRONG — fully opaque per step causes solid blob:
emitRGBA(vec4(color, 1.0));  // in VOLUME_RENDERING mode

// Correct — small per-step alpha accumulates naturally:
emitRGBA(vec4(color, 0.02));
```

### `hsvToRgb` is not available in most layers
`hsvToRgb` is only injected into segmentation layers. For image/annotation/skeleton shaders, define it manually if needed:
```glsl
vec3 hsvToRgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
```

### New colormaps must be pasted inline
`colormapViridis`, `colormapPlasma`, etc. are NOT built-in. Copy the function definition from the Colormap Library section into the shader before `void main()`.

### `transferFunction` window cannot be inverted
Unlike `invlerp range`, the `window` for `transferFunction` must satisfy start < end.

### `checkbox` triggers a shader recompile
Boolean checkboxes are compile-time constants. Toggling them recompiles. This is intentional but means they are not suitable for per-frame toggles. Use them for major mode switches.

### `uint64_t` is special
For 64-bit data, avoid `toNormalized()` — it's only defined for 32-bit-and-under types. Use comparisons or `toRaw()`:
```glsl
// Segment IDs (uint64_t) — compare for equality:
uint64_t v = getDataValue();
if (v.value.x == 42u && v.value.y == 0u) {
  emitRGB(vec3(1, 0, 0));
} else {
  emitTransparent();
}
```

---

## Quick Reference

### Layer → Emit Functions

| Layer | Emit functions |
|---|---|
| Image | `emitGrayscale(float)`, `emitRGB(vec3)`, `emitRGBA(vec4)`, `emitTransparent()`, `emitIntensity(float)` |
| Annotation | `setColor(vec4)`, `setPointMarkerColor(vec4)`, `setLineColor(vec4)`, `setBoundingBoxBorderColor(vec4)`, `setEllipsoidFillColor(vec4)` |
| Skeleton | `emitDefault()`, `emitRGB(vec3)`, `emitRGBA(vec4)` |
| Mesh | `emitGray()`, `emitRGB(vec3)`, `emitRGBA(vec4)` |

### Layer → Data Access

| Layer | Data access |
|---|---|
| Image | `getDataValue(ch)`, `getInterpolatedDataValue(ch)` |
| Annotation | `prop_name()`, `defaultColor()` |
| Skeleton | `prop_attrName()`, `segmentColor()` |
| Mesh | direct attribute variables (e.g., `curvature`) |

### Control → Best Use

| Control | Best for |
|---|---|
| `invlerp` | contrast/range normalization; shows histogram in UI — **prefer this** |
| `transferFunction` | complex color+opacity mappings with interactive control points |
| `slider(float)` | thresholds, scale factors, opacity, blend weights |
| `slider(int)` | channel selection, discrete levels |
| `color` | fixed hue/tint with interactive color picker |
| `checkbox` | major mode switches (recompiles shader) |

### Conversion Function Availability

| Type | `toRaw()` | `toNormalized()` |
|---|---|---|
| `float` | identity | identity |
| `uint8_t` | ✓ → uint | ✓ → [0,1] |
| `uint16_t` | ✓ → uint | ✓ → [0,1] |
| `uint32_t` | ✓ → uint | ✓ → [0,1] |
| `int8_t` | ✓ → int | **✗ not available** |
| `int16_t` | ✓ → int | **✗ not available** |
| `int32_t` | ✓ → int | **✗ not available** |
| `uint64_t` | ✗ (use `.value`) | **✗ not available** |

### Colormap Summary

| Name | Type | Range description |
|---|---|---|
| `colormapJet` | sequential | blue→cyan→green→yellow→red (built-in) |
| `colormapCubehelix` | sequential | dark spiral, perceptually uniform (built-in) |
| `colormapViridis` | sequential | purple→blue→green→yellow, colorblind-safe |
| `colormapPlasma` | sequential | purple→pink→yellow, high contrast |
| `colormapInferno` | sequential | black→purple→orange→yellow-white |
| `colormapMagma` | sequential | black→purple→pink→white |
| `colormapCoolwarm` | **diverging** | blue→white→red (center at x=0.5) |
| `colormapRdBu` | **diverging** | red→white→blue (center at x=0.5) |
