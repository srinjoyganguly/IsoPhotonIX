# Interactive 3D geometry workspace

The searchable [component library](COMPONENT_LIBRARY.md) includes 27 parametric
families, polygon editing, material records and layout placement. For a real
simulation from a library or custom shape, open [finite-layout fields](LAYOUT_FIELDS.md).
The [XY drawing guide](XY_LAYOUT.md) covers routes, contours and moving whole components.


The **3D design** workspace is a local, interactive geometry editor. It lets you
inspect and edit finite solids in physical units. A bounded **finite-height
waveguide study** now connects supported geometry to real Meep CPU and FDTDX
CPU/NVIDIA GPU calculations. Use **Open runnable 3D waveguide** in the study
panel, then follow the [3D simulation guide](3D_SIMULATION.md). The [finite-layout study](LAYOUT_FIELDS.md) runs eligible drawn solids and multi-component scenes. [Delete, undo, redo and port alignment](EDITOR_RECOVERY.md).
The **2D simulation** workspace retains the existing Meep CPU and
FDTDX CPU/NVIDIA GPU slab workflow, actual field playback and result exports.
Geometry edits do not modify that solver configuration.

See [ROADMAP.md](ROADMAP.md) for the staged numerical and CAD work, and
[PACKAGING.md](PACKAGING.md) for the proposed pip, local web and desktop
installation architecture. The current platform and managed solver coverage are tracked in the installation guide.

## Open and inspect a design

Start the local application using [SETUP.md](SETUP.md), then choose **3D design**.
The editor opens the saved browser draft when available; otherwise it opens the
ring-and-bus template. Three templates are available:

- **Strip waveguide:** a finite rectangular core above a substrate.
- **Directional coupler:** two parallel rectangular cores; edit their positions
  and widths to change the separation.
- **Ring & bus:** an annular core beside a straight guide. The annulus has a
  rectangular cross-section; it is not a circular-cross-section torus.

These general templates are geometric starting points, not executable studies;
the runnable waveguide has its own validated template. A template
change can be undone. Coordinates use micrometres: **x** is the nominal
propagation direction, **y** is lateral and **z** is vertical. The design domain
is a box centred at the origin. Its outline is a visual guide, not a configured
simulation boundary or PML.

The viewport requires WebGL 2. If graphics initialization fails or the context
is lost, an explanation and **Reopen viewport** control appear. The geometry
editor remains available. Browser graphics capability does not determine which
solver execution targets are available; those are checked separately by the
local service.

## Camera and selection controls

| Action | Control |
| --- | --- |
| Orbit around the current camera target | Drag with the left mouse button; one-finger drag on touch screens |
| Zoom | Mouse wheel, middle-button drag, two-finger pinch, or the **+ / −** buttons |
| Pan | Right-button drag; two-finger move on touch screens |
| Pan with the keyboard | Focus the canvas and use the arrow keys |
| Fit the full design domain | **Fit view**, or **F** while the canvas has focus |
| Isometric direction | **Iso** |
| View along z, showing the XY plane | **Top** |
| View from negative y, showing XZ | **Front** |
| View from positive x, showing YZ | **Right** |
| Select a solid | Click its visible surface, or its row in the Scene list |
| Clear selection | Click empty space in the viewport |
| Show or hide guide overlays | **Grid** and **Domain** |

The camera presets remain perspective views. They are not orthographic
measurement tools. The viewport displays its camera-to-target distance and the
number of visible objects. The coloured axis marker uses red x, green y and blue
z. Lower-index unselected materials appear slate and higher-index materials
cyan; the selected solid appears amber. These colours are presentation choices,
not calculated optical fields or a quantitative material colour scale.

## Create and edit geometry

Use **New shape** and **+ Add** to create a box, z-aligned cylinder or annular
ring at the origin. Select a solid to edit its name, centre position, dimensions,
rotation about z and refractive index in the Inspector. Boxes expose x/y/z
sizes. Cylinders expose outer radius and z thickness. Rings also expose inner
radius. Rotation of a circular cylinder or ring around its own z axis leaves its
visible geometry unchanged.

Numeric and name edits commit on **Enter** or when the input loses focus.
**Escape** cancels the pending text edit. Invalid edits show an explanation and
retain the last valid geometry. Objects must stay inside the design domain;
enlarge the domain before moving or enlarging an object beyond its bounds.

**Duplicate** creates a second object at the same location. Edit its position to
separate it. **Delete object**, template replacement and imports can be undone
with **Undo**; **Redo** reapplies an undone change. The editor keeps up to 50
prior design states for this page session. Reloading the page does not restore
that history or the camera position.

The eye control in the Scene list affects only viewport visibility. A hidden
solid remains in the document. Visibility must not be interpreted as excluding
a material from a future simulation. Overlaps are permitted for draft editing;
overlapping arbitrary drafts cannot be run. The supported straight-guide study
requires a touching core/substrate interface and defines material placement
explicitly, independent of list order.

Refractive index is stored as a real, scalar value. The validated straight-guide
study uses it in the actual solver material grid. General unsupported designs
remain drafts. Dispersion, absorption, anisotropy, material
libraries and wavelength validity are not represented by this field.

## Save and recover your work

Validated edits are autosaved as **one current draft** in this browser's local
storage for the application's origin. Changing the browser, port or profile
can give a different storage area. Clearing site data or using a private browser
session can remove it. Browser autosave is not a project library or a durable
research archive.

Use **Export geometry** to download a portable `photonics-geometry.json` file.
Keep this file in your own project folder. **Import geometry** reads a validated
geometry document and replaces the open design; Undo restores the previous
state. The interface accepts files smaller than 250 KB. Unknown fields,
unsupported schema versions and malformed values are rejected.

If another tab changes the saved draft, detected changes pause autosave in the
current tab and leave its open design intact. Export it first if you need both
versions. **Save this draft** explicitly replaces the shared browser draft with
the current design and resumes autosaving. This protection is not collaborative
editing or an atomic cross-tab locking protocol.

A corrupt saved draft is not overwritten merely by opening the app. The editor
shows a warning and a fresh template; a subsequent valid edit or import can
start a new saved draft. If browser storage is unavailable, exporting JSON is
still the way to retain the design.

## Preview the existing slab

**Preview current 2D slab** reads the most recently validated slab dimensions
from the 2D workspace and displays an illustrative finite extrusion. Its label
is **EXTRUDED 2D PREVIEW**. The solver's physical model remains invariant along
z; the displayed thickness is not a solved device thickness and changing this
preview does not affect the 2D solver.

Editing physical geometry changes the preview into a 3D draft. Imported and
reloaded geometry documents are also treated as drafts, even if their file
contains `mode: "slab-preview"`. A file cannot establish that it is linked to the
current solver configuration.

**Open tested 2D simulation** returns to the established slab workflow. Explicit
CPU/GPU target selection is still required there. The geometry editor does not
silently project a 3D design into that solver or generate substitute results.

## Geometry file format

This format is separate from `waveguide-config.json` and numerical result ZIPs.
It contains geometry and presentation visibility, not a source, mesh,
boundary condition, solver selection or simulation result. No code in the file
is executed. New designs use `isophotonix-geometry`. The importer also accepts
the earlier `photonics-workbench-geometry` identifier.

```json
{
  "schema": "isophotonix-geometry",
  "schema_version": 1,
  "units": "um",
  "name": "Example strip draft",
  "mode": "draft-3d",
  "domain": { "x": 10, "y": 6, "z": 2 },
  "objects": [
    {
      "id": "core",
      "name": "Strip core",
      "visible": true,
      "kind": "box",
      "position": { "x": 0, "y": 0, "z": 0 },
      "size": { "x": 8, "y": 0.5, "z": 0.22 },
      "radius": 1,
      "inner_radius": 0.6,
      "rotation_deg": 0,
      "refractive_index": 3.45
    }
  ]
}
```

All listed fields are required. Objects retain all dimension fields for a stable
schema: `size.x/y/z` defines a box; a cylinder uses `radius` and `size.z`; a ring
uses `radius`, `inner_radius` and `size.z`. Unused dimensions are retained and
validated but do not alter that shape. Positions specify centres, not corners.
Cylinder and ring axes are z-aligned. The only supported rotation is about z.

The current validation bounds are:

| Property | Limit |
| --- | --- |
| Object count | 0–100, with unique IDs |
| Domain lengths | 0.1–1,000 µm per axis |
| Object dimensions | At least 0.001 µm; the rotated shape must fit in the domain |
| Ring | Inner radius and radial wall thickness each at least 0.001 µm |
| Refractive index | Real finite value from 1 to 5 |
| Rotation input | −360° to +360°, normalized to [0°, 360°) |
| Names | 1–80 non-control characters |
| IDs | 1–80 letters, digits, underscores or hyphens; first character alphanumeric |

The validator also rejects nonfinite numbers, coercible numeric strings,
unexpected keys and unsupported shape kinds. These bounds protect an editable
geometry document. They do not establish optical suitability, fabrication
feasibility, convergence or numerical accuracy.

## Current boundaries

This milestone provides geometric inspection and parameter-based construction.
It does not yet include transform handles, arbitrary-axis rotation, snapping,
Boolean operations, polygon drawing, CAD/GDS import, layer-stack semantics,
general source/monitor editing, or arbitrary 3D studies. The current 3D study
has a fixed Ey source and six-face PML, bounded grid settings, and actual final
field volumes. Broader study models remain in [the roadmap](ROADMAP.md).
Existing 2D results continue to describe only the
immutable 2D run configuration that produced them.

## Reproduce the checks

The geometry-model tests use Node 24's native TypeScript support. From `frontend`:

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm test:geometry
```

With the local service running and Playwright installed as described in
[browser-tests.md](browser-tests.md):

```bash
PHOTONICS_BROWSER_CHANNEL=chrome pnpm test:geometry:browser
```

[Recorded editor checks](verification/3d-editor.json) include actual pointer
gestures, raycast selection, editing, import/export, cross-tab recovery,
responsive layouts and WebGL failure handling. The
[solver regressions](verification/3d-solver-regression.json) record separate
real 2D runs and numerical export checks. A 3D screenshot alone is not solver
verification.
