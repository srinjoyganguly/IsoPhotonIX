# Persistent circuit connections

A connection saves two named component ports in the design. Each port belongs to at most one connection. The saved endpoints follow their component IDs through regeneration, save, import and export. A connection can be planned while the ports are still physically apart.

## Connect two strips in the editor

1. Open a strip in the 3D editor. Place a second strip from the component library.
2. Open **Component connections**. Select the moving strip’s **in** port and the fixed strip’s **out** port.
3. Choose **Connect without moving** to record a planned connection. Inspect its separation in micrometres, opposing-angle error and width. A dashed line shows the planned link in the viewport.
4. Choose **Connect and align** for equal-width ports. The moving component rotates and translates as one piece. The same connection ID survives. Coincident, oppositely directed, width-matched endpoints show an aligned joint.
5. Select **Disconnect** to remove the saved link. The solids retain their positions. Deleting an instance also removes its attached links. Undo restores the instance and the links together.
6. Export the design as JSON, Python or a Jupyter notebook. The stand-alone geometry program and notebook reproduce the saved geometry and connections. The notebook displays each pair’s current alignment.

**What the field solver sees.** Both Meep and FDTDX use the materials and physical solids. A gap remains a gap until geometry changes. The alignment check measures positions, directions and widths. A mode mismatch, radiation loss or manufacturing violation needs its own study. Automatic bend routing, hierarchy, process layers, Boolean operations, GDS round-trips and fabrication design-rule checks remain in the [product checklist](NEXT_CHUNKS.md).

## Work with an exported design

The Python program contains the full editable design in `GEOMETRY`. From the unpacked design archive, use the bundled helpers to inspect a saved pair or align it. The same helpers are included in finite-layout field study exports.

```python
from geometry import GEOMETRY
from workbench_runtime import connect_ports, connection_status, disconnect_ports

edge = GEOMETRY["connections"][0]
a, b = edge["a"], edge["b"]
aligned = connect_ports(
    GEOMETRY, a["component_id"], a["port"], b["component_id"], b["port"],
    align=True, fit_domain=True,
)
print(connection_status(aligned))
disconnected = disconnect_ports(aligned, edge["id"])
```

Version 1 and earlier version 2 documents open with their original geometry. Existing drafts retain their old representation until the first saved connection. The optional `connections` array belongs to version 2. Import rejects dangling endpoints, duplicate connection IDs, self-connections and a second link on an occupied port.
