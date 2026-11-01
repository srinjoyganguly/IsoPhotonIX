"""Create an original Blender film from recorded electromagnetic field textures.

Blender 5.2: blender -b --python scripts/blender_journey.py -- --assets assets
  --output render --preview 430
Use --render to render the full PNG sequence. All field images are packed in
the saved blend. No simulated data are invented by this presentation script.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--preview", type=int)
parser.add_argument("--render", action="store_true")
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--scale", type=int, default=100)
parser.add_argument("--step", type=int, default=1, help="Render every Nth frame; use2 for12fps sampling of the24fps timeline")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "frames").mkdir(exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = args.samples
scene.eevee.use_raytracing = False
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = args.scale
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 576
scene.frame_step = args.step
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
scene.render.use_file_extension = True
scene.render.use_stamp_filename = False
scene.render.use_compositing = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "Medium High Contrast" if "Medium High Contrast" in [i.identifier for i in scene.view_settings.bl_rna.properties["look"].enum_items] else "None"
scene.world = bpy.data.worlds.new("Obsidian atmosphere")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.015, 0.025, 0.035, 1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.32

TEAL = (0.025, 0.69, 0.60, 1)
COPPER = (0.92, 0.40, 0.15, 1)
WHITE = (0.78, 0.90, 0.94, 1)
MUTED = (0.31, 0.47, 0.53, 1)
DARK = (0.012, 0.026, 0.034, 1)


def material(name, color, metallic=0, roughness=0.4, emission=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    if emission:
        node.inputs["Emission Color"].default_value = color
        node.inputs["Emission Strength"].default_value = emission
    return mat


obsidian = material("Polished obsidian", DARK, 0.65, 0.24)
core_mat = material("Copper dielectric schematic", COPPER, 0.68, 0.23)
teal = material("Teal signal ink", TEAL, emission=1.5)
copper = material("Copper signal ink", COPPER, emission=1.2)
white = material("Warm white type", WHITE, emission=1)
muted = material("Muted technical type", MUTED, emission=0.75)
grid_mat = material("Precision grid", (0.035, 0.10, 0.115, 1), emission=0.4)
card_mat = material("Camera panel", (0.009, 0.024, 0.032, 1), emission=0.8)


def box(name, loc, dims, mat, bevel=0.05, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Soft machined edges", "BEVEL")
        mod.width, mod.segments = bevel, 4
        obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    if parent:
        obj.parent = parent
    return obj


def curve(name, points, mat, radius=0.008, parent=None):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = radius
    data.bevel_resolution = 2
    spline = data.splines.new("POLY")
    spline.points.add(len(points)-1)
    for point, xyz in zip(spline.points, points):
        point.co = (*xyz, 1)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
    return obj


def visible(obj, start, end=576):
    for frame, hidden in [(1, True), (max(1, start-1), True), (start, False), (end+1, True)]:
        obj.hide_render = hidden
        obj.hide_viewport = hidden
        obj.keyframe_insert("hide_render", frame=frame)
        obj.keyframe_insert("hide_viewport", frame=frame)


def key(obj, prop, values):
    for frame, value in values:
        setattr(obj, prop, value)
        obj.keyframe_insert(prop, frame=frame)


camera_data = bpy.data.cameras.new("Journey camera")
camera = bpy.data.objects.new("Journey camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (9, -14, 11)
camera.rotation_euler = (Vector((0, 0, 0.0))-camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.type = "ORTHO"
camera.data.ortho_scale = 15.8
camera.data.lens = 45
scene.camera = camera


def text(name, body, x, y, size=0.3, mat=white, start=1, end=576, align="LEFT"):
    data = bpy.data.curves.new(name, "FONT")
    data.body, data.size, data.align_x = body, size, align
    data.space_character = 1.08
    data.extrude = 0
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.parent = camera
    obj.location = (x, y, -8.0)
    obj.data.materials.append(mat)
    visible(obj, start, end)
    return obj


def panel(name, x, y, width, height, start, end=576, accent=teal):
    obj = box(name, (x, y, -8.2), (width, height, 0.035), card_mat, 0.08, camera)
    visible(obj, start, end)
    strip = box(name+" accent", (x-width/2+0.04, y, -8.14), (0.035, height-0.14, 0.035), accent, 0.01, camera)
    visible(strip, start, end)
    return obj


def area(name, loc, color, power, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color, data.shape, data.size = power, color, "DISK", size
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = (Vector((0, 0, 0))-obj.location).to_track_quat("-Z", "Y").to_euler()


area("Soft teal key", (2, -4, 8), (0.35, 0.85, 1), 1100, 7)
area("Copper rim", (-6, 3, 5), (1, 0.47, 0.22), 1700, 5)
area("Overhead silk", (1, 5, 9), (0.76, 0.92, 1), 1800, 7)
box("Infinite studio plinth", (0, 0, -0.85), (200, 200, 0.2), material("Studio floor", (0.003, 0.007, 0.011, 1), 0, .82), 0)

model = bpy.data.objects.new("Schematic assembly", None)
scene.collection.objects.link(model)
model.location = (-0.55, 0, -0.08)
key(model, "rotation_euler", [(1, (0, 0, -.12)), (192, (0, 0, .04)), (384, (0, 0, .10)), (576, (0, 0, -.02))])
key(model, "scale", [(1,(.82,.82,.82)), (96,(.82,.82,.82)),
                     (192,(.90,.90,.90)), (288,(.96,.96,.96)),
                     (384,(1,1,1)), (576,(1,1,1))])
base = box("Dielectric domain schematic", (0, 0, -.19), (8.35, 4.35, .30), obsidian, .12, model)
curve("Domain outline", [(-4,-2,.01),(4,-2,.01),(4,2,.01),(-4,2,.01),(-4,-2,.01)], teal, .012, model)
for ix in range(-10, 11):
    curve("Grid x", [(ix*.4,-2,.025),(ix*.4,2,.025)], grid_mat, .0025, model)
for iy in range(-5, 6):
    curve("Grid y", [(-4,iy*.4,.025),(4,iy*.4,.025)], grid_mat, .0025, model)
core = box("Core width schematic", (0,0,.11), (8,.48,.15), core_mat, .025, model)
key(core, "scale", [(1,(1,1,1)),(100,(1,1,1)),(158,(1,4/3,1)),(384,(1,4/3,1))])
visible(core, 1, 383)
source = box("Finite line source marker", (-2.8,0,.24), (.025,.48,.25), teal, .008, model)
key(source, "scale", [(1,(1,1,1)),(100,(1,1,1)),(158,(1,4/3,1)),(384,(1,4/3,1))])
visible(source, 1, 383)
for y in (-.32,.32):
    o = curve("Final core boundary", [(-4,y,.055),(4,y,.055)], copper, .005, model)
    visible(o,384)
for x in (-3.2,3.2):
    for iy in range(12):
        curve("PML edge marker", [(x,-2+iy*.34,.04),(x,-1.8+iy*.34,.04)], muted, .005, model)

# Field values are supplied by the actual recorded run, normalized once with a
# shared symmetric color scale. Each texture is a discrete measured snapshot.
images = sorted(args.assets.glob("field_*.png"))
if len(images) != 25:
    raise RuntimeError(f"Expected 25 real field textures, got {len(images)}")
data = json.loads((args.assets/"metadata.json").read_text())
times = data.get("times_fs", [])
for index, path in enumerate(images):
    img = bpy.data.images.load(str(path.resolve()), check_existing=True)
    img.pack()
    img.filepath = "//assets/"+path.name
    for packed in img.packed_files:
        packed.filepath = img.filepath
    mat = bpy.data.materials.new(f"Measured Ez snapshot {index:02}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    emission.inputs["Strength"].default_value = 1
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,.045))
    plane = bpy.context.object
    plane.name = f"ACTUAL 2D FIELD {index:02}"
    plane.parent = model
    plane.scale = (4,2,1)
    plane.data.materials.append(mat)
    start = 384+index*4
    end = start+3 if index < 24 else 576
    visible(plane, start, end)
    if times:
        text(f"Simulation time {index}", f"RECORDED Ez   /   t = {times[index]:.2f} fs", -6.8, -2.65, .21, muted, start,end)

# Camera-space typography is original artwork, never a screenshot or a fake UI.
text("Brand", "PHOTONICS / WORKBENCH", -6.8, 3.88, .23, teal)
text("Local mark", "LOCAL COMPUTE     OPEN SOURCE SOLVERS", 6.8, 3.88, .16, muted, align="RIGHT")
chapters = [
    (1,96,"01  /  OPEN", "Begin with a waveguide.", "A bundled slab. A wavelength. A place to start."),
    (97,192,"02  /  EDIT", "Make a meaningful change.", "Core width 0.60 to 0.80 um. Validated before execution."),
    (193,288,"03  /  UNDERSTAND", "Know why a solver fits.", "Meep for a compact 2D study. FDTDX for explicit JAX / CUDA."),
    (289,383,"04  /  CHOOSE", "Choose your hardware.", "Your selection is explicit. GPU never silently becomes CPU."),
    (384,480,"05  /  SIMULATE", "Watch a real field emerge.", "Recorded FDTDX run. Actual Ez snapshots, replayed in time."),
    (481,576,"06  /  EXPORT", "Carry the evidence with you.", "Configuration. Numerical arrays. Device and solver provenance."),
]
for start,end,kicker,title,subtitle in chapters:
    text("Chapter "+kicker, kicker, -6.8, 3.10, .2, copper,start,end)
    text("Title "+kicker, title, -6.8, 2.38, .58, white,start,end)
    text("Subtitle "+kicker, subtitle, -6.8, 1.93, .235, muted,start,end)

panel("Parameters", 5.03, -.10, 3.16, 2.55, 1,192)
text("Model label", "SLAB WAVEGUIDE", 3.72,.77,.20,teal,1,192)
text("Wavelength", "1.55 um", 3.72,.25,.38,white,1,192)
text("Wavelength sub", "free-space wavelength",3.72,-.06,.17,muted,1,192)
text("Dimensions", "10 x 5 um   |   n = 2.0 / 1.0",3.72,-.53,.18,white,1,192)
text("Core original", "CORE  0.60 um",3.72,-.94,.21,copper,1,136)
text("Core validated", "CORE  0.80 um   /   VALID",3.72,-.94,.21,teal,137,192)

panel("Solver guide", 5.03, -.10, 3.16, 2.65,193,288)
text("Meep rec", "MEEP  /  CPU",3.72,.76,.25,white,193,288)
text("Meep why", "Native 2D. Low overhead.",3.72,.36,.18,muted,193,288)
text("FDTD alt", "FDTDX  /  JAX",3.72,-.22,.25,copper,193,288)
text("FDTD why", "Explicit CPU or NVIDIA.",3.72,-.62,.18,muted,193,288)
text("Guidance", "GUIDANCE, NOT A LOCK",3.72,-1.02,.16,teal,193,288)

panel("Selected GPU", 5.03,-.1,3.16,2.65,289,480)
text("GPU badge", "GPU SELECTED",3.72,.76,.24,teal,289,480)
text("GPU name", "NVIDIA RTX 3080 Ti",3.72,.28,.21,white,289,480)
text("GPU laptop", "Laptop GPU  /  16 GB",3.72,-.08,.18,muted,289,480)
text("GPU verified", "VERIFIED  cuda:0",3.72,-.65,.24,copper,289,480)
text("GPU source", "FDTDX 0.6.2 + JAX 0.10.2",3.72,-1.02,.16,muted,289,480)

panel("Run passport",5.03,-.1,3.16,2.85,481,576,accent=copper)
text("Passport heading","RUN PASSPORT",3.72,.85,.24,copper,481)
text("Passport id","b722e5f6  /  cuda:0",3.72,.41,.20,white,481)
text("Passport contents","config.json\narrays.npz\nmetadata.json",3.72,-.07,.20,muted,481)
text("Passport export","EXPORT  .ZIP",3.72,-1.10,.28,teal,481)

text("Schematic label", "3D SCHEMATIC  /  geometry preview", -6.8,-2.65,.19,muted,1,383)
text("Field disclaimer", "3D schematic  /  actual 2D FDTDX data  /  arbitrary source units", -6.8,-3.02,.18,muted,384)
text("No comparison", "Recorded simulation, not a live solve or performance comparison.", -6.8,-3.35,.16,muted,384)
text("Core purpose", "From a physical idea to reproducible electromagnetic evidence.",-6.8,-3.02,.20,muted,1,383)

labels = ["OPEN", "EDIT", "GUIDE", "SELECT", "RUN", "EXPORT"]
for index,label in enumerate(labels):
    x = -6.8 + index*2.42
    text("Rail "+label, f"{index+1:02}  {label}",x,-3.95,.18,muted)
    rail = box("Active "+label,(x+.71,-4.12,-8.0),(1.42,.025,.025),teal,.006,camera)
    visible(rail,index*96+1,min(576,(index+1)*96))

# Preserve original application provenance without private absolute paths.
scene["presentation"] = "Original 24-second Blender schematic with recorded FDTDX 2D field frames"
scene["scientific_scope"] = "Actual 2D Ez snapshots; illustrative 3D extrusion; no eigenmode, transmission, or benchmark claim"
scene["source_run"] = "b722e5f659854a7786afee1d6d3ca619"
scene["source_solver"] = "FDTDX 0.6.2 / JAX 0.10.2 / cuda:0"
scene["source_model"] = "10 x 5 um, 0.8 um core, n=2/1, wavelength1.55um"
scene["source_arrays_sha256"] = data.get("arrays_sha256", "")
scene["texture_scale_peak"] = float(data.get("peak", 0))
for screen in bpy.data.screens:
    for area in screen.areas:
        for space in area.spaces:
            if space.type == "FILE_BROWSER" and space.params:
                # Blender stores this in a fixed buffer; overwrite old bytes
                # before shortening it so private default directories cannot
                # survive after the terminating NUL in the saved blend.
                space.params.directory = b"_" * 1023
                space.params.directory = b"//"
                space.params.filename = ""
scene.render.filepath = "//frames/frame_"
scene.frame_set(430)
bpy.ops.wm.save_as_mainfile(filepath=str((args.output/"photonics-journey.blend").resolve()))
if args.preview:
    scene.frame_set(args.preview)
    scene.render.filepath = str((args.output/f"preview-{args.preview:03}.png").resolve())
    bpy.ops.render.render(write_still=True)
if args.render:
    scene.render.filepath = str((args.output/"frames"/"frame_").resolve())
    bpy.ops.render.render(animation=True)
