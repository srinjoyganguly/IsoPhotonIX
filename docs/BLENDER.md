# Photonics journey — original Blender film

This 24-second film presents the six-step workflow: open a slab waveguide,
edit a physical parameter, understand the solver guidance, explicitly select
the GPU, inspect an actual recorded electromagnetic field, and export the
Run Passport. It accompanies the actual application recording; it is an
editorial animation, not a screen capture or a live timed solve.

The scene, lighting, typography, camera motion, cards, and geometry were made
for this project. No commercial-tool screenshots, stock models, external
fonts, or music were used. The slab's thickness is an illustrative 3D
extrusion. The field plane is explicitly labeled as actual **2D FDTDX data**.
It does not depict a solved 3D mode or claim normalized transmission.

## Recorded numerical source

The 25 field textures come from the completed local GPU run
`b722e5f659854a7786afee1d6d3ca619`: FDTDX 0.6.2 / JAX 0.10.2 on the NVIDIA
RTX 3080 Ti Laptop GPU, actual device `cuda:0`. The physical model is a
10 × 5 µm dielectric slab, 0.8 µm core width, indices 2.0/1.0, wavelength
1.55 µm, 20 cells/µm, and 1 µm PML.

Every displayed texture is an actual saved Ez frame. The source arrays have
100 × 200 samples, enlarged to 800 × 400 by nearest-neighbor replication.
Image top is +y; image bottom is −y. A fixed symmetric scale of
±1.258102536201477 arbitrary source units is used throughout: teal for negative
Ez, copper for positive Ez, and dark near zero. The movie holds measured
frames; it does not synthesize or interpolate electromagnetic samples.

The recorded frames span 0–77.78825468 fs. The calculation itself ended at
79.98081578 fs, so the last movie texture is deliberately not described as
the final solver state. Each replayed frame shows its actual recorded time.

The original NPZ SHA-256 is:

```text
3f81aadc42362b683d48830f79480b4da715730ff552941acc1e43e3861e0da5
```

`assets/metadata.json` retains the times, fixed color scale, configuration,
versions, actual device, and that checksum. Independent checks compared all
25 PNGs pixel-for-pixel against the recorded arrays and confirmed the
timestamps and checksum.

## Reproduce the film

The release asset `photonics-journey-assets.zip` contains:

```text
photonics-journey.blend
blender_journey.py
BLENDER.md
assets/metadata.json
assets/field_000.png ... assets/field_024.png
```

The Blender file packs all 25 textures and uses only relative resource and
render-output paths. It can be opened without the original workstation.
The separate textures and script also allow rebuilding the scene from scratch.
The script is specifically authored for this recorded run; it is not a
general solver or an importer for arbitrary simulations.

Raw Blender PNGs can embed the source `.blend` filename in image metadata.
The distributed poster has all PNG metadata removed without changing any
pixels; strip metadata before publicly sharing other raw render frames. The
released scene and script also disable source-filename PNG metadata. Packed
image origin paths and the file browser's fixed-length directory buffer were
scrubbed, then the decompressed `.blend` was checked for private path markers.
The packed scene reloaded in a folder without external textures and reproduced
frame 430 pixel-for-pixel.

Tested renderer: **Blender 5.2.0 LTS**, build `fbe6228777e7`, Windows x86-64.
Settings: EEVEE, 16 render samples, 1280 × 720, Standard view transform,
24 fps, 576 frames, no ray tracing, no compositing. Installed Blender RNA/API
properties and real render output were checked before creating the film.

From the extracted source-asset directory:

```bash
blender --background --python blender_journey.py -- --assets assets --output render --samples 16 --step 1 --render
```

For a single preview, replace `--render` with `--preview 430`. The script saves
the packed `.blend` and writes the rendered PNGs to `render/frames/`.
Encode the complete native 24 fps sequence with FFmpeg:

```bash
ffmpeg -framerate 24 -i render/frames/frame_%04d.png -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart -an journey-blender.mp4
```

The first shader compilation took about 26 seconds; warmed frames took about
0.3–0.4 seconds on the development machine. Those are render observations,
not electromagnetic solver timings. A lower-rate emergency render can use
`--step 2`, but the published film uses all 576 native 24 fps frames.

Official references:

- [Blender Python API](https://docs.blender.org/api/current/)
- [EEVEE](https://docs.blender.org/manual/en/latest/render/eevee/index.html)
- [Command-line rendering](https://docs.blender.org/manual/en/latest/advanced/command_line/render.html)

Blender is the external GPL-licensed authoring tool. The new project script
uses the application's MIT license; the original scene and numerical
visualization assets are included with this project. External numerical
libraries retain the licenses listed in `CREDITS.md`.
