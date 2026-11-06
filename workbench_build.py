"""Copy only explicitly selected application files into release artifacts."""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
MODULES = ('__init__ app benchmarks calibrated_geometry calibrations calibration_performance calibration_fields planar_port_layout calibration_replay capabilities compare3d component_library geometry_draft jobs '
           'layout_examples layout_grid layout_study library_layout mode_benchmarks model passport ports ports3d ports_capabilities ports_probe probe '
           'reproduce reproduction_runtime runtime solver_environment source_snapshot sparameters spectra study3d taper_geometry '
           'taper_refinement taper_study taper_ports taper_port_geometry thin_film volume worker').split()
ADAPTERS = ('fdtdx_adapter fdtdx3d_adapter fdtdx_ports_adapter fdtdx_ports3d_adapter fdtdx_taper fdtdx_layout '
            'meep_adapter meep3d_adapter meep_ports_adapter meep_ports3d_adapter meep_taper meep_layout').split()
FILES = ('LICENSE', 'CREDITS.md', 'frontend/THIRD_PARTY_NOTICES.txt', 'frontend/dist/index.html',
         'docs/licenses/MEEP-GPL-2.0.txt', 'docs/licenses/MPB-GPL-2.0.txt', 'docs/licenses/BEAMZ-APACHE-2.0.txt',
         'environment-meep.yml', 'environment-meep-linux-64.lock',
         'environment-fdtdx-cpu-linux-64.lock', 'environment-fdtdx-cuda12-linux-64.lock', 'environment-beamz-cpu-linux-64.lock', 'requirements-api.txt', 'scripts/run_simulation.py', 'scripts/benchmark_component_references.py', 'scripts/benchmark_passive_cavity.py', 'scripts/benchmark_beamz_reference.py',
         'docs/verification/v014-grating-meshes.json','docs/verification/v014-gpu-performance.json','docs/verification/v014-gpu-performance.csv','docs/verification/v012-calibrated-index.json','docs/verification/v011-calibrated-index.json', 'scripts/calibrate_multiport.py', 'scripts/calibrate_fdtdx_multiport.py',
         'scripts/grating_mirror.py','scripts/grating_cuda_identity.py','scripts/grating_emission.py','scripts/lumerical_grating3d.py', 'scripts/grating_angular_spectrum.py', 'docs/verification/v012-grating-commercial-references.json','docs/verification/v012-grating-memory-budget.json',
         'docs/verification/v013-grating-dense-reference.json','docs/verification/v013-grating-dense-memory.json',
         'scripts/gpu_timing.py','scripts/phase_polygon_material.py','scripts/ring_wall_tensor.py','scripts/ring_stage_provenance.py', 'scripts/measure_loaded_ring_losses.py', 'scripts/calibrate_loaded_ring3d.py', 'scripts/calibrate_finite_grating.py', 'scripts/calibrate_planar_tm.py', 'scripts/calibrate_layout_ports.py', 'scripts/calibrate_layout_ports3d.py',
         'scripts/ring_ade_stage.py','scripts/ring_palik_dispersion.py','docs/verification/v013-ring-project-audit.json',
         'scripts/calibrate_multimode.py', 'scripts/calibrate_ring.py', 'scripts/calibrate_grating.py', 'scripts/calibration_checkpoint.py',
         'docs/verification/v010-passive-cavity.json', 'docs/verification/v010-beamz-cpu.json', 'docs/verification/v010-beamz-gpu.json',
         'requirements-fdtdx.txt', 'requirements-fdtdx-cuda.txt', 'docs/verification/public-mode-benchmarks.json',
         'docs/verification/public-component-comparisons.json', 'docs/verification/v09-family-fields.json', 'examples/finite-waveguide-3d.json', 'examples/ports-gap.json', 'examples/ports-straight.json',
         'examples/ports3d-gap.json', 'examples/ports3d-straight.json', 'examples/slab-waveguide.json')


class FrontendAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'script' and 'src' in values:
            reference = values['src']
        elif tag == 'link' and values.get('rel') in ('stylesheet', 'modulepreload'):
            reference = values['href']
        else:
            return
        if not re.fullmatch(r'/assets/[\w-]+\.(?:js|css)', reference, flags=re.ASCII):
            raise ValueError(f'Unexpected asset in the compiled interface: {reference!r}')
        self.paths.add('frontend/dist' + reference)


def runtime_files():
    index = ROOT / 'frontend/dist/index.html'
    if not index.is_file():
        raise RuntimeError('Build the web interface with pnpm build in frontend before building a wheel from source.')
    assets = FrontendAssets()
    assets.feed(index.read_text())
    if not any(p.endswith('.js') for p in assets.paths) or not any(p.endswith('.css') for p in assets.paths):
        raise RuntimeError('The compiled interface requires a local script and stylesheet.')
    selected = {*FILES, *assets.paths, *(f'backend/{m}.py' for m in MODULES),
                *(f'backend/adapters/{m}.py' for m in ADAPTERS)}
    import json
    grating_routes=json.loads((ROOT/'docs/verification/v014-grating-meshes.json').read_text())['qualified']
    for route in grating_routes:
        if route['resolution'] >= 16:
            mirror='-mirror' if route['domain']=='mirror-y' else ''
            selected.add(f"docs/verification/v014-grating-memory-{route['geometry']}{mirror}-r{route['resolution']}.json")
    for case in ('coupler','splitter'):
        phase_file = f'docs/verification/v012-{case}-phase.json'
        if (ROOT / phase_file).is_file():
            selected.add(phase_file)
    for name in sorted(selected):
        source = ROOT / name
        if not source.is_file() or source.resolve() != source:
            raise RuntimeError(f'Missing regular release input: {name}')
        yield name


def copy_payload(destination):
    for name in runtime_files():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


class ApplicationBuild(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / 'photonics_workbench/_app'
        # Remove stale hashed bundles from previous builds in this build area.
        if destination.exists():
            shutil.rmtree(destination)
        copy_payload(destination)


class ApplicationSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_payload(Path(base_dir))
