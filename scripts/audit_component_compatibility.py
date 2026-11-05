"""Check current component designs against the original v0.7 Git sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.component_library import build_component, get_catalog

BASE_REVISION = '175a198'


def baseline(folder):
    package = folder / 'legacy_library'
    package.mkdir()
    (package / '__init__.py').write_text('')
    for module in ('geometry_draft', 'component_library'):
        content = subprocess.check_output(['git', 'show', f'{BASE_REVISION}:backend/{module}.py'], cwd=ROOT)
        (package / f'{module}.py').write_bytes(content)
    spec = importlib.util.spec_from_file_location('legacy_library', package/'__init__.py',
                                                  submodule_search_locations=[str(package)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from legacy_library.component_library import get_catalog as legacy_catalog, build_component as legacy_build
    return legacy_catalog(), legacy_build


def digest(geometry):
    return hashlib.sha256(json.dumps(geometry, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audit():
    families = []
    with tempfile.TemporaryDirectory() as temp:
        old, generate_old = baseline(Path(temp))
        for recipe in old['components']:
            identifier = recipe['id']
            edited = {'height_um': .36}
            for key in ('length_um', 'radius_um', 'period_um'):
                if key in recipe['defaults']:
                    edited[key] = round(recipe['defaults'][key] * 1.125, 6)
                    break
            cases = {}
            for name, kwargs in (
                ('default', {}),
                ('edited', {'parameters': edited}),
                ('edited_transformed_material', {'parameters': edited, 'placement': {
                    'x': 4., 'y': -3., 'z': .5, 'rotation_deg': 37.}, 'material_id': 'silicon-reference'}),
            ):
                previous = generate_old(identifier, **kwargs)['geometry']
                current = build_component(identifier, **kwargs)['geometry']
                if current != previous or digest(current) != digest(previous):
                    raise ValueError(f'{identifier} changed its saved {name} geometry since {BASE_REVISION}.')
                cases[name] = {'geometry_sha256': digest(current), 'objects': len(current['objects']),
                               'ports': len(current['components'][0]['ports'])}
            families.append({'id': identifier, 'cases': cases, 'edited_parameters': edited})
    return {'passed': True, 'baseline_commit': subprocess.check_output(
        ['git', 'rev-parse', BASE_REVISION], cwd=ROOT, text=True).strip(),
        'checked_at': datetime.now(timezone.utc).isoformat(), 'legacy_families': len(families),
        'new_catalog_families': len(get_catalog()['components']), 'geometry_comparisons': 3 * len(families),
        'scope': 'Complete validated geometry documents, including generated solids, ports, contours, materials and recipes. Catalogue capability labels are outside this geometry comparison.',
        'families': families}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f"{result['geometry_comparisons']} exact geometry comparisons across {result['legacy_families']} v0.7 families.")
