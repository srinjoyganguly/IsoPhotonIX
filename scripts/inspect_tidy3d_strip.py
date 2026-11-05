"""Independent local Tidy3D SDK check of the public waveguide-plugin example.

This optional script requires the Tidy3D Python package. Its local mode solver
uses CPU and does not submit remote jobs. No Tidy3D dependency is added to the
workbench application or its MPB benchmark.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from benchmark_component_references import FREQUENCY_STEP, PERMITTIVITY, REFERENCES, TIDY3D, WAVELENGTH


def main():
    import tidy3d as td
    from tidy3d.plugins import waveguide

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    silicon = td.material_library['cSi']['Li1993_293K']
    silica = td.material_library['SiO2']['Horiba']
    for ratio, expected in PERMITTIVITY.items():
        frequency = ratio * td.C_0 / WAVELENGTH
        core, clad = silicon.eps_model(frequency), silica.eps_model(frequency)
        if not np.allclose([core.real, clad.real, clad.imag, core.imag], [*expected,0.], rtol=1e-12,atol=1e-15):
            raise ValueError('Material version differs from the reported MPB real-permittivity samples.')

    td.config.simulation.use_local_subpixel = False
    rows = []
    keys = ['te_neff','tm_neff','te_group','tm_group','te_area_um2','tm_area_um2']
    for resolution in (15,30,60):
        guide = waveguide.RectangularDielectric(wavelength=WAVELENGTH, core_width=.5,
            core_thickness=.22, core_medium=silicon, clad_medium=silica,
            grid_resolution=resolution,
            mode_spec=td.ModeSpec(num_modes=2,group_index_step=FREQUENCY_STEP))
        data = guide.mode_solver.solve()
        values = np.concatenate([data.n_eff.values.ravel(), data.n_group.values.ravel(),
                                 data.mode_area.values.ravel()])
        if values.shape != (6,) or not np.isfinite(values).all():
            raise RuntimeError('Unexpected local mode solver output.')
        measured = dict(zip(keys, map(float, values)))
        errors = {key:abs(value-REFERENCES['tidy3d-strip-plugin'][key][2])/REFERENCES['tidy3d-strip-plugin'][key][2]
                  for key,value in measured.items()}
        rows.append({'grid_resolution': resolution, 'computed': measured, 'relative_errors_to_public_output':errors})
        print(resolution,measured,flush=True)
    report = {'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'source_url':TIDY3D,
              'tidy3d_version':td.__version__,'solver':'Installed Tidy3D SDK local CPU mode solver',
              'use_local_subpixel':False,'group_frequency_step_fraction':FREQUENCY_STEP,
              'geometry':'500 nm × 220 nm cSi/Li1993_293K core in SiO2/Horiba cladding at 1.55 µm.',
              'scope':'Local discretization contrast using the same named complex media. The public notebook output remains the reference. MPB is an independent solver.',
              'rows':rows}
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
