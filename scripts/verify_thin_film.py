"""Compare the planar-film spectrum with independently sampled 1-D Meep flux and complex fields."""
import argparse
import cmath
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import meep as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.thin_film import film_spectrum

LEFT, RIGHT = -1.75, 1.75
THICKNESS, INCIDENT, SUBSTRATE = 0.4, 1.0, 1.45
RESOLUTIONS = [80, 160, 320]
CELL, PML, AFTER_PULSE = 14, 2, 100


def measure(wave, index, resolution, reference=None, *, short_boundary=False):
    cell, pml, duration = (10, 1, 65) if short_boundary else (CELL, PML, AFTER_PULSE)
    f = 1 / wave
    geometry = []
    if reference is not None:
        eps = index**2
        end = THICKNESS / 2
        geometry = [
            mp.Block(center=mp.Vector3(z=(end+cell/2)/2), size=mp.Vector3(mp.inf,mp.inf,cell/2-end), material=mp.Medium(index=SUBSTRATE)),
            mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf,mp.inf,THICKNESS),
                     material=mp.Medium(epsilon=eps.real, D_conductivity=2*math.pi*f*eps.imag/eps.real))]
    sim = mp.Simulation(dimensions=1, cell_size=mp.Vector3(z=cell), resolution=resolution, default_material=mp.Medium(index=INCIDENT),
                        geometry=geometry, boundary_layers=[mp.PML(pml)], sources=[mp.Source(mp.GaussianSource(f, fwidth=.22*f, cutoff=5 if short_boundary else 7),
                        component=mp.Ex,center=mp.Vector3(z=-3))])
    try:
        flux = [sim.add_flux(f, 0, 1, mp.FluxRegion(center=mp.Vector3(z=z), direction=mp.Z), decimation_factor=0 if short_boundary else 1) for z in (LEFT, RIGHT)]
        fields = [sim.add_dft_fields([mp.Ex],f,0,1,center=mp.Vector3(z=z),size=mp.Vector3(),decimation_factor=0 if short_boundary else 1) for z in (LEFT, RIGHT)]
        if reference is not None:
            sim.load_minus_flux_data(flux[0], reference)
        sim.run(until_after_sources=duration)
        return (sim.get_flux_data(flux[0]), [float(mp.get_fluxes(m)[0]) for m in flux],
                [complex(np.mean(sim.get_dft_array(m, mp.Ex, 0))) for m in fields])
    finally:
        sim.reset_meep()


def pair(wave, sample, resolution, *, short_boundary=False):
    index = complex(sample['n'], sample['k'])
    reference, incident_flux, incident_fields = measure(wave, index, resolution, short_boundary=short_boundary)
    _, device_flux, device_fields = measure(wave, index, resolution, reference, short_boundary=short_boundary)
    if min(incident_flux) <= 0 or abs(incident_fields[0]) == 0:
        raise RuntimeError('Empty-run incident normalization failed.')
    k0 = 2*math.pi / wave
    front_distance, back_distance = -THICKNESS/2 - LEFT, RIGHT - THICKNESS/2
    r = (device_fields[0] / incident_fields[0] - 1) * cmath.exp(-2j*k0*INCIDENT*front_distance)
    t = device_fields[1] / incident_fields[0] * cmath.exp(-1j*k0*(INCIDENT*front_distance + SUBSTRATE*back_distance))
    R, T = -device_flux[0]/incident_flux[0], device_flux[1]/incident_flux[0]
    p = lambda z: [z.real,z.imag]
    measured = {'R':R,'T':T,'A':1-R-T,'r':p(r),'t':p(t)}
    errors = {key:abs(measured[key]-sample[key]) for key in ('R','T','A')}
    errors.update(r=abs(r-complex(*sample['r'])), t=abs(t-complex(*sample['t'])))
    return {'resolution_cells_per_um':resolution,'incident_flux':incident_flux,'device_scattered_and_transmitted_flux':device_flux,
            'incident_total_Ex':[p(z) for z in incident_fields],'device_total_Ex':[p(z) for z in device_fields],
            'meep':measured,'absolute_differences':errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    mp.verbosity(0)
    base={'id':'film','name':'Qualification film','source':'Synthetic passive index for software validation',
          'notes':'','reference_wavelength_um':1.55}
    record={'model':'tabulated-nk','wavelength_um':[1.35,1.55,1.75],'n':[2.1,2,1.9],'k':[.12,.08,.04]}
    cases=[]
    for material,waves in [({'model':'constant-nk','n':2,'k':0},[1.55]),(record,[1.45,1.55])]:
        config={'material':{**base,**material},'thickness_um':THICKNESS,'incident_index':INCIDENT,'substrate_index':SUBSTRATE,'wavelengths_um':waves}
        analytic=film_spectrum(config)
        for sample in analytic['samples']:
            sequence=[pair(sample['wavelength_um'],sample,res) for res in RESOLUTIONS]
            final=sequence[-1]['absolute_differences']; coarse=sequence[0]['absolute_differences']
            nonzero = ['R','T','r','t'] if sample['k'] == 0 else ['R','T','A','r','t']
            passed = max(final.values()) < .002 and all(final[key] < coarse[key] for key in nonzero)
            case = {'config':config,'analytic':sample,'mesh_sequence':sequence}
            if sample['k'] == 0:
                short = [pair(sample['wavelength_um'],sample,res,short_boundary=True) for res in RESOLUTIONS]
                case['short_boundary_control'] = {'cell_length_um':10,'pml_um':1,'run_after_pulse_meep_units':65,
                    'dft_decimation_factor':'Meep automatic','pulse_cutoff':5,'mesh_sequence':short}
                # With no material loss, residual power estimates the numerical boundary/measurement floor.
                passed = passed and all(m['absolute_differences']['A'] < 1e-7 for m in sequence)
                passed = passed and final['A'] < short[-1]['absolute_differences']['A']
                case['zero_absorption_mesh_decrease'] = final['A'] < coarse['A']
            case['passed'] = passed
            cases.append(case)
    report={'passed':all(c['passed'] for c in cases),'checked_at':datetime.now(timezone.utc).isoformat(),'meep_version':mp.__version__,
            'method':'Each mesh/wavelength uses its own homogeneous reference followed by a device run, a Gaussian Ex source, two flux monitors and DFT Ex probes. Subtract the incident Fourier fields before measuring reflected power. Reference complex amplitudes to the two physical film faces using the analytic phase of each lossless exterior. Absorption is residual power.',
            'loss_model':'For each independent bin, ε∞=Re[(n+ik)²], σ_D=2πf Im[(n+ik)²]/ε∞. This gives the requested complex index exactly at that frequency. These separate narrowband checks do not assert a broadband causal fit.',
            'mesh_criterion':'Every finest-mesh absolute R/T/A and complex r/t difference must be <0.002. Nonzero observables must improve from 80 to 320 cells/µm. For analytically zero absorption, all three absolute residuals must be <1e-7 and the finest wide-boundary residual must improve on the short-boundary control. The raw monotonic trend for this residual is recorded separately.',
            'measurement':{'incident_monitor_z_um':LEFT,'transmitted_monitor_z_um':RIGHT,'cell_length_um':CELL,'pml_um':PML,
                           'source_z_um':-3,'run_after_pulse_meep_units':AFTER_PULSE,'dft_decimation_factor':1},'cases':cases}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Passed:', report['passed'], [case['mesh_sequence'][-1]['absolute_differences'] for case in cases])
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
