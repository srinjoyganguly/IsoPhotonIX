"""Quantify a same-geometry, same-source grating material control and its independent slab screen."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.grating_phase_matching import screen
from scripts.lumerical_grating3d import fiber_radial_field, lp01

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def fiber_aperture(receipt):
    """Check the fraction of an infinite LP01 profile within the actual incident aperture."""
    from scipy.integrate import quad
    report=json.loads(receipt.read_text())
    m=report['settings']
    dx=1/m['resolution']
    theta=np.radians(m['angle_deg'])
    u,w,_=lp01(m['fiber_radius'],m['fiber_core_n'],m['fiber_clad_n'],1.55)
    def intensity(r):
        return fiber_radial_field(r/m['fiber_radius'],u,w)**2
    def radial(r):
        return float(r*intensity(np.array([r]))[0])
    radius=m['fiber_radius']
    total=2*np.pi*(quad(radial,0.,radius)[0]+quad(radial,radius,np.inf)[0])/np.cos(theta)
    with np.load(receipt.parent/report['arrays_file'],allow_pickle=False) as data:
        # E_z co-location is half a cell below a scalar x or y coordinate.
        limits=[]
        for key in ('ref_upper_x_um','ref_upper_y_um'):
            axis=data[key]
            limits.append((axis[0]-dx,axis[-1]))
    center=m['fiber_center_x']+report['stages']['reference']['sample_plane_z_um']*np.tan(theta)
    integrals=[]
    for order in (96,192):
        nodes,weights=np.polynomial.legendre.leggauss(order)
        x,y=[(b-a)*nodes/2+(a+b)/2 for a,b in limits]
        area=np.prod([b-a for a,b in limits])/4
        values=intensity(np.hypot((x[:,None]-center)*np.cos(theta),y[None,:]))
        integrals.append(float(area*(weights@values@weights)/total))
    return {'resolution':m['resolution'],'fraction_of_infinite_lp01_in_sampled_aperture':integrals[-1],
            'quadrature_change':abs(integrals[-1]-integrals[0]),'aperture_bounds_xy_um':limits,
            'scope':'Independent weak-guidance scalar fiber profile at 1550 nm on the horizontal plane. Actual normalization uses the numerical incident Poynting flux.'}


def native_pair(baseline, experiment, commercial):
    rows=[json.loads(p.read_text()) for p in (baseline,experiment)]
    first,second=rows
    settings=second['settings'].copy()
    if settings.pop('grating_material_model')!='direct-yee' or first['settings']!=settings:
        raise ValueError('Physical input changed beyond the prespecified material sampling.')
    flux_checks=[]
    for run,path in zip(rows,(baseline,experiment)):
        raw=path.parent/run['arrays_file']
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=run['arrays_sha256'] or not run['passed']:
            raise ValueError('Invalid native control archive.')
        if run['actual_device']!='cuda:0' or not all(run['result']['valid_beam_bins']):
            raise ValueError('Expected five qualified bins on NVIDIA.')
        with np.load(raw,allow_pickle=False) as a:
            measured=abs(a['output_mode'])**2/a['incident_power']
            np.testing.assert_allclose(measured,run['result']['beam_to_guide'],rtol=1e-7)
            dx=1/run['settings']['resolution']
            plane=a['dev_output'][:,:,0]
            # Both archives store co-located E/H on the output and incident-reference planes.
            outgoing=-.5*np.real(plane[:,1]*plane[:,5].conj()-plane[:,2]*plane[:,4].conj())
            scattered=a['dev_upper']-a['ref_upper']
            upward=.5*np.real(scattered[:,0]*scattered[:,4].conj()-scattered[:,1]*scattered[:,3].conj())
            def integrated(values):
                return (values.sum(axis=(-2,-1),dtype=np.float64)*dx**2/a['incident_power']).tolist()
            flux_checks.append({'signed_total_outgoing_flux_fraction':integrated(outgoing),
                                'upper_scattered_net_flux_fraction':integrated(upward),
                                'projected_fiber_return_power_fraction':run['radiation']['fiber_return_power']})
    for role in ('reference','device'):
        for key in ('source_eh_sha256','source_time_offsets_sha256','scalar_permittivity_sha256','shape','output_plane_x_um','sample_plane_z_um'):
            if first['stages'][role][key]!=second['stages'][role][key]:
                raise ValueError(f'Confounded control ({role}, {key}).')
    if first['device_name']!=second['device_name']:
        raise ValueError('Native target model changed.')
    old,new=[np.asarray(r['result']['beam_to_guide']) for r in rows]
    target=np.asarray(commercial['samples']['S21']['power'])
    np.testing.assert_array_equal(first['result']['wavelength_um'],commercial['wavelength_um'])
    np.testing.assert_array_equal(second['result']['wavelength_um'],commercial['wavelength_um'])
    center=commercial['wavelength_um'].index(1.55)
    center_errors=[float(100*abs(val[center]-target[center])) for val in (old,new)]
    maes=[float(100*abs(val-target).mean()) for val in (old,new)]
    return {'resolution':settings['resolution'],'baseline_receipt':baseline.name,'experiment_receipt':experiment.name,
            'baseline_arrays_sha256':first['arrays_sha256'],'experiment_arrays_sha256':second['arrays_sha256'],
            'same_source_eh_and_time_offsets':True,'same_physical_input_and_scalar_grid':True,
            'same_gpu_model':first['device_name'],'published_power_fraction':target.tolist(),
            'baseline_power_fraction':old.tolist(),'direct_yee_power_fraction':new.tolist(),
            'difference_from_published_percentage_points':(100*(new-target)).tolist(),
            'baseline_finite_aperture_fluxes':flux_checks[0],'direct_yee_finite_aperture_fluxes':flux_checks[1],
            'baseline_five_bin_mae_pp':maes[0],'direct_yee_five_bin_mae_pp':maes[1],
            'baseline_center_error_pp':center_errors[0],'direct_yee_center_error_pp':center_errors[1],
            'improved_both_primary_metrics':bool(center_errors[1]<center_errors[0] and maes[1]<maes[0]),
            'baseline_strongest_sample_um':commercial['wavelength_um'][int(np.argmax(old))],
            'direct_yee_strongest_sample_um':commercial['wavelength_um'][int(np.argmax(new))],
            'direct_yee_component_hashes':second['stages']['device']['electric_component_permittivity_sha256']}


def summarize(directory):
    source=json.loads((directory/'v012-grating-commercial-references.json').read_text())
    lum=source['lumerical']
    meshes=(12,14,24,48,96)
    local={kind:[screen(mesh,kind) for mesh in meshes] for kind in ('volume-average','direct-yee')}
    center=screen(None)
    material=screen(None,material_samples=lum)
    studies=[native_pair(directory/f'v012-grating3d-lumerical-r{mesh}.json',directory/f'v013-grating3d-lumerical-direct-r{mesh}.json',lum) for mesh in (12,14)]
    tidy=json.loads((directory/'v012-grating3d-tidy3d-r14.json').read_text())
    ref=next(x for x in tidy['comparisons'] if x['geometry_matched'])
    return {'wavelength_um':lum['wavelength_um'],'design':'Curved Lumerical grating with physical polished fiber.',
            'prespecified_metrics':'Reduce both the absolute 1550 nm S21-power discrepancy and the mean absolute error over all five original wavelengths. No geometric or material fitting to the target spectrum.',
            'commercial_file_sha256':lum['s_file_sha256'],'published_peak_wavelength_um':lum['samples']['S21']['peak_wavelength_um'],
            'published_s12_minus_s21_pp':(100*(np.array(lum['samples']['S12']['power'])-lum['samples']['S21']['power'])).tolist(),
            'fiber_aperture':[fiber_aperture(directory/f'v012-grating3d-lumerical-r{mesh}.json') for mesh in (12,14)],
            'direct_yee_studies':studies,'slab_phase_screen':{'frozen_material_mesh_controls':local,'continuous_constant_index':center,
             'continuous_tabulated_index':material,'material_shift_nm':1000*(material['estimated_first_order_wavelength_um']-center['estimated_first_order_wavelength_um']),
             'scope':'Independent TE slab dispersion approximation; ignores the curved finite tooth scattering, taper, asymmetric substrate and realistic fiber overlap. The tabulated-material calculation describes the reduced slab, not a dispersive 3D run.'},
            'tidy3d_other_device':{'published':ref['published_power'],'local':tidy['result']['beam_to_guide'],
                'mae_pp':ref['mean_absolute_difference_percentage_points'],'source':ref['url'],'receipt':'v012-grating3d-tidy3d-r14.json',
                'scope':'Independent unchanged finite uniform taper/Gaussian control; the candidate curved-chip model is Lumerical-specific.'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory),indent=2,allow_nan=False)+'\n')
