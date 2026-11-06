"""Measure guided-input emission into a finite air aperture and reciprocal Gaussian beam."""
import json
import numpy as np

from scripts.grating_angular_spectrum import air_angular_flux
from scripts.lumerical_grating3d import reflected_fiber_amplitude


def observables(ref, device, beam, resolution):
    """Return measured channel powers and angular flux from three native acquisitions."""
    from scripts.calibrate_finite_grating import normal_flux, WL
    dx=1/resolution
    incident_guide=abs(ref['output_amplitudes'][:,0])**2
    incident_beam=-normal_flux(beam['upper'],dx)
    if np.any(incident_guide<=0) or np.any(incident_beam<=0):
        raise ValueError('Receiving Gaussian and guided reference must carry positive incident spectral power.')
    scatter=device['upper']-ref['upper']
    # Normalize the outgoing Gaussian overlap to both independently measured inputs.
    overlap=reflected_fiber_amplitude(scatter,beam['upper'],incident_beam,dx)
    channel=overlap*np.sqrt(incident_beam/incident_guide)
    reflection=(device['output_amplitudes'][:,1]-ref['output_amplitudes'][:,1])/ref['output_amplitudes'][:,0]
    angles=air_angular_flux(scatter,dx,WL)
    binned=angles['polar_bin_power']/incident_guide[:,None]
    return {'incident_guide':incident_guide,'incident_beam':incident_beam,
            'gaussian_channel':channel,'guide_reflection':reflection,
            'angle_fraction':binned,'angle_edges_deg':angles['polar_edges_deg'],
            'upgoing_fraction':binned.sum(axis=1),
            'direct_aperture_fraction':normal_flux(scatter,dx)/incident_guide,
            **{'angular_'+key:value for key,value in angles.items()}}


def publish(m, output, ref_path, rr, dev_path, dr, beam_path, br):
    from scripts.calibrate_finite_grating import WL,file_hash,TIDY_URL,ANSYS_URL
    with np.load(ref_path/'arrays.npz',allow_pickle=False) as ref, np.load(dev_path/'arrays.npz',allow_pickle=False) as device, np.load(beam_path/'arrays.npz',allow_pickle=False) as beam:
        keys=observables(ref,device,beam,m['resolution'])
        stored={prefix+'_'+key:archive[key] for prefix,archive in [('ref',ref),('dev',device),('beam',beam)] for key in archive.files}
        pml=dr['pml_cells']
        horizontal=device['grid_x_um'][pml:-pml].tolist()
        vertical=device['grid_z_um'][pml:-pml].tolist()
    stored.update(keys)
    efficiency=abs(keys['gaussian_channel'])**2
    returned=abs(keys['guide_reflection'])**2
    upper=keys['upgoing_fraction']
    reference=keys['incident_guide']; incident_beam=keys['incident_beam']
    settled=rr['settled'] and dr['settled'] and br['settled']
    valid=((reference>=.01*max(reference)) & (incident_beam>=.01*max(incident_beam)) & (efficiency<=1.02) &
           (upper+returned<=1.05) & (efficiency<=upper+.03) & np.isfinite(efficiency) & settled)
    stored['valid']=valid
    np.savez_compressed(output/'arrays.npz',**stored)
    report={'schema_version':1,'case_id':'finite-grating-3d','vendor':'Workbench','actual_device':dr['actual_device'],
            'device_name':dr['device_name'],'passed':bool(np.all(valid)),'settings':m,
            'stages':{'reference':rr,'device':dr,'beam_reference':br},'arrays_sha256':file_hash(output/'arrays.npz'),
            'result':{'resolution':m['resolution'],'wavelength_um':WL.tolist(),'guide_to_beam':efficiency.tolist(),
                      'guide_radiation_up':upper.tolist(),'guide_reflection':returned.tolist(),'valid_energy_bins':valid.tolist(),
                      'center_guide_to_beam':float(efficiency[2]),'finite_aperture_signed_flux':keys['direct_aperture_fraction'].tolist()},
            'radiation':{'polar_edges_deg':keys['angle_edges_deg'].tolist(),'fraction_of_incident':keys['angle_fraction'].tolist(),
                         'scope':'Propagating upward power estimated on a finite rectangular aperture. Each polar bin integrates azimuth; angles from 80° to 90° and rays beyond the aperture are excluded.'},
            'field_preview':{'array':'dev_center_fields','index':[1],'axis_order':'xy','horizontal_axis':'x','vertical_axis':'z',
                             'horizontal_um':horizontal,'vertical_um':vertical,'component':'Ey','wavelength_um':1.55,
                             'description':'Real Ey phasor near y = 0 with the silicon guide excited.'},
            'comparisons':[{'vendor':'Tidy3D','url':TIDY_URL,'geometry_matched':False,'scope':'The vendor describes reciprocal guided excitation and Gaussian overlap. The compact device has different physical dimensions.'},
                           {'vendor':'Lumerical','url':ANSYS_URL,'geometry_matched':False,'scope':'Its published reverse S12 is for curved teeth and a polished fiber.'}]}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Guided-to-Gaussian:',efficiency.tolist(),'upper aperture:',upper.tolist(),'qualified:',report['passed'],flush=True)
    return report
