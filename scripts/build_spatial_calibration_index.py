"""Create the local browser's spatial calibration views from native result receipts."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from backend.calibration_fields import field_plane

BASE = Path(__file__).resolve().parents[1] / 'docs/verification'
RING_URLS = {'Lumerical':'https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction',
             'Tidy3D':'https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/'}
GRATING_URLS = {'Lumerical':'https://optics.ansys.com/hc/en-us/articles/360042305334-Grating-coupler',
                'Tidy3D':'https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/'}


def case_entry(dimension, route, summary, limits):
    return dict(summary=summary, dimension=dimension, routes=[route], limits=limits,
                comparisons=[], recorded=[], series=[], wavelengths_um=[])


def subset(record, keys):
    return {key:record[key] for key in keys if key in record}


def phase_studies():
    """Summaries only; full signed coefficients, plane registrations and masks are linked."""
    studies={'ring-3d':[],'grating-3d':[]}
    for path in sorted(BASE.glob('v014-*-phase*.json')):
        study=json.loads(path.read_text())
        key=study.get('component','')+'-3d'
        if key not in studies or not study.get('native_samples'):
            continue
        samples=[]
        for sample in study['native_samples']:
            central=sample.get('central_vendor_mode_check')
            wavelength=central['port_overlaps'][0]['wavelength_um'] if central else None
            samples.append({'mesh':sample['resolution'],'central_wavelength_um':wavelength,
                            'absolute':central['channels'] if central else {},
                            'passed':sample.get('native_run_passed',True),'valid':sum(sample['valid_bins']),'total':len(sample['valid_bins']),
                            'band':{port:stats['absolute_phase_residual_rad'] for port,stats in
                                    sample.get('single_central_basis_band',{}).get('channels',{}).items()},
                            'commercial':{k:v['phase_increment_delta_rad'] for k,v in sample.get('commercial_comparison',{}).items()
                                          if 'phase_increment_delta_rad' in v},
                            'commercial_power':{k:v['power_delta'] for k,v in sample.get('commercial_comparison',{}).items()
                                                if 'power_delta' in v}})
        pairs=[{'meshes':pair['meshes'],'channels':{port:{
                'power':values['power_delta']['max_abs'],'complex':values['same_gauge_complex_delta']['max_abs'],
                'phase':values['same_gauge_phase_delta_rad']['max_abs'],'strong':values['same_gauge_phase_delta_rad']['count']}
                for port,values in pair['channels'].items()}} for pair in study['mesh_pairs']]
        studies[key].append({'receipt':path.name,'vendor':study['vendor'],'amplitude_floor':study['amplitude_floor'],
                             'samples':samples,'pairs':pairs})
    return studies


def spatial_index():
    index = {
        'ring-3d':case_entry('Finite-height, two-bus loaded ring','FDTDX NVIDIA',
            'The incident straight-bus run calibrates all four output ports against the injected guided mode.',
            'Finite observation time, material dispersion and mesh affect the absolute pole wavelengths.'),
        'grating-3d':case_entry('Finite transverse width and taper','FDTDX NVIDIA',
            'The separately measured incident beam or fiber field normalizes the output TE mode.',
            'The compact control and the published geometries use different widths, tapers and incident fields.'),
        'multimode-tm':case_entry('Planar Hz polarization','Meep CPU',
            'Each side excites both TM modes independently. All coefficients refer to the selected physical port planes.',
            'Compare the independent modal interface reference and the measured effects of moving the planes.'),
        'layout-ports-3d':case_entry('Full-vector finite-height silicon extrusion','FDTDX NVIDIA',
            'A measured straight incident reference accompanies each excitation of the selected polygon ports.',
            'These are cardinal-axis quasi-TE ports with a fixed silicon height and oxide background.'),
        'layout-ports-2d':case_entry('Planar Ez polarization','Meep CPU',
            'An isolated straight lead calibrates each excitation of the polygon design.',
            'These finite apertures collect the fundamental guided mode. Each new layout has its own convergence requirements.'),
    }
    best_ring, best_grating = {}, {}
    ring_order = {}
    diagnostics_file = BASE/'v014-power-diagnostics.json'
    if not diagnostics_file.is_file():
        diagnostics_file = BASE/'v013-power-diagnostics.json'
    diagnostics = json.loads(diagnostics_file.read_text())['records']
    gates_path = BASE/'v014-accuracy-gates.json'
    gates = json.loads(gates_path.read_text()) if gates_path.is_file() else {}
    if gates and gates['diagnosis_sha256'] != hashlib.sha256(diagnostics_file.read_bytes()).hexdigest():
        raise ValueError('Accuracy gates belong to a different power comparison.')
    paths = {*BASE.glob('v012-*.json'), *BASE.glob('v013-ring3d-*.json'), *BASE.glob('v013-grating3d-*.json'), *BASE.glob('v014-ring3d-*.json'), *BASE.glob('v014-grating3d-*.json')}
    stability_path=BASE/'v014-ring-wall-stability.json'
    stability=json.loads(stability_path.read_bytes())
    for path in sorted(paths):
        if path.name == 'v012-calibrated-index.json':
            continue
        data = json.loads(path.read_text())
        if not isinstance(data,dict):
            continue
        case = {'loaded-ring-3d':'ring-3d','finite-grating-3d':'grating-3d'}.get(data.get('case_id'), data.get('case'))
        if case not in index:
            continue
        if case in ('ring-3d','grating-3d'):
            if data.get('actual_device','').split(':')[0] not in ('cuda','gpu'):
                continue
            raw = BASE/data.get('arrays_file','')
            if raw.parent != BASE or not raw.is_file() or not data.get('arrays_sha256'):
                continue
            if hashlib.sha256(raw.read_bytes()).hexdigest() != data['arrays_sha256']:
                raise ValueError(f'Numerical archive changed for {path.name}')
            if case=='ring-3d':
                if path.name==stability['source_receipt'] and stability['usable_spectrum'] is False:
                    if (stability['source_report_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()
                            or stability['source_arrays_sha256']!=data['arrays_sha256']):
                        raise ValueError('Invalid whole-run stability rejection.')
                    model=data['physical_model']
                    label=f"{data['vendor'].capitalize()} · {model['resolution']} cells/µm · {model['ring_interface']}"
                    index[case].setdefault('rejected',[]).append({'receipt':path.name,'diagnostic':stability_path.name,
                        'label':label,'reason':stability['reason']})
                    continue
                if not data.get('measurement_complete') or not data['valid_bins']:
                    continue
                vendor=data['vendor']
                series=data['series'][vendor]
                panel={'series':{vendor:subset(series,('wavelength_um','through','drop','reflection','add_side','valid_bins'))},
                       **subset(data,('case_id','vendor','vendor_comparison','loaded_loss'))}
                mesh=data['physical_model']['resolution']
                model=data['physical_model']
                label=(f"{vendor.capitalize()} · {mesh} cells/µm · {data['duration_um_c']:g} µm/c · "
                       f"{model['pml']} PML cells · {model['aperture_um']:g} µm port")
                if model.get('ring_interface','legacy') != 'legacy':
                    label += f" · {model['ring_interface']}"
                if model.get('ring_material') == 'palik-ade':
                    label += ' · dispersive Palik fit'
                measurement=f"{data['valid_bins']} of {data['total_bins']} samples passed the observation-window checks."
                nominal=path.name.endswith(f"-t{int(data['duration_um_c'])}.json")
                # The first record opens in the GUI. Start with the qualified default
                # geometry and mesh; keep the shortened time-window diagnostics accessible.
                wall=model.get('ring_interface')=='wall-tensor'
                ring_order[path.name]=(not data['passed'],vendor!='lumerical',not wall,not nominal,-mesh,-data['duration_um_c'])
                key=(data['passed'],mesh,wall,data['duration_um_c'],nominal)
                if data.get('measured_drop_fsr_nm') is not None and (vendor not in best_ring or key>best_ring[vendor][0]):
                    best_ring[vendor]=(key,data)
            else:
                if not data.get('passed'):
                    continue
                settings=data['settings']
                geometry,mesh=settings['geometry'],settings['resolution']
                label=f'{geometry} · {mesh} cells/µm'
                if settings.get('grating_material_model','volume-average') != 'volume-average':
                    label += f" · {settings['grating_material_model']}"
                if settings.get('grating_domain') == 'mirror-y':
                    label += ' · mirror symmetry'
                if settings.get('grating_spectrum') == 'dense':
                    label += ' · 21 wavelengths'
                panel=subset(data,('case_id','vendor','result','radiation','comparisons','settings'))
                measurement=f"Measured input and output powers on {data['device_name']}."
                key=(mesh,settings.get('grating_material_model')=='direct-yee',
                     settings.get('grating_spectrum','standard')=='standard',settings.get('grating_domain','full')=='full')
                if geometry in ('lumerical','tidy3d') and (geometry not in best_grating or key>best_grating[geometry][0]):
                    best_grating[geometry]=(key,data)
            detail = diagnostics.get(path.name)
            if detail:
                if (detail['source_report_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest() or
                        detail['native_arrays_sha256'] != data['arrays_sha256']):
                    raise ValueError(f'Regenerate power diagnostics for {path.name}.')
                panel['power_diagnostic'] = {key:value for key,value in detail.items() if key!='additional_spectra'}
                if detail.get('additional_spectra'):
                    panel['vendor_comparison']['channels'].update(
                        {name:{'paired_spectrum':series} for name,series in detail['additional_spectra'].items()})
            qualification=gates.get('records',{}).get(path.name)
            if qualification:
                panel['accuracy_criteria']=qualification['criteria']
            clocks = data.get('stages',{})
            time_key = 'elapsed_s' if case=='ring-3d' else 'stepping_seconds'
            if all(clocks.get(role,{}).get('performance') for role in ('reference','device')):
                ref_time=clocks['reference']['performance']['synchronized_step_seconds']
                dev_time=clocks['device']['performance']['synchronized_step_seconds']
                measurement+=f" Synchronized GPU stepping took {ref_time:.1f} s for the reference and {dev_time:.1f} s for the device."
            elif all(time_key in clocks.get(role,{}) for role in ('reference','device')):
                reference_s=clocks['reference'][time_key]
                device_s=clocks['device'][time_key]
                basis='acquisition' if case=='ring-3d' else 'time stepping'
                measurement+=f" Recorded NVIDIA {basis}: {reference_s:.0f} s for the incident reference, {device_s:.0f} s for the device."
            view={'label':label,'receipt':path.name,'report':panel,'qualification':measurement}
            field=field_plane(data,raw)
            if field is not None:
                view['field']=field
        else:
            if not data.get('passed') or 's_real' not in data:
                continue
            panel=subset(data,('channels','ports','port_order','wavelength_um','s_real','s_imag','power'))
            label=f"{data.get('parameters',{}).get('resolution','Native')} cells/µm"
            view={'label':label,'receipt':path.name,'report':panel}
            source=data.get('source_arrays',{})
            raw=BASE/source['file']
            if raw.parent!=BASE or hashlib.sha256(raw.read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError(f'Public planar archive changed for {path.name}')
            field=field_plane(data,raw)
            if field is not None:
                view['field']=field
        index[case]['recorded'].append(view)
    index['ring-3d']['recorded'].sort(key=lambda record: ring_order[record['receipt']])
    for brand,url in RING_URLS.items():
        result = best_ring.get(brand.lower())
        if result:
            r=result[1]
            model=r['physical_model']
            note=(f"At {model['resolution']} cells/µm with {model.get('ring_interface','legacy')} walls and "
                  f"{r['duration_um_c']:g} µm/c observation, central spacing is {r['measured_drop_fsr_nm']:.2f} nm "
                  f"versus {r['published_drop_fsr_nm']:.2f} nm published.")
            delta=r.get('vendor_comparison',{}).get('channels',{}).get('drop',{}).get('mean_absolute_power_difference')
            if delta is not None:
                note+=f" Mean absolute drop-power difference over the common qualified band: {delta:.4f}."
        else:
            note='Published 3D through and drop spectra for this ring geometry.'
        index['ring-3d']['comparisons'].append(dict(vendor=brand,url=url,description=note))
    for brand,url in GRATING_URLS.items():
        result=best_grating.get(brand.lower())
        note='Published finite grating geometry and guided-output comparison.'
        if result:
            r=result[1]
            reference=next(c for c in r['comparisons'] if c.get('geometry_matched') and c['vendor']==brand)
            j=r['result']['wavelength_um'].index(1.55)
            published=reference.get('published_power',reference.get('published_s21_power'))
            model=r['settings']
            note=(f"At 1.55 µm on the {model['resolution']} cells/µm mesh "
                  f"({model.get('grating_material_model','volume-average')}, {model.get('grating_domain','full')}), the measured guided output is "
                  f"{r['result']['beam_to_guide'][j]*100:.2f}% versus {published[j]*100:.2f}% published. "
                  f"The mean absolute difference over {len(published)} wavelengths is {reference['mean_absolute_difference_percentage_points']:.2f} percentage points.")
        index['grating-3d']['comparisons'].append(dict(vendor=brand,url=url,description=note))
    for case in ('multimode-tm','layout-ports-2d','layout-ports-3d'):
        index[case]['comparisons'] = [
            dict(vendor='Lumerical',url='https://optics.ansys.com/hc/en-us/articles/360042304134-Asymmetric-slab-dielectric-waveguide',
                 description='The published asymmetric-slab reference checks guided indices separately from the device scattering matrix.'),
            dict(vendor='Tidy3D',url='https://www.flexcompute.com/tidy3d/examples/notebooks/EMESolver/',
                 description=('The finite-height reference checks port modes separately from the measured circuit matrix.' if case=='layout-ports-3d' else 'A local Tidy3D mode solve checks the port modes. The complete planar matrix is compared with an independent wave-equation calculation.'))]
    for case,studies in phase_studies().items():
        index[case]['phase_studies']=studies
    return index


if __name__=='__main__':
    index_json = json.dumps(spatial_index(), separators=(',', ':'), allow_nan=False)
    with NamedTemporaryFile(mode='w', dir=BASE, delete=False) as temporary:
        temporary.write(index_json + '\n')
    destination = BASE/'v012-calibrated-index.json'
    staging = Path(temporary.name)
    staging.chmod(0o644)
    staging.replace(destination)
