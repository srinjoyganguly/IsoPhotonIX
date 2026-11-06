"""Verify native fields and collect finite-height polygon-port controls."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibrate_layout_ports3d import SOURCES, file_digest, json_digest, local_phasors, modal_amplitudes, normalize

ANSYS = 'https://optics.ansys.com/hc/en-us/articles/360042799933-Bent-waveguide-analysis'
TIDY = 'https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/how-do-i-run-the-mode-solver-locally.html'
RUNS = ('elbow-r16','straight-x-r12','straight-y-r12','elbow-r12','elbow-shifted-r12')


def mode_controls(folder):
    result = {}
    for engine in ('tidy3d','mpb'):
        first = json.loads((folder/f'{engine}.json').read_text())
        last = json.loads((folder/f'{engine}96.json').read_text())
        worker = ROOT/'scripts/compare_layout_ports3d_modes.py'
        if first['source_sha256']!=file_digest(worker) or last['source_sha256']!=first['source_sha256']:
            raise ValueError('Unmatched transverse solver source.')
        if {k:v for k,v in first.items() if k!='rows'}!={k:v for k,v in last.items() if k!='rows'}:
            raise ValueError('The two transverse studies have different physical settings.')
        rows = dict(first['rows'],**last['rows'])
        residual = max(float(np.max(abs(np.subtract(x['x']['neff'],x['y']['neff'])))) for x in rows.values())
        if residual>1e-6:
            raise ValueError('Orthogonal lead eigenmodes have inconsistent indices.')
        result[engine] = dict(first,rows=rows,max_axis_neff_delta=residual)
    frame_check = json.loads((folder/'frames.json').read_text())
    if any(file_digest(ROOT/name)!=expected for name,expected in frame_check['sources'].items()):
        raise ValueError('The signed native frame source differs from its receipt.')
    if len(frame_check['rows'])!=8 or max(row['reciprocal_projection_wrong_direction_ratio'] for row in frame_check['rows'])>1e-5:
        raise ValueError('The signed native source projection did not qualify on both meshes and axes.')
    return {'local_transverse_modes':result,'local_tidy3d_documentation':TIDY,'native_frame_check':frame_check,
            'mode_solver_spread_neff_96':np.subtract(result['tidy3d']['rows']['96']['x']['neff'],
                                                       result['mpb']['rows']['96']['x']['neff']).tolist(),
            'ansys_lumerical_reference':{'article':ANSYS,'kind':'published Lumerical MODE curved-mode loss and straight-to-bend overlap estimate',
                'radius_um':1.5,'curved_loss_db_per_um':.0009,'straight_bent_mode_overlap':.988,'estimated_quarter_bend_loss_db':.107,
                'scope':'Published circular bend and two straight/bent mode junctions. The Workbench default is an abrupt polygon elbow. The published loss estimate has a different geometry and observable.'}}


def verify_run(path, checkpoints):
    report = json.loads((path/'report.json').read_text())
    if (report['case']!='layout-ports-3d' or report['actual_device'].split(':')[0] not in ('gpu','cuda')
            or not report['passed'] or any(not p['settled'] for p in report['stages'])):
        raise ValueError(f'Native setup or stage qualification failed in {path.name}.')
    if report['sources']!={name:file_digest(ROOT/name) for name in SOURCES}:
        raise ValueError('The recorded native source differs from the exported implementation.')
    res = report['parameters']['resolution']
    ports = report['sampled_ports']
    stages = {s['label']:s for s in report['stages']}
    expected = {f'port{i}-{kind}' for i in range(len(ports)) for kind in ('reference','device')}
    if len(stages)!=len(report['stages']) or set(stages)!=expected:
        raise ValueError('Incomplete native source or device set.')
    results = {}
    with np.load(path/'arrays.npz',allow_pickle=False) as native:
        for label,stage in stages.items():
            entry = checkpoints/stage['stage_key']
            receipt = json.loads((entry/'receipt.json').read_text())
            spec,rr = receipt['specification'],receipt['report']
            launch = int(label.partition('-')[0][4:])
            reference = label.endswith('-reference')
            if (json_digest(spec)!=stage['stage_key'] or json_digest(rr)!=receipt['report_sha256']
                    or rr['field_device']!=report['actual_device'] or not rr['settled']
                    or spec['sources']!=report['sources'] or spec['design']!=report['parameters']['design']
                    or spec['resolution']!=res or spec['launch']!=launch or spec['reference']!=reference
                    or receipt['archive_sha256']!=stage['archive_sha256'] or file_digest(entry/'arrays.npz')!=stage['archive_sha256']):
                raise ValueError('Stage provenance or recorded arrays differ from the run.')
            with np.load(entry/'arrays.npz',allow_pickle=False) as data:
                for k in data.files:
                    np.testing.assert_array_equal(data[k],native[f'{label}_{k}'])
                axes = [ports[launch]['axis']]*2 if reference else [p['axis'] for p in ports]
                amplitudes = []
                for i,axis in enumerate(axes):
                    a,_ = modal_amplitudes(local_phasors(data[f'phasors_{i}'],axis),data[f'basis_e_{i}'],data[f'basis_h_{i}'],1/res**2)
                    amplitudes.append(a)
                np.testing.assert_allclose(amplitudes,data['amplitudes'],rtol=2e-6,atol=1e-30)
                results[label] = np.asarray(amplitudes)
        pairs = []
        for i in range(len(ports)):
            ref,dev = f'port{i}-reference',f'port{i}-device'
            pairs.append((results[ref],results[dev],stages[ref],stages[dev]))
        model = {'ports':ports,'frequency_per_um':1/np.asarray(report['wavelength_um'])}
        s, incident, through, valid = normalize(model,pairs)
        for k,v in [('s',s),('incident',incident),('reference_through',through)]:
            np.testing.assert_allclose(native[k],v,rtol=5e-6,atol=1e-10)
        np.testing.assert_allclose(s,np.array(report['s_real'])+1j*np.array(report['s_imag']),rtol=5e-6,atol=1e-10)
        np.testing.assert_allclose(abs(s)**2,report['power'],rtol=5e-6)
        if not np.all(valid):
            raise ValueError('Incomplete matrix excitation.')
        archive = {k:native[k] for k in ('s','incident','reference_through','wavelength_um')}
        archive.update({label+'_amplitudes':a for label,a in results.items()})
        if path.name=='elbow-r16':
            archive[report['field_preview']['array']] = native[report['field_preview']['array']]
            for k in native.files:
                if k.startswith('port0-device_') and k.partition('_')[2].startswith(('phasors_','basis_','midplane_permittivity')):
                    archive[k] = native[k]
    return dict(report,native_arrays_sha256=file_digest(path/'arrays.npz'),native_report_sha256=file_digest(path/'report.json')),archive


def delta(a,b):
    visible = np.minimum(abs(a),abs(b))>=.02
    return {'power_abs_delta_max':float(np.max(abs(abs(a)**2-abs(b)**2))),
            'complex_abs_delta_max':float(np.max(abs(a-b))),'phase_floor':.02,
            'phase_delta_rad_max':float(np.max(abs(np.angle(a[visible]*b[visible].conj())))) if np.any(visible) else None}


def publish(folder, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    references = mode_controls(folder)
    studies, combined, records = {},{},{}
    for name in RUNS:
        if not (folder/name/'report.json').exists():
            continue
        report, data = verify_run(folder/name,folder/'checkpoint')
        records[name] = report
        studies[name] = {**{key:report[key] for key in ('checks','channels','reference_through_power','actual_device','native_report_sha256','native_arrays_sha256')},
                         'complete_native_stages':len(report['stages']),'resolution':report['parameters']['resolution']}
        prefix = '' if name=='elbow-r16' else name+'_'
        combined.update({prefix+k:v for k,v in data.items()})
    if 'elbow-r16' not in records:
        raise ValueError('A fully measured native default elbow is required.')
    report = records['elbow-r16']
    checks = {}
    waves = np.asarray(report['wavelength_um'])
    for name in ('straight-x-r12','straight-y-r12'):
        if name not in studies:
            continue
        matrix = combined[name+'_s']
        reference = combined[name+'_reference_through']
        trans = np.stack([matrix[:,1,0],matrix[:,0,1]],axis=1)
        p = records[name]['sampled_ports']
        length = abs(p[0]['sampled_center_um'][p[0]['axis']]-p[1]['sampled_center_um'][p[1]['axis']])
        neff = np.asarray(records[name]['stages'][0]['mode_neff'][p[0]['name']])
        prediction = np.exp(2j*np.pi*length*neff/waves)
        checks[name] = {'straight_reflection_power_max':float(max(abs(matrix[:,0,0]).max(),abs(matrix[:,1,1]).max())**2),
                        'through_vs_reference':delta(trans,reference),'transmission_power_error_max':float(np.max(abs(abs(trans)**2-1))),
                        'through_phase_vs_mode_index_rad_max':float(np.max(abs(np.angle(trans*prediction[:,None].conj()))))}
    if {'straight-x-r12','straight-y-r12'}<=set(studies):
        checks['x_y_straight_exchange'] = delta(combined['straight-x-r12_s'],combined['straight-y-r12_s'])
    if 'elbow-r12' in studies:
        checks['elbow_mesh_12_to_16'] = delta(combined['elbow-r12_s'],combined['s'])
    if {'elbow-r12','elbow-shifted-r12'}<=set(studies):
        base = records['elbow-r12']
        moved = records['elbow-shifted-r12']
        neffs = []
        displacements = []
        for p,q in zip(base['sampled_ports'],moved['sampled_ports']):
            i = p['axis']
            displacements.append(-p['inward_sign']*(q['sampled_center_um'][i]-p['sampled_center_um'][i]))
            neffs.append(base['stages'][1]['mode_neff'][p['name']])
        propagation = np.exp(2j*np.pi*np.asarray(neffs).T*np.array(displacements)/waves[:,None])
        prediction = combined['elbow-r12_s']*propagation[:,:,None]*propagation[:,None,:]
        # The isolated through paths also measure the grid's actual longitudinal
        # propagation. Resolve each phase wrap by the nearby transverse mode index.
        straight = combined['elbow-r12_reference_through']
        lengths = np.asarray([2*abs(p['sampled_center_um'][p['axis']]) for p in base['sampled_ports']])
        beta_mode = 2*np.pi*np.asarray(neffs).T/waves[:,None]
        beta_fdtd = beta_mode + np.angle(straight*np.exp(-1j*beta_mode*lengths))/lengths
        discrete_phase = np.exp(1j*beta_fdtd*np.array(displacements))
        discrete_prediction = combined['elbow-r12_s']*discrete_phase[:,:,None]*discrete_phase[:,None,:]
        checks['shifted_planes'] = {'displacements_um':displacements,
            'transverse_mode_index_prediction':delta(combined['elbow-shifted-r12_s'],prediction),
            'straight_reference_phase_prediction':delta(combined['elbow-shifted-r12_s'],discrete_prediction),
            'straight_reference_phase_neff':(beta_fdtd*waves[:,None]/(2*np.pi)).tolist()}
    lead = np.asarray(report['stages'][1]['mode_neff'][report['channels'][0]])
    for engine,control in references['local_transverse_modes'].items():
        checks[f'lead_neff_vs_{engine}_96'] = {'native_minus_reference':(lead-np.array(control['rows']['96']['x']['neff'])).tolist()}
    rawfile = destination.with_name(destination.stem+'-arrays.npz')
    tmp = rawfile.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**combined)
    tmp.replace(rawfile)
    source = {'file':rawfile.name,'sha256':file_digest(rawfile)}
    report.update(studies=studies,controls=checks,references=references,source_arrays=source,
                  material_sampling='Eight by eight lateral quadrature points per Yee component cell, exact slab thickness overlap. Ex/Ey use arithmetic volume mixing. Ez uses harmonic mixing across the flat vertical interface and an arithmetic lateral area average.',
                  qualification='All published matrices were reconstructed from native E/H phasors and signed reciprocal overlaps. Mesh and reference-plane differences are reported separately.')
    tmp = destination.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(destination)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    publish(args.measurements,args.output)
