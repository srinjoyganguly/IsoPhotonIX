"""Audit and expose measured complex modal amplitudes without power/phase fitting."""
import csv
from functools import lru_cache
import io
import math
from pathlib import Path

import numpy as np

STAGES = ('reference_forward', 'device_forward', 'reference_reverse', 'device_reverse')
ENTRIES = (('S11',0,0), ('S21',1,0), ('S12',0,1), ('S22',1,1))
PHASE_FLOOR = 1e-8


def validate_sparameters(arrays, metadata=None):
    keys = ('wavelength_um','frequency_per_um','s','incident_amplitude','outgoing_amplitude',
            'reference_through','valid_bin','mode_neff','mode_valid')
    a = {name:np.asarray(arrays[name]) for name in keys}
    w = a['wavelength_um']
    if w.ndim != 1 or not 3 <= len(w) <= 9:
        raise ValueError('Modal spectra require three to nine one-dimensional wavelength samples.')
    n = len(w)
    shapes = {'frequency_per_um':(n,), 's':(n,2,2), 'incident_amplitude':(n,2),
              'outgoing_amplitude':(n,2,2), 'reference_through':(n,2),
              'valid_bin':(n,), 'mode_neff':(n,), 'mode_valid':(n,)}
    if any(a[key].shape != shape for key,shape in shapes.items()):
        raise ValueError('Modal coefficient, coordinate and validity dimensions disagree.')
    for key in ('wavelength_um','frequency_per_um'):
        if a[key].dtype.kind not in 'fiu' or not np.isfinite(a[key]).all():
            raise ValueError('Modal frequency coordinates must be finite real values.')
    if np.any(w <= 0) or np.any(np.diff(w) <= 0) or not np.allclose(a['frequency_per_um'],1/w,rtol=1e-7,atol=0):
        raise ValueError('Increasing positive wavelengths must match their reciprocal frequencies.')
    for key in ('valid_bin','mode_valid'):
        if a[key].dtype.kind != 'b':
            raise ValueError('Mode and spectral validity must be recorded as boolean arrays.')
    valid = a['valid_bin']
    for key in ('s','incident_amplitude','outgoing_amplitude','reference_through','mode_neff'):
        if a[key].dtype.kind != 'c':
            raise ValueError(f'{key} must retain complex coefficients, including their phase.')
        if np.isinf(a[key]).any():
            raise ValueError('Infinite modal coefficients are invalid.')
        if key not in ('s','reference_through') and not np.isfinite(a[key]).all():
            raise ValueError('Raw incident/outgoing coefficients and mode indices must be finite.')
        if not np.isfinite(a[key][valid]).all():
            raise ValueError('Qualified bins require finite modal coefficients.')
    power = abs(a['incident_amplitude'])**2
    maxima = np.max(power,axis=0)
    if np.any(maxima <= 0):
        raise ValueError('Both reference directions require measured nonzero incident modes.')
    strong = np.all(power > maxima*1e-6,axis=1)
    if np.any(valid & (~strong | ~a['mode_valid'])):
        raise ValueError('Qualified bins require guided modes and sufficient incident power in both directions.')
    expected = np.divide(a['outgoing_amplitude'],a['incident_amplitude'][:,None,:],
                         out=np.full((n,2,2),complex(np.nan,np.nan)),where=power[:,None,:]>maxima[None,None,:]*1e-6)
    measured = np.isfinite(a['s']) & strong[:,None,None]
    if not np.allclose(a['s'][measured],expected[measured],rtol=2e-5,atol=1e-8):
        raise ValueError('S does not equal the saved outgoing modal amplitude divided by its incident reference.')
    if metadata is not None:
        config = metadata.get('config',{})
        if config.get('study_type') != 'waveguide-ports-3d' or config.get('frequency_points') != n:
            raise ValueError('The modal spectrum does not match its saved study type or sample count.')
        expected_f = np.linspace(1/config['wavelength_max_um'],1/config['wavelength_min_um'],n)[::-1]
        if not np.allclose(a['frequency_per_um'],expected_f,rtol=1e-7,atol=0):
            raise ValueError('Modal frequencies differ from the requested band.')
        settlement = metadata.get('settlement',{})
        for name in STAGES:
            stage = settlement.get(name)
            if not isinstance(stage,dict) or type(stage.get('settled')) is not bool:
                raise ValueError('All four directional settling records are required.')
            thresholds = stage.get('thresholds',{})
            if not isinstance(thresholds,dict):
                raise ValueError('Settling thresholds must be recorded as an object.')
            for metric,setting,threshold in (('energy_ratio','decay_by','energy_decay'),
                                            ('spectral_relative_change','spectrum_tolerance','spectral_relative_change')):
                requested = config.get(setting)
                if type(requested) not in (float,int) or not math.isfinite(requested) or requested <= 0:
                    raise ValueError('The requested settling thresholds must be finite and positive.')
                recorded = ([stage[setting]] if setting in stage else []) + ([thresholds[threshold]] if threshold in thresholds else [])
                if any(type(v) not in (float,int) or not math.isfinite(v) or not math.isclose(v,requested,rel_tol=1e-12,abs_tol=0) for v in recorded):
                    raise ValueError('Recorded settling thresholds differ from the requested study.')
                value = stage.get(metric)
                if stage['settled'] and (type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= requested):
                    raise ValueError(f'{name} claims settling with missing or contradictory {metric}.')
        if np.any(valid) and not all(settlement[name]['settled'] for name in STAGES):
            raise ValueError('Unsettled directional runs cannot qualify a complete S-matrix.')
        if metadata.get('sparameters_valid') is not bool(np.all(valid)):
            raise ValueError('The recorded matrix validity disagrees with its bin flags.')
        ports = metadata.get('ports',{})
        if any(not isinstance(ports.get(key),str) or not ports[key].strip() for key in ('phase_convention','mode_gauge','normalization')):
            raise ValueError('Complex S-parameters require recorded phase, mode gauge and normalization conventions.')
        requested_port = config['geometry']['domain']['x']/2-config['pml_um']-1.5
        for key,sign in (('port_1_x_um',-1),('port_2_x_um',1)):
            value = ports.get(key)
            if type(value) not in (float,int) or not math.isfinite(value) or abs(value-sign*requested_port) > .51/config['resolution']:
                raise ValueError('The recorded physical reference plane differs from the requested port placement.')
        cladding = max([config['cladding_index']] + [o['refractive_index'] for o in config['geometry']['objects'] if o['id']=='substrate'])
        guided = (a['mode_neff'].real>cladding)&(abs(a['mode_neff'].imag)<1e-5)
        if np.any(a['mode_valid'] & ~guided):
            raise ValueError('A qualified mode must be guided above the cladding light line in this lossless study.')
    for value in a.values():
        value.setflags(write=False)
    return a


@lru_cache(maxsize=2)
def _read(filename,modified_ns,size):
    del modified_ns,size
    with np.load(filename,allow_pickle=False) as arrays:
        return validate_sparameters(arrays)


def stored_sparameters(output,metadata=None):
    path = Path(output)/'sparameters.npz'
    stat = path.stat()
    return validate_sparameters(_read(str(path.resolve()),stat.st_mtime_ns,stat.st_size),metadata)


def _masked(values,mask):
    return [float(value) if ok else None for value,ok in zip(values,mask)]


def sparameters_response(a,metadata):
    valid = a['valid_bin']
    parameters = {}
    for name,row,column in ENTRIES:
        value = a['s'][:,row,column]
        phase_valid = valid & (abs(value)>PHASE_FLOOR)
        unwrapped = np.full(len(valid),np.nan)
        # Unwrap each contiguous qualified segment independently, never bridge
        # a masked sample or attach a phase to a numerically zero amplitude.
        for indices in np.split(np.flatnonzero(phase_valid),np.flatnonzero(np.diff(np.flatnonzero(phase_valid))!=1)+1):
            if len(indices):
                unwrapped[indices] = np.unwrap(np.angle(value[indices]))
        parameters[name] = {'real':_masked(value.real,valid), 'imag':_masked(value.imag,valid),
            'magnitude':_masked(abs(value),valid), 'power':_masked(abs(value)**2,valid),
            'phase_rad':_masked(np.angle(value),phase_valid),
            'phase_unwrapped_rad':_masked(unwrapped,phase_valid), 'phase_valid':phase_valid.tolist()}
    column_power = np.sum(abs(a['s'])**2,axis=1)
    return {'wavelength_um':a['wavelength_um'].tolist(), 'frequency_per_um':a['frequency_per_um'].tolist(),
            'valid_bin':valid.tolist(), 'mode_valid':a['mode_valid'].tolist(),
            'mode_neff_real':a['mode_neff'].real.tolist(), 'mode_neff_imag':a['mode_neff'].imag.tolist(),
            'parameters':parameters,
            'incident_modal_power':(abs(a['incident_amplitude'])**2).tolist(),
            'column_power':[_masked(column_power[:,col],valid) for col in (0,1)],
            'unaccounted_modal_fraction':[_masked(1-column_power[:,col],valid) for col in (0,1)],
            'reciprocity_abs':_masked(abs(a['s'][:,0,1]-a['s'][:,1,0]),valid),
            'phase_floor':PHASE_FLOOR, 'metadata':metadata,
            'matrix_order':'S[frequency,outgoing port,incoming port]; S21 is transmission from port 1 to port 2.',
            'phase_note':'Unwrapped radians within contiguous valid segments. Phase is undefined when amplitude is at most 1e-8. Physical monitor reference planes; no automatic de-embedding.',
            'residual_note':'Unaccounted fundamental-mode power includes other modes, radiation and numerical error; it is not absorption.'}


def sparameters_csv(a):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    columns = ['wavelength_um','frequency_per_um']
    for name,_,_ in ENTRIES:
        columns += [f'{name}_{key}' for key in ('real','imag','magnitude','power','phase_rad')]
    writer.writerow(columns+['valid_bin'])
    def number(v):
        return format(float(v),'.17g') if np.isfinite(v) else ''
    for i,wavelength in enumerate(a['wavelength_um']):
        row = [number(wavelength),number(a['frequency_per_um'][i])]
        for _,outgoing,incoming in ENTRIES:
            s = a['s'][i,outgoing,incoming]
            row.extend(number(v) for v in (s.real,s.imag,abs(s),abs(s)**2))
            row.append(number(np.angle(s)) if a['valid_bin'][i] and abs(s)>PHASE_FLOOR else '')
        writer.writerow(row+['true' if a['valid_bin'][i] else 'false'])
    return stream.getvalue()
