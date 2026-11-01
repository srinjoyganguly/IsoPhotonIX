"""Read and audit signed port spectra; invalid bins never become plotted zeros."""
import csv
from functools import lru_cache
import io
import math
from pathlib import Path

import numpy as np

RAW = ('wavelength_um','frequency_per_um','incident_flux','reflected_flux','transmitted_flux','reference_transmitted_flux')
RATIOS = ('reflectance','transmittance','uncollected_fraction')


def validate_spectrum(arrays, metadata=None):
    values={key:np.asarray(arrays[key]) for key in (*RAW,*RATIOS,'valid_bin')}
    n=len(values['wavelength_um'])
    if not 3 <= n <= 61 or any(v.ndim != 1 or len(v)!=n for v in values.values()):
        raise ValueError('Spectral arrays must have matching one-dimensional frequency bins.')
    if values['valid_bin'].dtype.kind != 'b':
        raise ValueError('Spectrum bin validity must be explicitly boolean.')
    if any(values[key].dtype.kind not in 'fiu' for key in (*RAW,*RATIOS)):
        raise ValueError('Flux and ratio arrays must contain real numeric values.')
    if any(not np.isfinite(values[key]).all() for key in RAW):
        raise ValueError('Raw spectral coordinates and power flux must be finite.')
    wavelength=values['wavelength_um']; frequency=values['frequency_per_um']
    if np.any(wavelength<=0) or np.any(np.diff(wavelength)<=0) or not np.allclose(frequency,1/wavelength,rtol=1e-6,atol=0):
        raise ValueError('Positive ascending wavelengths must match their reciprocal frequencies.')
    incident=values['incident_flux']; valid=values['valid_bin']
    if not np.any(incident>0) or np.any(valid & (incident<=np.max(incident)*1e-6)):
        raise ValueError('Qualified bins need non-negligible positive measured incident power.')
    for key in RATIOS:
        if np.isinf(values[key]).any() or not np.isfinite(values[key][valid]).all():
            raise ValueError('Qualified power ratios must be finite; undefined bins must be marked invalid.')
    positive=incident>np.max(incident)*1e-6
    for key,expected in (
        ('reflectance',np.divide(-values['reflected_flux'],incident,out=np.full(n,np.nan),where=positive)),
        ('transmittance',np.divide(values['transmitted_flux'],incident,out=np.full(n,np.nan),where=positive))):
        measured=positive & np.isfinite(values[key])
        if not np.allclose(values[key][measured],expected[measured],rtol=2e-5,atol=1e-7):
            raise ValueError(f'{key} does not match the recorded signed power normalization.')
    if not np.allclose(values['uncollected_fraction'],1-values['reflectance']-values['transmittance'],rtol=2e-5,atol=1e-7,equal_nan=True):
        raise ValueError('Uncollected flux must retain the unclipped 1-R-T residual.')
    if metadata is not None:
        config=metadata.get('config',{})
        if config.get('study_type') != 'waveguide-ports-2d' or config.get('frequency_points') != n:
            raise ValueError('Spectrum bin count must match its saved port-study configuration.')
        expected_frequencies=np.linspace(1/config['wavelength_max_um'],1/config['wavelength_min_um'],n)[::-1]
        if not np.allclose(frequency,expected_frequencies,rtol=1e-6,atol=0):
            raise ValueError('Spectrum frequencies must match the configured wavelength band.')
        settlement=metadata.get('settlement',{})
        if not all(isinstance(settlement.get(stage),dict) and type(settlement[stage].get('settled')) is bool for stage in ('reference','device')):
            raise ValueError('Reference and device settling records are required.')
        diagnostics = (('energy_ratio','decay_by','energy_decay'),
                       ('spectral_relative_change','spectrum_tolerance','spectral_relative_change'))
        for stage_name in ('reference','device'):
            stage = settlement[stage_name]
            thresholds = stage.get('thresholds',{})
            if not isinstance(thresholds,dict):
                raise ValueError(f'{stage_name} settling thresholds must be a recorded object.')
            for metric,setting,threshold_key in diagnostics:
                requested = config.get(setting)
                if type(requested) not in (int,float) or not math.isfinite(requested) or requested <= 0:
                    raise ValueError(f'Saved configuration must contain a finite positive {setting}.')
                recorded = ([stage[setting]] if setting in stage else []) + ([thresholds[threshold_key]] if threshold_key in thresholds else [])
                if any(type(value) not in (int,float) or not math.isfinite(value) or not math.isclose(value,requested,rel_tol=1e-12,abs_tol=0.) for value in recorded):
                    raise ValueError(f'{stage_name} recorded {setting} differs from its requested configuration.')
                value = stage.get(metric)
                if stage['settled'] and (type(value) not in (int,float) or not math.isfinite(value) or value < 0 or value > requested):
                    raise ValueError(f'{stage_name} claims settled but its {metric} does not satisfy the requested {setting}.')
        if np.any(valid) and not all(settlement[stage]['settled'] for stage in ('reference','device')):
            raise ValueError('Unsettled reference or device results cannot mark bins qualified.')
        if metadata.get('spectra_valid') is not bool(np.all(valid)):
            raise ValueError('Recorded spectrum qualification disagrees with its bin flags.')
    for value in values.values(): value.setflags(write=False)
    return values


@lru_cache(maxsize=2)
def _read(filename,mtime,size):
    del mtime,size
    with np.load(filename,allow_pickle=False) as arrays:
        return validate_spectrum(arrays)


def stored_spectrum(output,metadata=None):
    path=Path(output)/'spectra.npz'; stat=path.stat()
    return validate_spectrum(_read(str(path.resolve()),stat.st_mtime_ns,stat.st_size),metadata)


def spectrum_response(values,metadata):
    valid=values['valid_bin']
    result={key:values[key].tolist() for key in (*RAW,'valid_bin')}
    for key in RATIOS:
        result[key]=[float(value) if okay else None for value,okay in zip(values[key],valid)]
    result.update(metadata=metadata,bin_reasons=[None if okay else 'Weak incident reference or unsettled simulation; inspect the recorded diagnostics.' for okay in valid],
                  ratio_definition='R=-signed reflected flux/incident; T=transmitted flux/incident; residual=1-R-T. Invalid ratios are null, not zero.')
    return result


def spectrum_csv(values):
    """Retain actual ratios even when unqualified; CSV includes mandatory flags."""
    stream=io.StringIO(newline=''); writer=csv.writer(stream)
    columns=(*RAW,*RATIOS,'valid_bin')
    writer.writerow(columns)
    for i in range(len(values['valid_bin'])):
        writer.writerow([('true' if values[key][i] else 'false') if key=='valid_bin' else
                         (format(float(values[key][i]),'.17g') if np.isfinite(values[key][i]) else '') for key in columns])
    return stream.getvalue()
