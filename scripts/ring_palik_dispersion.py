"""Narrow-band passive dielectric fit to the ring project's five stored Palik samples."""
import hashlib
import json
from pathlib import Path

import numpy as np

LIGHT = 299792458.
POLE_WAVELENGTH_UM = 0.50
OMEGA_0 = 2*np.pi*LIGHT/(POLE_WAVELENGTH_UM*1e-6)
WAVELENGTHS = np.array([1.500, 1.525, 1.550, 1.575, 1.600])
TRAIN = [0, 2, 4]
HOLDOUT = [1, 3]
MATERIAL_SOURCE_SHA256 = 'eabaa8f2e629c4f550d7d70a990b269e2c597aa78cceb3a3257e63dce1b36030'
MATERIAL_FILE = 'docs/verification/v013-ring-project-audit.json'


def material_samples():
    raw = (Path(__file__).resolve().parents[1]/MATERIAL_FILE).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MATERIAL_SOURCE_SHA256:
        raise ValueError('The ring material input differs from the pinned project audit.')
    data = json.loads(raw)
    np.testing.assert_array_equal(WAVELENGTHS, data['material_wavelengths_um'])
    samples = {}
    for key in ('Si (Silicon) - Palik', 'SiO2 (Glass) - Palik'):
        row = data['material_tables_epsilon'][key]
        if np.any(row['imaginary']):
            raise ValueError('This undamped model requires the filed lossless passband.')
        samples[key] = np.asarray(row['real'], dtype=np.float64)
    return samples


def recurrence(dt_s):
    k = (OMEGA_0*dt_s)**2
    if not 0 < k < 4:
        raise ValueError('The discrete Lorentz pole is outside its stable time step.')
    return 2-k, -1., k


def epsilon_at(c1, c2, c3, eps_inf, wavelength_um, dt_s):
    """Dielectric function of the *discrete* FDTDX polarization recurrence."""
    omega_dt = 2*np.pi*LIGHT*dt_s / (np.asarray(wavelength_um)*1e-6)
    # For an undamped pole, use the cancellation-resistant real denominator.
    if c2 != -1.:
        raise ValueError('This band-limited fit uses an undamped Lorentz pole.')
    denominator = (2-c1) - 4*np.sin(omega_dt/2)**2
    return eps_inf + c3/denominator


def fit_effective_cells(samples, dt_s):
    """Fit epsilon-infinity and recurrence strength to three positive dielectric samples."""
    values = np.asarray(samples, dtype=np.float64)
    if values.shape[0]!=len(TRAIN) or not np.isfinite(values).all():
        raise ValueError('Expected the three finite training permittivities.')
    c1,c2,k = recurrence(dt_s)
    basis = epsilon_at(c1,c2,k,0.,WAVELENGTHS[TRAIN],dt_s)
    inverse = np.linalg.pinv(np.column_stack((np.ones(len(TRAIN)),basis)))
    eps_inf, strength = np.tensordot(inverse, values, axes=(1,0))
    if np.min(strength) < -1e-5 or np.min(eps_inf) < 1-1e-5:
        raise ValueError('Effective material fit violates the passive dielectric bounds.')
    eps_inf, strength = np.maximum(eps_inf,1.), np.maximum(strength,0.)
    if np.any(k*(1+strength/eps_inf)>=4):
        raise ValueError('The fitted longitudinal oscillator exceeds the time-step limit.')
    return eps_inf, k*strength


def pole_config(dt_s):
    """Return Si/oxide high-frequency permittivities, shared c1/c2, and both c3 weights."""
    samples = material_samples()
    si,oxide = (fit_effective_cells(values[TRAIN],dt_s) for values in samples.values())
    c1,c2,_ = recurrence(dt_s)
    return float(si[0]),float(oxide[0]),c1,c2,float(si[1]),float(oxide[1])
