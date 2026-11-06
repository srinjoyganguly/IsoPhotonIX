"""Propagating plane-wave flux above a grating, through the recorded finite air aperture."""
import numpy as np


def angular_components(plane, spacing_um, wavelengths_um, upward=True):
    """Resolve tangential E/H into +/-z plane waves under exp(-i omega t).

    Each FFT bin represents a direction cosine. Flux weights include TE/TM
    admittances and the discrete Parseval area. Grazing bins beyond 80° are
    excluded to avoid the singularity of the finite-aperture decomposition.
    """
    plane = np.asarray(plane)
    wavelengths = np.asarray(wavelengths_um)
    if plane.ndim != 4 or plane.shape[:2] != (len(wavelengths), 6) or spacing_um <= 0:
        raise ValueError('Expected [wavelength, six field components, x, y] on a uniform air plane.')
    if not np.isfinite(plane).all() or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
        raise ValueError('Angular projection requires finite fields and positive wavelengths.')
    nx, ny = plane.shape[-2:]
    fx, fy = np.meshgrid(np.fft.fftfreq(nx, spacing_um), np.fft.fftfreq(ny, spacing_um), indexing='ij')
    fr = np.hypot(fx, fy)
    tx = np.divide(fx, fr, out=np.ones_like(fr), where=fr > 0)
    ty = np.divide(fy, fr, out=np.zeros_like(fr), where=fr > 0)
    ex,ey,hx,hy = np.fft.fft2(plane[:,[0,1,3,4]], axes=(-2,-1)).transpose(1,0,2,3)
    radial_e, s_e = tx*ex+ty*ey, -ty*ex+tx*ey
    radial_h, s_h = tx*hx+ty*hy, -ty*hx+tx*hy
    cosines = wavelengths[:,None,None]*fr
    propagating = cosines < np.sin(np.deg2rad(80.))
    cz = np.sqrt(np.where(propagating, 1-cosines**2, 1.))
    sign = 1 if upward else -1
    e_s = .5*(s_e-sign*radial_h/cz)
    e_t = .5*(radial_e+sign*s_h*cz)
    return e_s, e_t, cz, tx, ty, propagating, fx, fy


def air_angular_flux(plane, spacing_um, wavelengths_um, upward=True):
    """Integrate propagating TE/TM power through the recorded finite air aperture."""
    wavelengths = np.asarray(wavelengths_um)
    nx, ny = plane.shape[-2:]
    e_s,e_t,cz,tx,ty,propagating,fx,fy = angular_components(plane,spacing_um,wavelengths,upward)
    cosines = wavelengths[:,None,None]*np.hypot(fx,fy)
    power = np.where(propagating, .5*(cz*abs(e_s)**2+abs(e_t)**2/cz), 0.)*spacing_um**2/(nx*ny)
    theta = np.rad2deg(np.arcsin(np.clip(cosines,0,1)))
    bins = np.arange(0., 81., 10.)
    binned = np.stack([np.where((theta>=lo)&(theta<hi),power,0).sum(axis=(-2,-1))
                       for lo,hi in zip(bins[:-1],bins[1:])],axis=1)
    return {'power':power, 'polar_bin_power':binned, 'polar_edges_deg':bins,
            'spatial_frequency_x_per_um':fx[:,0], 'spatial_frequency_y_per_um':fy[0,:]}


def incident_beam_check(plane, spacing_um, wavelengths_um, incident_power):
    """Measure the incident beam direction and contamination from a finite air aperture."""
    down=air_angular_flux(plane,spacing_um,wavelengths_um,upward=False)
    up=air_angular_flux(plane,spacing_um,wavelengths_um)
    power=down['power']
    total=power.sum(axis=(-2,-1))
    fx=down['spatial_frequency_x_per_um']
    moment=(power*(np.asarray(wavelengths_um)[:,None,None]*fx[None,:,None])).sum(axis=(-2,-1))/total
    return {'mean_x_angle_deg':np.rad2deg(np.arcsin(np.clip(moment,-1.,1.))).tolist(),
            'downward_propagating_to_direct':(total/incident_power).tolist(),
            'upward_propagating_to_direct':(up['power'].sum(axis=(-2,-1))/incident_power).tolist(),
            'incident_to_band_max':(incident_power/np.max(incident_power)).tolist()}
