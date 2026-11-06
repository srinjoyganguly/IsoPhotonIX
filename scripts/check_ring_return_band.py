"""Resolve the returned/add wave line shapes of a measured ring without adjusting its scattering column."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from scripts.calibrate_loaded_ring3d import PULSE_END
from scripts.check_ring_wall_windows import stage_arrays
from scripts.compare_ring_interface_models import digest


def spectrum(stage,wl,fraction):
    t=stage['time_um_c']
    end=int(len(t)*fraction)
    traces=stage['projections'][:end]
    basis=stage.get('projection_wavelengths_um',np.array([1.49,1.52,1.55,1.58,1.61]))
    left=(np.searchsorted(basis,wl,side='right')-1).clip(0,len(basis)-2)
    alpha=(wl-basis[left])/(basis[left+1]-basis[left])
    # Block the frequency transform to bound memory while preserving every time sample.
    values=[]
    for first in range(0,len(wl),16):
        select=slice(first,first+16)
        fourier=np.exp(2j*np.pi*t[:end][None,:]/wl[select,None])
        raw=(fourier @ traces.reshape(end,-1)).reshape(-1,*traces.shape[1:])*(t[1]-t[0])
        rows=np.arange(len(raw))
        weight=alpha[select,None,None]
        values.append((1-weight)*raw[rows,:,left[select],:]+weight*raw[rows,:,left[select]+1,:])
    return np.concatenate(values)


def peak_summary(wl,power,duration,wavelength=1.55):
    center=np.argmax(power)
    level=power[center]/2
    before=np.flatnonzero(power[:center]<level)
    after=np.flatnonzero(power[center+1:]<level)
    width=None
    if len(before) and len(after):
        i=before[-1];j=after[0]+center+1
        a=np.interp(level,power[i:i+2],wl[i:i+2])
        b=np.interp(level,power[j-1:j+1][::-1],wl[j-1:j+1][::-1])
        width=float(1000*(b-a))
    peaks=find_peaks(power,prominence=.01)[0]
    resolved_peak=bool(np.any(peaks==center))
    return dict(strongest_peak_nm=float(1000*wl[center]) if resolved_peak else None,
                largest_sample_nm=float(1000*wl[center]),maximum_at_scan_boundary=bool(center in (0,len(wl)-1)),
                peak_power=float(power[center]),fwhm_nm=width if resolved_peak else None,
                peaks_with_one_percentage_point_prominence_nm=(wl[peaks]*1000).tolist(),
                rayleigh_spacing_estimate_nm=float(1000*wavelength**2/(duration-PULSE_END)))


def measure(path,reference_path=None):
    from scripts.diagnose_ring_stability import rejection_for
    rejected=rejection_for(path)
    if rejected:
        return dict(source_report=path.name,arrays_sha256=rejected['source_arrays_sha256'],
                    usable_spectrum=False,reason=rejected['reason'],windows=[],published=None)
    report=json.loads(path.read_text()); raw=path.parent/report['arrays_file']
    if digest(raw)!=report['arrays_sha256']:
        raise ValueError('Unexpected field archive.')
    with np.load(raw,allow_pickle=False) as arrays:
        ref=stage_arrays(arrays,'reference');dev=stage_arrays(arrays,'device')
        original_wl=arrays['wavelength_um']
        original=arrays['s']
    wl=np.arange(154750,155351,5)*1e-5  # µm; 0.05 nm diagnostic spacing
    ref_signal=spectrum(ref,wl,1.)
    inc=ref_signal[:,0,0]
    background=ref_signal[:,0,1]/inc
    histories=[]
    for fraction in (.8,1.):
        signal=spectrum(dev,wl,fraction)
        s=np.stack((signal[:,0,1]-ref_signal[:,0,1],signal[:,1,0],signal[:,2,1],signal[:,3,0]),axis=1)/inc[:,None]
        if fraction==1:
            candidates=np.rint((original_wl-wl[0])/(wl[1]-wl[0])).astype(int)
            common=(candidates>=0)&(candidates<len(wl))
            np.testing.assert_allclose(wl[candidates[common]],original_wl[common],atol=1e-12,rtol=0)
            np.testing.assert_allclose(s[candidates[common]],original[common],atol=5e-10,rtol=1e-8)
        powers=abs(s)**2
        center=int(np.argmin(abs(wl-1.55)))
        significant=(powers[:,0]>=.01)&(powers[:,3]>=.01)
        relative=np.angle(s[significant,3]*s[significant,0].conj())
        differential=np.angle(np.exp(1j*(relative-relative[np.argmin(abs(wl[significant]-1.55))]))) if len(relative) else relative
        end=float(dev['time_um_c'][int(len(dev['time_um_c'])*fraction)-1])
        hist=dict(window_fraction=fraction,observed_um_c=end,lines={key:peak_summary(wl,powers[:,i],end)
                    for key,i in (('reflection',0),('drop',2),('add_side',3))},
                at_1550=dict(channel_powers=powers[center].tolist(),normalized_complex_s_pairs=[[float(z.real),float(z.imag)] for z in s[center]],
                    straight_background_abs_s=float(abs(background[center])),reflection_add_combined_power=float(powers[center,0]+powers[center,3])),
                both_returns_above_one_percent_points=int(significant.sum()),
                maximum_return_power_difference=float(np.max(abs(powers[:,0]-powers[:,3]))),
                maximum_add_to_reflection_phase_change_relative_to_1550_rad=float(np.max(abs(differential))) if len(relative) else None,
                wavelength_um=wl.tolist(),power={key:powers[:,i].tolist() for i,key in enumerate(('reflection','through','drop','add_side'))})
        histories.append(hist)
    published=None
    if reference_path is not None:
        from scripts.compare_loaded_ring3d import lumerical_port_spectra
        w,ports,source=lumerical_port_spectra(reference_path)
        band=(w>=wl[0]) & (w<=wl[-1])
        published={'attachment_sha256':source['attachment_sha256'],'native_samples':int(band.sum()),
                   'maximum_sample_interval_nm':float(np.max(np.diff(w[band]))*1000),'channels':{}}
        for channel,values in ports.items():
            peak=np.argmax(values[band])
            local_peak=histories[-1]['lines']['reflection']['strongest_peak_nm']
            published['channels'][channel]=dict(largest_native_sample_nm=float(w[band][peak]*1000),
                largest_native_sample_power=float(values[band][peak]),
                interpolated_power_at_local_return_peak=float(np.interp(local_peak/1000,w,values)) if local_peak else None)
    return dict(source_report=path.name,arrays_sha256=report['arrays_sha256'],grid_step_nm=.05,
        published=published,
        scope='Direct complex Fourier transform of the measured time traces, checked against every stored 0.5 nm sample in the band. Finer sampling gives a denser evaluation of the same finite-time signal. Distinct physical lines still require separation, sufficient observation time and mesh stability.',
        ports='Local positive-Ey mode bases and actual native monitoring planes. The phase variation compares the two measured returned channels; it does not establish eigenmode parity or direction.',
        windows=histories,observable_limits='One excitation; reciprocity relates reverse excitations. Return/add equality cannot be inferred from reciprocity. Three staggered Hz point histories plus instantaneous real E snapshots do not give a complex counter-propagating modal expansion around the ring.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--lumerical-reference',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(args.report,args.lumerical_reference),indent=2,allow_nan=False)+'\n')
