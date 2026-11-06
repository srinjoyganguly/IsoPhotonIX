"""Fit loaded poles to native early/late power spectra and independent ring field decay traces.

The equal-bus rate decomposition is conditional on a single travelling mode,
equal couplings, and negligible direct background at the drop port. Radiation,
backscatter and numerical leakage contribute to the apparent internal loss.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def fit_symmetric_peak(wave, power, center, spacing):
    """Fit constrained temporal coupled-mode powers in physical frequency detuning."""
    take=abs(wave-center)<.28*spacing
    w=wave[take]
    if len(w)<12:
        return None
    through=power[take,1]
    drop=power[take,2]
    def residual(parameters):
        pole, width, internal=parameters
        detuning=2*pole/width*(pole/w-1)
        shape=1/(1+detuning**2)
        return np.r_[1-(1-internal**2)*shape-through, (1-internal)**2*shape-drop]
    f=least_squares(residual,(center,.002,.1),bounds=([center-.002,.00005,0],[center+.002,.025,.97]))
    pole,width,internal=map(float,f.x)
    Q=pole/width
    return dict(wavelength_um=pole,linewidth_nm=1000*width,loaded_q=Q,
                internal_fraction=internal,external_fraction_per_bus=(1-internal)/2,
                rms_power_residual=float(np.sqrt(np.mean(f.fun**2))),
                maximum_power_residual=float(np.max(abs(f.fun))),
                rms_reflected_add_power=float(np.sqrt(np.mean((power[take,0]+power[take,3])**2))),
                samples=int(take.sum()))


def ring_decay(time, probes, center, fsr):
    """Demodulate multiple ring probes and fit two disjoint intervals of free decay."""
    f=1/center
    # One standard deviation longer than the inverse adjacent frequency spacing
    # strongly rejects other longitudinal modes, while retaining a freely decaying pole.
    sigma=center**2/fsr
    first=65+4.5*sigma
    last=time[-1]-4.5*sigma
    if last-first<2*sigma:
        return {'qualified':False,'reason':'Too little post-source time for two independent windows.'}
    centers=np.linspace(first,last,36)
    carrier=np.exp(2j*np.pi*f*time)
    signal=(np.exp(-.5*((time[None,:]-centers[:,None])/sigma)**2) * carrier[None,:]) @ probes
    fits=[]
    for probe in range(probes.shape[1]):
        signal_peak=float(np.max(abs(signal[:,probe])))
        if signal_peak < max(float(np.max(abs(signal)))*.01,1e-20):
            continue
        for indices in (np.arange(18),np.arange(18,36)):
            amp=abs(signal[indices,probe])
            if amp.min()<1e-10*signal_peak:
                continue
            t=centers[indices]
            slope, intercept=np.polyfit(t,np.log(amp),1)
            phase_slope,phase_offset=np.polyfit(t,np.unwrap(np.angle(signal[indices,probe])),1)
            freq=f-phase_slope/(2*np.pi)
            if slope>=0 or freq<=0:
                continue
            fits.append(dict(probe=probe,start_um_c=float(t[0]),stop_um_c=float(t[-1]),
                quality=float(-np.pi*freq/slope),wavelength_um=float(1/freq),
                log_envelope_rms=float(np.sqrt(np.mean((np.log(amp)-slope*t-intercept)**2))),
                phase_rms_rad=float(np.sqrt(np.mean((np.unwrap(np.angle(signal[indices,probe]))-phase_slope*t-phase_offset)**2)))))
    if len(fits)<4:
        return dict(qualified=False,reason='Fewer than four decaying probe-window fits.',fits=fits)
    values=np.array([r['quality'] for r in fits])
    q=float(np.median(values))
    span=float(np.ptp(values)/q)
    residual=max(row['log_envelope_rms'] for row in fits)
    return dict(qualified=bool(span<=.12 and residual<=.06 and
                all(abs(row['wavelength_um']-center)<.0015 and row['phase_rms_rad']<.1 for row in fits)),
                demodulation_sigma_um_c=sigma,loaded_q=q,relative_q_range=span,
                maximum_log_envelope_residual=residual,fits=fits)


def native_port_spectra(time, signals, basis, waves, fraction):
    """Evaluate only the bins near a pole directly from its native sampled modal traces."""
    count=int(len(time)*fraction)
    t=time[:count]
    dt=t[1]-t[0]
    groups=np.searchsorted(basis,waves,side='right').clip(1,len(basis)-1)-1
    result=np.empty((len(waves),4,2),complex)
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group)
        for start in range(0,len(ids),24):
            selected=ids[start:start+24]
            wavelengths=waves[selected]
            dft=np.exp(2j*np.pi*t[None,:]/wavelengths[:,None])*dt
            span=(signals[:count,:,group:group+2,:]).reshape(count,-1)
            projections=(dft @ span).reshape(len(selected),4,2,2)
            weight=((wavelengths-basis[group])/(basis[group+1]-basis[group]))[:,None,None]
            result[selected]=(1-weight)*projections[:,:,0,:]+weight*projections[:,:,1,:]
    return result


def fine_powers(device, reference, basis, waves):
    r=native_port_spectra(reference['time'],reference['signals'],basis,waves,1.)
    incident=r[:,0,0]
    values=[]
    for fraction in (1.,.8):
        f=native_port_spectra(device['time'],device['signals'],basis,waves,fraction)
        out=np.stack([f[:,0,1]-r[:,0,1],f[:,1,0],f[:,2,1],f[:,3,0]],axis=1)/incident[:,None]
        power=abs(out)**2
        power[:,1]/=abs(r[:,1,0]/incident)**2
        values.append(power)
    return values


def measure(folder, report=None):
    if report is None:
        report=json.loads((folder/'report.json').read_text())
    with np.load(folder/'arrays.npz',allow_pickle=False) as a:
        wave=a['wavelength_um']
        basis=a['basis_wavelength_um']
        native={k:{'time':a[k+'_time_um_c'],'signals':a[k+'_projections']} for k in ('reference','device')}
        t,probes=a['device_time_um_c'],a['device_ring_Hz']
        valid=a['valid_bins']
    candidate=[p['wavelength_um'] for p in report['peaks']]
    peaks=[]
    for i,center in enumerate(candidate):
        adjacent=[abs(center-peak) for j,peak in enumerate(candidate) if j!=i]
        if not adjacent:
            continue
        spacing=min(adjacent)
        # The 0.5 nm overview samples cannot reliably resolve a narrow linewidth.
        # Compute 0.1 nm samples from the actual two-stage signals within this resonance.
        dense_step=.0001
        half=int(.28*spacing/dense_step)
        fine=center+np.arange(-half,half+1)*dense_step
        power,early=fine_powers(native['device'],native['reference'],basis,fine)
        fitted=fit_symmetric_peak(fine,power,center,spacing)
        if fitted is None:
            continue
        cut=fit_symmetric_peak(fine,early,center,spacing)
        decay=ring_decay(t,probes,center,spacing)
        all_bins=bool(valid[abs(wave-center)<.28*spacing].all())
        width_change=abs(fitted['linewidth_nm']-cut['linewidth_nm'])/fitted['linewidth_nm']
        duration=t[-1]-57.15
        temporal_nm=1000*center**2/duration
        width_resolved=fitted['linewidth_nm']>=max(4*1000*dense_step,2*temporal_nm)
        qualified=bool(all_bins and width_change<=.1 and fitted['rms_power_residual']<=.05 and
                     fitted['maximum_power_residual']<=.13 and fitted['rms_reflected_add_power']<=.04 and
                     width_resolved and decay['qualified'] and abs(decay['loaded_q']/fitted['loaded_q']-1)<=.12)
        internal=fitted['internal_fraction']; q=fitted['loaded_q']
        # Bound the weak internal loss conservatively using spectral residual and cut variation.
        uncertainty=max(abs(internal-cut['internal_fraction']),fitted['maximum_power_residual'],.01)
        split=dict(loaded_q=q,absolute_internal_fraction_uncertainty_proxy=uncertainty,
                   apparent_external_q_each_bus=2*q/(1-internal),
                   apparent_internal_q=q/internal if internal>2*uncertainty else None,
                   apparent_internal_q_lower_bound=q/min(internal+uncertainty,1.))
        peaks.append(dict(sampled_drop_peak_um=center,line_fit_spacing_nm=dense_step*1000,qualified=qualified,all_nearby_bins_valid=all_bins,
                    linewidth_resolved=bool(width_resolved),finite_window_resolution_nm=temporal_nm,
                    relative_width_change_last_20_percent=width_change,full_fit=fitted,prefix_fit=cut,
                    field_decay=decay,qualified_equal_bus_loss=split if qualified else None))
    return dict(vendor=report['vendor'],resolution=report['physical_model']['resolution'],duration_um_c=report['duration_um_c'],
            source_arrays_sha256=report['arrays_sha256'],peaks=peaks,qualified_peaks=sum(p['qualified'] for p in peaks),
            model='Symmetric single travelling resonance. Two through/drop bus rates are assumed equal, and all remaining linewidth is apparent internal loss. Independent decaying field probes and the earlier 80% of the record gate the estimate. Absolute internal Q is withheld for a loss fraction comparable to the power-fit residual.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=measure(args.run)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
