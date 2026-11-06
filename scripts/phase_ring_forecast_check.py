"""Score the held-out device against the frozen straight-only ring forecast."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import file_sha, measured_beta, native_data

FORECAST_SHA256 = '4fd370d32314c05ffa173c29e8cfb5624d195f86d778a3effafd7adc59bf7f45'


def assign_lines(predictions, observed_nm):
    targets = np.asarray([row['prediction_r20_nm'] for row in predictions])
    half_spacing = float(np.min(np.diff(targets))/2)
    if half_spacing <= 0:
        raise ValueError('Frozen line targets must increase.')
    rows = []
    for target in predictions:
        prediction = target['prediction_r20_nm']
        lo, hi = target['sampling_and_prefix_interval_nm']
        candidates = [float(value) for value in observed_nm if abs(value-prediction)<half_spacing]
        row = dict(target, peak_candidates_r20_nm=candidates, observed_r20_nm=None,
                   observed_minus_prediction_nm=None, observed_r20_minus_r24_nm=None, status='unassigned')
        if len(candidates)==1:
            observed = candidates[0]
            row.update(observed_r20_nm=observed, observed_minus_prediction_nm=observed-prediction,
                       observed_r20_minus_r24_nm=observed-target['anchor_r24_nm'],
                       status='inside interval' if lo<=observed<=hi else 'outside interval')
        rows.append(row)
    return half_spacing, rows


def evaluate(forecast_path, device20):
    from scripts.calibrate_loaded_ring3d import directional_spectrum
    if file_sha(forecast_path)!=FORECAST_SHA256:
        raise ValueError('The forecast differs from the public preregistration.')
    forecast = json.loads(forecast_path.read_text())
    report, archive = native_data(device20, allow_partial_ring=True)
    with archive:
        model = report['physical_model']
        if (model['resolution']!=20 or model['vendor']!='lumerical'
                or model['ring_interface']!='reciprocal-wall'):
            raise ValueError('Only the preregistered 20-cell reciprocal-wall geometry can score the forecast.')
        waves = archive['wavelength_um']
        np.testing.assert_array_equal(waves,forecast['spectral_grid_um'])
        source = next(s for s in forecast['straight_reference_inputs'] if s['resolution']==20)
        length = (model['ports'][1][0]-model['ports'][0][0])/20
        np.testing.assert_allclose(length,source['port_separation_um'],rtol=0,atol=1e-14)
        n = np.interp(waves,archive['basis_wavelength_um'],archive['reference_mode_neff'][0])
        ref = {'projections':archive['reference_projections'],'time_um_c':archive['reference_time_um_c'],
               'projection_wavelengths_um':archive['basis_wavelength_um']}
        beta_errors = {}
        for fraction, field in ((.8,'beta_early_rad_per_um'),(1.,'beta_full_rad_per_um')):
            modal = directional_spectrum(ref,fraction)
            beta = measured_beta(modal[:,1,0]/modal[:,0,0],waves,n,length)
            expected = np.asarray(source[field])
            np.testing.assert_allclose(beta,expected,rtol=0,atol=1e-12)
            beta_errors[field] = float(np.max(abs(beta-expected)))
        # qualified_peaks is the original drop-peak detector restricted to
        # the native wavelength validity mask; no fitting or recentering.
        waves_nm = [p['wavelength_um']*1000 for p in report['qualified_peaks']]
        half_spacing, rows = assign_lines(forecast['predictions'],waves_nm)
    return {'forecast_file':forecast_path.name,'forecast_sha256':FORECAST_SHA256,
            'device20_file':device20.name,'device20_report_sha256':file_sha(device20),
            'device20_arrays_sha256':report['arrays_sha256'],'native_run_passed':report['passed'],
            'native_valid_bins':report['valid_bins'],'native_bins':report['total_bins'],
            'straight_beta_max_discrepancy_rad_per_um':beta_errors,'assignment_half_width_nm':half_spacing,
            'measured_qualified_drop_peaks_nm':waves_nm,'lines':rows,
            'scope':'Only the predeclared propagation-only hypothesis is scored. A miss shows straight propagation alone does not predict that line at this precision. An inside-interval result is compatible with this mechanism, without identifying angular order or establishing commercial agreement.'}


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=result['lines']
    fig,ax=plt.subplots(figsize=(7.7,3.1))
    for i,line in enumerate(lines):
        anchor=line['anchor_r24_nm']
        predicted=line['predicted_r20_minus_r24_nm']
        lo,hi=line['sampling_and_prefix_interval_nm']
        observed=line['observed_r20_minus_r24_nm']
        ax.errorbar(predicted,i,xerr=[[predicted-(lo-anchor)],[(hi-anchor)-predicted]],fmt='s',capsize=6,
                    markersize=6,color='#116d8b',label='Straight-guide prediction and sampling range' if i==0 else None)
        if observed is not None:
            ax.plot(observed,i,'o',markersize=8,color='#ba501b',label='Measured ring shift' if i==0 else None)
            ax.annotate(f'{observed:+.1f}',(observed,i),xytext=(-9,9),textcoords='offset points',
                        ha='center',color='#87350d',fontsize=9)
    ax.set(yticks=range(len(lines)),yticklabels=[f"{l['anchor_r24_nm']:g} nm" for l in lines],
           xlabel='Wavelength shift from 24 to 20 cells/µm (nm)',ylabel='24-cell ring peak',
           ylim=(len(lines)-.5,-.5),xlim=(-.25,6.65))
    ax.set_title('Straight-guide phase overpredicts the ring shift',loc='left',fontweight='bold',pad=12)
    ax.grid(axis='x',alpha=.2)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.54),ncol=2,frameon=False,fontsize=9)
    for edge in ('top','right','left'):
        ax.spines[edge].set_visible(False)
    fig.subplots_adjust(left=.18,right=.98,bottom=.3,top=.82)
    fig.savefig(path,dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('forecast','device20','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--figure',type=Path)
    args=p.parse_args()
    result=evaluate(args.forecast,args.device20)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.figure:
        plot(result,args.figure)
