"""Compare a GPU-loaded ring with its filed Lumerical samples or the pinned Tidy3D plot."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

ANSYS_SHA = '497d0ed797b82d5887bd67ee1630712fb5ac38de0eee6a7b99d55e63fc938bd8'
TIDY_SHA = 'b1dac0b3a47518cc5461c5457511748f5435eaea53703d9fdff1236adbd24908'
PLOT_SHA = 'a9fc0b553c3e597bc4726063db06fb62d49523da06f0fe46b5c17d7ae1d56878'


def lumerical_spectrum(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ANSYS_SHA:
        raise ValueError('Unexpected Lumerical attachment revision.')
    # The S-parameter file gives scattering magnitude (not power) as its second column.
    lines=path.read_text().splitlines()
    def rows(receive,launch):
        row=next(i for i,s in enumerate(lines) if s.strip()==f'("port {receive}", "TE", 1, "port {launch}", 1, "transmission")')
        count=int(re.fullmatch(r'\((\d+), 3\)',lines[row+1]).group(1))
        f,amp,_=np.loadtxt(lines[row+2:row+2+count]).T
        return 299792458e6/f[::-1],amp[::-1]**2
    wave,drop=rows(2,1)
    _,through=rows(3,1)
    return wave,through,drop,{'kind':'FDTD text attachment','input_port':1,'drop_port':2,
        'through_port':3,'attachment_sha256':ANSYS_SHA,'reference_samples':len(wave)}


def tidy_spectrum(path):
    from PIL import Image
    if hashlib.sha256(path.read_bytes()).hexdigest()!=TIDY_SHA:
        raise ValueError('Unexpected pinned Tidy3D notebook revision.')
    cell=json.loads(path.read_text())['cells'][24]
    if '10 * np.log10(T_drop)' not in ''.join(cell['source']):
        raise ValueError('Expected transmission in dB on the published ordinate.')
    data=base64.b64decode(cell['outputs'][0]['data']['image/png'])
    if hashlib.sha256(data).hexdigest()!=PLOT_SHA:
        raise ValueError('Unexpected published full-band spectrum.')
    image=np.array(Image.open(io.BytesIO(data)).convert('RGB'),dtype=int)
    if image.shape!=(432,574,3):
        raise ValueError('Unexpected plot dimensions.')
    # Calibrated pixel centers on successive major ticks: x 91→542 spans 1.50→1.60 µm;
    # y 26→350 spans 0→−30 dB. Restrict colors to the actual plotted area above the legend.
    columns=np.arange(91,543)
    wave=1.5+(columns-91)*.1/(542-91)
    values=[]
    uncertainty=[]
    for color in ('red','blue'):
        channel=0 if color=='red' else 2
        signals=[]
        for x in columns:
            stripe=image[23:373,x,:]
            mask=(stripe[:,channel]>175)&(stripe[:,channel]-stripe[:,1]>95)&(stripe[:,channel]-stripe[:,2-channel]>95)
            rows=np.flatnonzero(mask)+23
            if 80<=x<=110:
                rows=rows[rows<323]
            if not len(rows):
                signals.append(np.nan)
                continue
            uncertainty.append((rows[-1]-rows[0])/2*30/324)
            db=(26-np.median(rows))*30/324
            signals.append(10**(db/10))
        values.append(np.array(signals))
    return wave,*values,{'kind':'published full-band figure digitization','notebook_sha256':TIDY_SHA,
        'plot_sha256':PLOT_SHA,'horizontal_pixel_nm':100/451,'vertical_pixel_db':30/324,
        'maximum_half_trace_range_db':float(np.max(uncertainty)),
        'scope':'A steep line can cover many y pixels within one column. Readout varies across the plotted linewidth.'}


def compare(report, path):
    vendor=report['vendor']
    wave,through,drop,source=(lumerical_spectrum if vendor=='lumerical' else tidy_spectrum)(path)
    row=report['series'][vendor]
    wl=np.array(row['wavelength_um'])
    bins=np.array(row['valid_bins']) & (wl>=max(1.5,wave.min())) & (wl<=min(1.6,wave.max()))
    result={'vendor':vendor,'resolution':report['physical_model']['resolution'],
            'duration_um_c':report['duration_um_c'],'source':source,'source_url':report['source_url'],
            'passed_numerical_spectrum':report['passed'],'wavelength_shift_applied_nm':0,
            'source_wavelength_band_um':[float(wave.min()),float(wave.max())],'channels':{},
            'measured_drop_fsr_nm':report['measured_drop_fsr_nm'],'published_drop_fsr_nm':report['published_drop_fsr_nm']}
    for label,native in [('through',through),('drop',drop)]:
        valid=np.isfinite(native)
        reference=np.interp(wl[ bins],wave[valid],native[valid])
        local=np.array(row[label])[bins]
        if not len(local):
            result['channels'][label]={'qualified_sample_count':0}
            continue
        peak_index=np.argmax(reference)
        result['channels'][label]=dict(qualified_sample_count=len(local),
            mean_absolute_power_difference=float(np.mean(abs(local-reference))),
            rms_power_difference=float(np.mean((local-reference)**2)**.5),
            at_vendor_peak_sample=dict(wavelength_um=float(wl[bins][peak_index]),local_power=float(local[peak_index]),
                                       published_power=float(reference[peak_index])),
            paired_spectrum=dict(wavelength_um=wl[bins].tolist(),local_power=local.tolist(),published_power=reference.tolist()))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='directory containing native report.json')
    parser.add_argument('--source',type=Path,required=True,help='S-parameter text or pinned notebook for the selected vendor')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    result=compare(report,args.source)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
