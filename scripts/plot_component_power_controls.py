"""Plot measured equal-mesh component controls against their own commercial references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(directory, filename):
    path=directory/filename
    report=json.loads(path.read_text())
    with (directory/report['arrays_file']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=report['arrays_sha256'] or not report['passed']:
            raise ValueError(f'Unverified numerical result: {filename}')
    return report


def curve(report):
    if report['case_id']=='loaded-ring-3d':
        row=report['vendor_comparison']['channels']['drop']['paired_spectrum']
        return [np.asarray(row[k]) for k in ('wavelength_um','local_power','published_power')]
    match=next(c for c in report['comparisons'] if c.get('geometry_matched'))
    return [np.asarray(report['result']['wavelength_um']),np.asarray(report['result']['beam_to_guide']),
            np.asarray(match.get('published_power',match.get('published_s21_power')))]


def plot(directory, output, dispersive=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    cases=[
        ('Lumerical ring · 16 cells/µm','Drop power (%)',
         [('v012-ring3d-lumerical-r16-t3200.json','Baseline'),
          ('v013-ring3d-lumerical-r16-t3200-exact-area.json','Exact area')]),
        ('Tidy3D ring · 16 cells/µm','Drop power (%)',
         [('v012-ring3d-tidy3d-r16-t4800.json','Baseline'),
          ('v013-ring3d-tidy3d-r16-t4800-exact-area.json','Exact area')]),
        ('Lumerical grating · 12 cells/µm','Fiber-to-guide coupling (%)',
         [('v013-grating3d-lumerical-dense-volume-average-r12.json','Baseline'),
          ('v013-grating3d-lumerical-dense-direct-yee-r12.json','Direct Yee')]),
        ('Tidy3D grating · uniform finite teeth','Beam-to-guide coupling (%)',
         [('v012-grating3d-tidy3d-r12.json','12 cells/µm'),('v012-grating3d-tidy3d-r14.json','14 cells/µm')])]
    if dispersive:
        cases[0][2].append((dispersive,'Exact area + dispersion'))
    palette=['#b15307','#00776f','#614797']
    for axis,(title,units,variants) in zip(axes.flat,cases):
        for i,(filename,label) in enumerate(variants):
            wave,local,published=curve(load(directory,filename))
            if i==0:
                axis.plot(wave*1000,published*100,'--',color='#344053',label='Published',linewidth=1.7)
            marker='o' if len(wave)<=25 else None
            axis.plot(wave*1000,local*100,color=palette[i],label=label,linewidth=1.55,marker=marker,markersize=3)
        axis.axvline(1550,color='#a3acb2',ls=':',lw=1)
        axis.set(xlabel='Wavelength (nm)',ylabel=units,title=title,ylim=(0,None))
        axis.set_xlim((1535,1570) if units=='Drop power (%)' else (1499,1601))
        axis.legend(frameon=False,fontsize=8,loc='upper left')
        axis.grid(axis='y',color='#dee1e3',lw=.6)
        axis.spines[['top','right']].set_visible(False)
    figure.savefig(output,dpi=160,facecolor='white')
    plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'docs/verification')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dispersive',help='Qualified 16-cell dispersive Lumerical ring receipt filename')
    args=parser.parse_args()
    plot(args.directory,args.output,args.dispersive)
