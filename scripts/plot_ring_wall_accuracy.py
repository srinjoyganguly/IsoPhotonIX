"""Plot public reference powers against the measured, qualified loaded-ring spectra."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.compare_loaded_ring3d import lumerical_port_spectra, tidy_spectrum
from scripts.compare_ring_interface_models import BASE, digest


def curve(directory,vendor,mesh,duration):
    name=f'v014-ring3d-{vendor}-r{mesh}-t{duration}-wall-tensor.json'
    record=json.loads((directory/name).read_text())
    path=directory/record['arrays_file']
    if digest(path)!=record['arrays_sha256']:
        raise ValueError('The public ring data have changed.')
    with np.load(path,allow_pickle=False) as data:
        return data['wavelength_um']*1000, np.where(data['valid_bins'][:,None],100*abs(data['s'])**2,np.nan)


def plot(directory,lumerical,tidy,output):
    l_w,l_values,_=lumerical_port_spectra(lumerical)
    t_w,_,t_drop,_=tidy_spectrum(tidy)
    fig,axes=plt.subplots(3,1,figsize=(9,8.8),layout='constrained')
    colors={16:'#A85700',20:'#146578',24:'#66438E'}
    titles=('Lumerical add/drop ring · drop port', 'Lumerical add/drop ring · reflected power near 1550 nm',
            'Tidy3D add/drop ring · drop port')
    for ax,title in zip(axes,titles):
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
        ax.set_ylabel('Power (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.19)
        ax.tick_params(labelsize=9)
    axes[0].plot(l_w*1000,100*l_values['drop'],color='#222222',lw=2.2,label='Filed Lumerical')
    axes[1].plot(l_w*1000,100*l_values['reflection'],color='#222222',lw=2.2,label='Filed Lumerical')
    axes[2].plot(t_w*1000,100*t_drop,color='#222222',lw=2.2,label='Digitized Tidy3D')
    for mesh in (16,20,24):
        from scripts.diagnose_ring_stability import rejection_for
        duration=6000 if mesh==16 else 3200
        if rejection_for(directory/f'v014-ring3d-lumerical-r{mesh}-t{duration}-wall-tensor.json'):
            continue
        w,power=curve(directory,'lumerical',mesh,duration)
        axes[0].plot(w,power[:,2],color=colors[mesh],lw=1.35,label=f'{mesh} cells/µm')
        axes[1].plot(w,np.where(power[:,0]>0,power[:,0],np.nan),color=colors[mesh],lw=1.45,label=f'{mesh} cells/µm')
    for mesh in (16,20):
        w,power=curve(directory,'tidy3d',mesh,4800)
        axes[2].plot(w,power[:,2],color=colors[mesh],lw=1.35,label=f'{mesh} cells/µm')
    for ax in (axes[0],axes[2]):
        ax.set_xlim(1500,1600);ax.set_ylim(0,105)
    axes[1].set_xlim(1547.5,1553.5);axes[1].set_yscale('log');axes[1].set_ylim(.000001,100)
    axes[1].axvline(1550,color='#777777',alpha=.45,lw=.8,ls=':')
    for ax in axes:
        ax.legend(loc='upper right',ncol=4,fontsize=8,framealpha=.91,handlelength=1.8)
        ax.set_xlabel('Wavelength (nm)')
    fig.savefig(output,dpi=170,metadata={'Title':'Finite-height ring mesh and vendor power comparison'})
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=BASE)
    parser.add_argument('--lumerical',type=Path,required=True)
    parser.add_argument('--tidy3d',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(args.directory,args.lumerical,args.tidy3d,args.output)
