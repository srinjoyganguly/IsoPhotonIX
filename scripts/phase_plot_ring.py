"""Plot physically registered ring phase and reflection from published numerical receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.phase_component_comparison import optical_sfile


def plot(receipt,vendor_file,output):
    import matplotlib.pyplot as plt
    cases=json.loads(receipt.read_text())['native_samples']
    fig,axes=plt.subplots(3,1,figsize=(8.5,7.8),sharex=True,layout='constrained')
    colors=plt.cm.viridis(np.linspace(.15,.8,len(cases)))
    for row,color in zip(cases,colors):
        w=np.asarray(row['wavelength_um'])
        valid=np.asarray(row['valid_bins'])&(w>=1.5)&(w<=1.6)
        fixed=row['single_central_basis_band']
        phase=np.array(fixed['signed_native_minus_vendor_phase_rad'],dtype=float)
        label=f"{row['resolution']} cells/µm"
        for ax,j in zip(axes[:2],(1,2)):
            values=phase[:,j].copy()
            # Avoid drawing a spurious vertical line across a wrapped ±π branch.
            values[1:][abs(np.diff(values))>np.pi]=np.nan
            ax.plot(w*1000,np.where(valid,values,np.nan),label=label,color=color,linewidth=1.3)
        s=np.array(fixed['registered_native_real'])+1j*np.array(fixed['registered_native_imag'])
        power=abs(s[:,0])**2
        axes[2].plot(w*1000,np.where(valid,power,np.nan),label=label,color=color,linewidth=1.3)
    vendor,_,support=optical_sfile(vendor_file,'ring',w)
    axes[2].plot(1000*w,np.where(support,abs(vendor[:,0])**2,np.nan),label='Lumerical',color='#222222',linestyle='--',linewidth=1.2)
    for ax,title in zip(axes,['Through phase residual','Drop phase residual','Input reflection power']):
        ax.set_title(title,loc='left',fontsize=12,weight='bold')
        ax.grid(alpha=.2)
        ax.set_xlim(1500,1600)
    for ax in axes[:2]:
        ax.axhline(0,color='0.5',linewidth=.7)
        ax.set_ylabel('Native − Lumerical (rad)')
        ax.set_ylim(-np.pi-.1,np.pi+.1)
    axes[2].set_yscale('log');axes[2].set_ylabel('Power fraction');axes[2].set_xlabel('Vacuum wavelength (nm)')
    axes[2].legend(frameon=False,loc='upper right',ncol=min(4,len(cases)+1))
    fig.suptitle('Bus-loaded ring at the filed modal port planes',fontsize=15,weight='bold')
    fig.savefig(output,dpi=145)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path)
    parser.add_argument('--vendor-sfile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.receipt,args.vendor_sfile,args.output)
