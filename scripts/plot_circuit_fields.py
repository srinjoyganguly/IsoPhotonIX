"""Plot the measured XY fields from two exported circuit field studies on a shared scale."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from backend.library_layout import connection_status


def plane(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        config=json.loads(archive.read('reference/config.json'))
        if config['study_type']!='layout-3d' or not config['geometry'].get('connections'):
            raise ValueError('Choose a measured finite-layout study with a named connection.')
        with np.load(io.BytesIO(archive.read('reference/arrays.npz')),allow_pickle=False) as sample:
            z=int(np.argmin(abs(sample['z_um'])))
            return (config,sample['x_um'],sample['y_um'],float(sample['z_um'][z]),
                    sample['fields'][1,z],sample['epsilon'][z])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('planned',type=Path)
    parser.add_argument('aligned',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    first,second=plane(args.planned),plane(args.aligned)
    c1,x,y,z,field1,epsilon1=first
    c2,x2,y2,z2,field2,epsilon2=second
    np.testing.assert_array_equal(x,x2)
    np.testing.assert_array_equal(y,y2)
    if z!=z2 or c1['geometry']['connections']!=c2['geometry']['connections']:
        raise ValueError('Both studies must preserve the same named link and field plane.')
    if {k:v for k,v in c1.items() if k!='geometry'}!={k:v for k,v in c2.items() if k!='geometry'}:
        raise ValueError('The two studies have different simulation settings.')
    before,after=(connection_status(c['geometry']) for c in (c1,c2))
    if len(before)!=1 or before[0]['state']!='needs_route' or after[0]['state']!='aligned':
        raise ValueError('Choose studies containing a single planned and then aligned pair.')
    dx=x[1]-x[0];dy=y[1]-y[0]
    extent=[x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2]
    peak=max(float(abs(field1).max()),float(abs(field2).max()))
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,sharey=True,layout='constrained')
    gap=before[0]['distance_um']
    material=(epsilon1!=c1['cladding_index']**2)|(epsilon2!=c2['cladding_index']**2)
    iy,ix=np.nonzero(material)
    if not len(ix):
        raise ValueError('No material differs from cladding on the sampled plane.')
    for ax,field,epsilon,title in zip(axes,(field1,field2),(epsilon1,epsilon2),(f'Planned connection · {gap:.3g} µm gap','Aligned connection · touching ports')):
        im=ax.imshow(field,origin='lower',extent=extent,vmin=-peak,vmax=peak,cmap='RdBu_r',interpolation='nearest')
        ax.contour(x,y,epsilon,levels=[(epsilon.min()+epsilon.max())/2],colors=['#455767'],linewidths=.8)
        ax.set_title(title,loc='left',fontsize=12)
        ax.set_ylabel('y (µm)')
        ax.set_xlim(max(x[0],x[ix.min()]-.7),min(x[-1],x[ix.max()]+.7))
        ax.set_ylim(max(y[0],y[iy.min()]-.8),min(y[-1],y[iy.max()]+.8))
    axes[-1].set_xlabel('x (µm)')
    fig.suptitle(f'Ey on the same XY plane at z = {z:.3f} µm',fontsize=15)
    fig.colorbar(im,ax=axes,label='Ey (point-source units)',shrink=.8)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
