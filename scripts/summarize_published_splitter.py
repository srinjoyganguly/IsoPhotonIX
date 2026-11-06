"""Compare local, independently excited Y-junction studies to the published Tidy3D curve."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.compare_published_splitter import ANSYS, NOTEBOOK_SHA, TIDY, WAVELENGTHS

# Positive excess loss in dB. Digitized from the original vendor notebook's
# stored plot at the five Meep sample wavelengths, without interpolating its maximum.
VENDOR_LOSS = np.array([.2364864865,.1594594595,.1317567568,.1371621622,.1520270270])


def summarize(folders,output):
    archive={}
    rows=[]
    for folder in folders:
        native=json.loads((folder/'report.json').read_text())
        if not native['passed'] or native['vendor_notebook_sha256'] != NOTEBOOK_SHA:
            raise ValueError('Only settled native measurements for this published model can be compared.')
        samples=folder/'arrays.npz'
        with np.load(samples,allow_pickle=False) as fields:
            power=abs(fields['arms']/fields['incident'][0])**2
            np.testing.assert_allclose(fields['wavelength_um'],WAVELENGTHS,rtol=0,atol=1e-14)
            np.testing.assert_allclose(power,fields['power'],rtol=1e-13,atol=1e-15)
            for name in fields:
                archive[f"mesh{native['resolution']}_{name}"]=fields[name]
        loss=-10*np.log10(power.sum(axis=0))
        np.testing.assert_allclose(loss,native['total_excess_loss_db'],rtol=0,atol=1e-12)
        upper_loss=-10*np.log10(2*power[1])
        rows.append({'published_observable_loss_db':upper_loss.tolist(),
                     'published_observable_delta_db':(upper_loss-VENDOR_LOSS).tolist(),
                     'published_observable_mean_absolute_delta_db':float(abs(upper_loss-VENDOR_LOSS).mean()),
                     'resolution':native['resolution'],'native_arrays_sha256':hashlib.sha256(samples.read_bytes()).hexdigest(),
                     'loss_db':loss.tolist(),'vendor_loss_db':VENDOR_LOSS.tolist(),
                     'delta_db':(loss-VENDOR_LOSS).tolist(),'mean_absolute_delta_db':float(abs(loss-VENDOR_LOSS).mean()),
                     'measured_arm_powers':power.tolist(),'late_monitor_changes':native['late_complex_monitor_changes'],
                     'native_version':native['version'],'native_scope':native['scope']})
    rows.sort(key=lambda r:r['resolution'])
    if len({r['resolution'] for r in rows})!=len(rows):
        raise ValueError('Duplicate numerical mesh.')
    target=output.with_suffix('.npz')
    np.savez_compressed(target,**archive)
    receipt={'passed':True,'case':'Tidy3D optimized Y reconstruction','wavelength_um':WAVELENGTHS.tolist(),
             'source_url':TIDY,'source_notebook':'https://github.com/flexcompute/tidy3d-notebooks/blob/develop/YJunction.ipynb',
             'source_notebook_sha256':NOTEBOOK_SHA,'curve_readout_uncertainty_db':.004,
             'curve_axes_pixels_xyxy':[73,15,569,385],
             'observable':'The published curve doubles its upper arm, so compare its positive excess loss against -10 log10(2 P_upper). loss_db independently uses both measured arm powers, -10 log10(P_lower+P_upper). All powers use the measured straight reference.',
             'geometry':'Reconstruction of the sinusoidal centreline in the example text. The archived runnable notebook calls PhotonForge Path.s_bend with its default zero Euler fraction. Its plotted contours differ from the text-equation path by about 0.05 µm in the bends.',
             'path_api':'https://docs.flexcompute.com/projects/photonforge/en/v1.0.1/_autosummary/photonforge.Path.html',
             'rows':rows,'arrays_file':target.name,'arrays_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'lumerical':{'source_url':ANSYS,'scope':'The vendor Y-branch tutorial describes near-equal division for its own optimized device. Its loss curve is outside the matched Tidy3D geometry comparison.'},
             'limits':'Digitization uncertainty, the reconstructed path, frozen central material indices and discretization are independent uncertainties.'}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    summarize(args.study,args.output)
