"""Check both signs of the port amplitude against Ey from actual saved Meep reference fields."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from calibrate_multiport import build_simulation, project


def audit(directory):
    import meep as mp
    mp.verbosity(0)
    folder=Path(directory)
    config=json.loads((folder/'config.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    if report['solver']!='Meep' or report['version']!=mp.__version__:
        raise ValueError('Use the Meep release that produced this result.')
    model=config['model']
    rows=[]
    for sign in (1,-1):
        index=next(i for i,p in enumerate(model['ports']) if p['inward_sign']==sign)
        port=model['ports'][index]; name=port['name']
        archive=folder/(name+'-monitors.npz')
        sim,monitors=build_simulation(mp,model,index,config['resolution'],True)
        monitor=monitors[index]
        try:
            with np.load(archive,allow_pickle=False) as arrays:
                stored=arrays['incident'].copy()
                data=type(sim.get_flux_data(monitor))(E=arrays['reference_E'],H=arrays['reference_H'])
                sim.load_flux_data(monitor,data)
            y,z=sim.get_array_metadata(dft_cell=monitor)[1:3]
            iy,iz=np.argmin(abs(y-port['y'])),np.argmin(abs(z))
            electric=np.array([np.squeeze(sim.get_dft_array(monitor,mp.Ey,f))[iy,iz] for f in range(3)])
            if np.min(abs(electric))<=1e-8 or np.min(abs(stored))<=1e-8:
                raise ValueError('A centre sample or mode coefficient is too weak to check phase.')
            # Convert old signed flux-overlap coefficients once. A run from the
            # corrected adapter already stores the electric-amplitude convention.
            corrected=stored*sign if report.get('phase_convention_version',1)==1 else stored
            projected,_,_,_=project(mp,sim,monitor,port,model,sign,config['resolution'])
            projection_error=float(np.linalg.norm(projected-corrected)/np.linalg.norm(corrected))
            if projection_error>1e-8:
                raise ValueError('Live native projection differs from the archived amplitude conversion.')
            old=corrected*sign
            old_phase=np.angle(electric/old); new_phase=np.angle(electric/corrected)
            rows.append({'port':name,'inward_sign':sign,'physical_plane_x_um':port['x'],
                'sample_yz_um':[float(y[iy]),float(z[iz])],'frequency_per_um':model['frequency_per_um'],
                'native_reference_Ey':np.stack((electric.real,electric.imag),axis=-1).tolist(),
                'phase_Ey_over_historic_flux_amplitude_rad':old_phase.tolist(),
                'phase_Ey_over_electric_amplitude_rad':new_phase.tolist(),
                'live_native_projection_relative_l2':projection_error,
                'source_monitor_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
        finally:
            sim.reset_meep()
    maximum=max(abs(v) for row in rows for v in row['phase_Ey_over_electric_amplitude_rad'])
    if maximum>.025:
        raise ValueError(f'Native signed Ey disagrees with electric amplitude by up to {maximum:.6g} rad.')
    return {'passed':True,'case':model['case'],'meep_version':mp.__version__,
            'method':'Original native reference E/H DFT data are loaded onto their exact Meep monitor. Direct centre-nearest Ey phase is compared to the raw and sign(group velocity)-corrected amplitudes. No new time stepping or fitted rotation.',
            'max_absolute_electric_phase_residual_rad':maximum,'guard_rad':.025,'rows':rows,
            'source':'https://github.com/NanoComp/meep/blob/v1.35.0/src/mpb.cpp#L963-L982'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.run)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
