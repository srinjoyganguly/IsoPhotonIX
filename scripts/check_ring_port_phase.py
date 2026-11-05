"""Verify the ring projector's Ey-positive backward coefficient against a real dielectric step."""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .calibrate_ring import electric_directional_components
else:
    from calibrate_ring import electric_directional_components


def fields(mp,resolution,side,step):
    incoming,outgoing=(1.,2.) if side<0 else (2.,1.)
    physical_side=-side
    if step:
        geometry=[mp.Block(size=mp.Vector3(4.,mp.inf,mp.inf),center=mp.Vector3(physical_side*2.),
                           material=mp.Medium(index=outgoing))]
    else:
        geometry=[]
    frequency=1/1.55
    source=mp.Source(mp.GaussianSource(frequency,fwidth=.18),component=mp.Ey,
                     center=mp.Vector3(side*2.),size=mp.Vector3(0,1.))
    monitor=mp.Vector3(side,0)
    sim=mp.Simulation(cell_size=mp.Vector3(8.,1.),k_point=mp.Vector3(),resolution=resolution,
                      geometry=geometry,default_material=mp.Medium(index=incoming),
                      sources=[source],boundary_layers=[mp.PML(.8,direction=mp.X)])
    try:
        dft=sim.add_dft_fields([mp.Ey,mp.Hz],frequency,0,1,center=monitor,size=mp.Vector3())
        sim.run(until_after_sources=45)
        values=[complex(np.asarray(sim.get_dft_array(dft,c,0)).item()) for c in (mp.Ey,mp.Hz)]
        waves=electric_directional_components(values[0]/2,values[1]/(2*incoming))
        return values[0],waves
    finally:
        sim.reset_meep()


def pair(value):
    return [float(value.real),float(value.imag)]


def main():
    import meep as mp
    mp.verbosity(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for mesh in (32,64):
        for side,n1,n2 in [(-1,1.,2.),(1,2.,1.)]:
            ref,a=fields(mp,mesh,side,False)
            device,b=fields(mp,mesh,side,True)
            index=0 if side<0 else 1
            projected=(b[1-index]-a[1-index])/a[index]
            direct=(device-ref)/ref
            analytic=(n1-n2)/(n1+n2)*np.exp(4j*np.pi*n1/1.55)
            rows.append(dict(resolution=mesh,incident_from='left' if side<0 else 'right',
                             electric_projected_reflection=pair(projected),direct_ey_reflection=pair(direct),
                             fresnel_reflection_at_port=pair(analytic),
                             complex_error_vs_direct=float(abs(projected-direct)),
                             complex_error_vs_fresnel=float(abs(projected-analytic))))
    passed=all(row['complex_error_vs_direct']<.006 for row in rows) and all(
        row['complex_error_vs_fresnel']<.012 for row in rows if row['resolution']==64)
    report=dict(solver='Meep CPU',version=mp.__version__,passed=passed,rows=rows,
        observable='Ey reflection at a plane 1 µm from an interface, checked against native DFT field subtraction and de-embedded Fresnel reflection. Both propagation directions use a positive Ey anchor.')
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
