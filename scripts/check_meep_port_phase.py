"""Verify the signed MPB coefficient convention with a normal-incidence Fresnel step."""
import argparse
import json
from pathlib import Path
import numpy as np

FREQUENCY=1/1.55


def stage(mp,resolution,side,device):
    # Uniform periodic transverse section. Locate the interface at x=0.
    port=mp.Vector3(side)
    incident_n = 1. if side<0 else 2.
    geometry=[mp.Block(center=mp.Vector3(2.),size=mp.Vector3(4.,mp.inf,mp.inf),material=mp.Medium(index=2.))]
    source=mp.Source(mp.GaussianSource(FREQUENCY,fwidth=.16,cutoff=5),component=mp.Ez,
                     center=mp.Vector3(side*2),size=mp.Vector3(0,.5))
    sim=mp.Simulation(cell_size=mp.Vector3(8.,.5),resolution=resolution,
        default_material=mp.Medium(index=1. if device else incident_n),geometry=geometry if device else [],
        boundary_layers=[mp.PML(1.,direction=mp.X)],sources=[source],dimensions=2,
        k_point=mp.Vector3())
    plane=mp.Volume(center=port,size=mp.Vector3(0,.5))
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        monitor=sim.add_mode_monitor(FREQUENCY,0,1,mp.ModeRegion(center=port,size=mp.Vector3(0,.5)))
        local=sim.add_dft_fields([mp.Ez],FREQUENCY,0,1,center=port,size=mp.Vector3())
        sim.run(until_after_sources=45)
        e=complex(np.asarray(sim.get_dft_array(local,mp.Ez,0)).item())
        coeff=[]
        for branch,sign in enumerate((1,-1)):
            mp.set_random_seed(0)
            sol=sim.get_eigenmode_coefficients(monitor,[1],eig_parity=mp.ODD_Z,eig_resolution=resolution,
                direction=mp.X,kpoint_func=lambda f,b:mp.Vector3(sign*incident_n*f))
            mp.set_random_seed(0)
            mode=sim.get_eigenmode(FREQUENCY,mp.X,plane,1,sol.kpoints[0],parity=mp.ODD_Z,
                                   resolution=resolution)
            anchor=mode.amplitude(port,mp.Ez)
            if not abs(anchor)>1e-7 or not sign*sol.vgrp[0]>0:raise ValueError('Invalid signed mode.')
            coeff.append(complex(sol.alpha[0,0,branch]*anchor/abs(anchor)))
        return e,np.array(coeff)
    finally:
        sim.reset_meep()


def pair(value):
    return [float(value.real),float(value.imag)]


def main():
    import meep as mp
    mp.verbosity(0)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    rows=[]
    for res in (32,64,96):
        for side,n1,n2 in [(-1,1.,2.),(1,2.,1.)]:
            ref,a=stage(mp,res,side,False);dev,b=stage(mp,res,side,True)
            forward = 0 if side==-1 else 1
            raw=(b[1-forward]-a[1-forward])/a[forward]
            signed_flux_norm= -raw
            direct=(dev-ref)/ref
            theory=(n1-n2)/(n1+n2)*np.exp(4j*np.pi*n1*FREQUENCY)
            rows.append({'resolution':res,'incident_from':'left' if side<0 else 'right','native_field_reflection':pair(direct),
                         'unsigned_MPB_ratio':pair(raw),'signed_MPB_ratio':pair(signed_flux_norm),'analytic':pair(theory),
                         'unsigned_minus_direct_abs':abs(raw-direct),'signed_minus_direct_abs':abs(signed_flux_norm-direct),
                         'signed_minus_analytic_abs':abs(signed_flux_norm-theory)})
    passed=all(row['signed_minus_direct_abs']<.004 and row['unsigned_minus_direct_abs']>.55 for row in rows) and all(
        row['signed_minus_analytic_abs']<.009 for row in rows if row['resolution']==96)
    artifact={'passed':passed,'solver':'Meep','version':mp.__version__,'source_url':'https://github.com/NanoComp/meep/blob/v1.35.0/src/mpb.cpp#L963-L982',
      'observable':'Electric reflected/incident amplitude one micrometre from a dielectric step. Direct native Ez DFT subtraction and Fresnel theory independently determine the phase.',
      'rows':rows}
    with args.output.open('x') as f:json.dump(artifact,f,indent=2);f.write('\n')
    if not passed:raise SystemExit('Check recorded field/phase residuals.')
if __name__=='__main__':main()
