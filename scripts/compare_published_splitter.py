"""Reconstruct the published Tidy3D thirteen-width Y and measure both output arms."""
import argparse
import fcntl
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

TIDY = 'https://www.flexcompute.com/tidy3d/examples/notebooks/YJunction/'
ANSYS = 'https://optics.ansys.com/hc/en-us/articles/360042800353-Y-branch'
NOTEBOOK_SHA = 'bc049b36b9b0e04cd83d79fd73550a2fe1a6e5b10562683957b0d3a9f21f95e9'
WIDTHS = [.50,.50,.60,.70,.90,1.26,1.40,1.40,1.40,1.40,1.31,1.20,1.20]
WAVELENGTHS = np.array([1.50,1.525,1.55,1.575,1.60])
# Real permittivities sampled at 1.55 µm from the archived notebook's named
# Palik_Lossless variants (Tidy3D 2.11.2). The archived program is the model
# authority; the web transcription also uses the historical name Palik_LowLoss.
CORE_EPS, CLAD_EPS = 12.091141286297306, 2.0867588783784097


def outline(extent):
    x = np.linspace(-4,-2,len(WIDTHS))
    upper_stem=[(-extent,.25), *zip(x,np.array(WIDTHS)/2)]
    t=np.linspace(0.,1.,121)
    center_x=-2+6*t
    center_y=.35+2*(t-np.sin(2*np.pi*t)/(2*np.pi))
    slope=(1-np.cos(2*np.pi*t))/3
    normal_y=1/np.hypot(1,slope)
    normal_x=-slope*normal_y
    outer=list(zip(center_x+.25*normal_x,center_y+.25*normal_y))
    inner=list(zip(center_x-.25*normal_x,center_y-.25*normal_y))
    top_outer=upper_stem+outer[1:]+[(extent,2.6)]
    top_inner=[(extent,2.1), *reversed(inner)]
    top=top_outer+top_inner
    # Join upper/lower inner boundaries with the vertical notch at x=-2.
    return top+[(x,-y) for x,y in reversed(top)]


def stage(resolution, reference):
    import meep as mp
    mp.verbosity(0)
    pml=.8
    cell=mp.Vector3(10+2*pml,1.2+4+1.5*1.55+2*pml,2.2+2*pml)
    silicon=mp.Medium(epsilon=CORE_EPS)
    if reference:
        solids=[mp.Block(size=mp.Vector3(mp.inf,.5,.22),material=silicon)]
    else:
        points=outline(cell.x/2+.1)
        solids=[mp.Prism([mp.Vector3(x,y,-.11) for x,y in reversed(points)],height=.22,material=silicon)]
    aperture=mp.Vector3(0,2,1.32)
    freq=1/1.55
    source=mp.EigenModeSource(mp.GaussianSource(freq,fwidth=.12,cutoff=5),center=mp.Vector3(-4.5),
        size=aperture,eig_band=1,eig_match_freq=True,eig_kpoint=mp.Vector3(2.5*freq),
        eig_parity=mp.ODD_Y+mp.EVEN_Z,eig_resolution=resolution,eig_tolerance=1e-9,direction=mp.X)
    sim=mp.Simulation(cell_size=cell,resolution=resolution,geometry=solids,sources=[source],
                      symmetries=[mp.Mirror(mp.Z)],boundary_layers=[mp.PML(pml)],
                      default_material=mp.Medium(epsilon=CLAD_EPS))
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        positions=[(-4.10,0.)] if reference else [(4.5,-2.35),(4.5,2.35)]
        monitors=[sim.add_mode_monitor((1/WAVELENGTHS).tolist(),mp.ModeRegion(center=mp.Vector3(x,y),size=aperture))
                  for x,y in positions]
        sim.run(until_after_sources=65.)
        before=[sim.get_flux_data(m) for m in monitors]
        sim.run(until=20.)
        values=[];change=[]
        for monitor,old in zip(monitors,before):
            new=sim.get_flux_data(monitor)
            delta=sum(np.linalg.norm(x-y)**2 for x,y in zip(new,old))
            scale=sum(np.linalg.norm(x)**2 for x in new)
            change.append(float(np.sqrt(delta/scale)))
            mp.set_random_seed(0)
            coef=sim.get_eigenmode_coefficients(monitor,[1],direction=mp.X,eig_parity=mp.EVEN_Z,
                eig_resolution=resolution, eig_tolerance=1e-9,kpoint_func=lambda f,b:mp.Vector3(2.5*f))
            values.append(np.asarray(coef.alpha[0,:,0]))
        return np.asarray(values),change
    finally:
        sim.reset_meep()


def recorded_stage(directory,resolution,reference):
    kind='reference' if reference else 'device'
    archive=directory/(kind+'.npz')
    spec={'resolution':resolution,'reference':reference,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    receipt=directory/(kind+'.json')
    with (directory/(kind+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            meta=json.loads(receipt.read_text())
            if meta['spec']!=spec or meta['sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest():
                raise ValueError('Incompatible published-splitter checkpoint.')
            with np.load(archive,allow_pickle=False) as data:
                return data['modes'],data['delta'].tolist()
        modes,delta=stage(resolution,reference)
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            raw=Path(tmp)/'stage.npz'
            np.savez_compressed(raw,modes=modes,delta=np.asarray(delta))
            spec_path=Path(tmp)/'stage.json'
            spec_path.write_text(json.dumps({'spec':spec,'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})+'\n')
            raw.replace(archive)
            spec_path.replace(receipt)
        return modes,delta


def main():
    import meep as mp
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution',type=int,choices=(16,20,24,28),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['reference','device'],help='save one independently resumable native stage')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():
        parser.error('Choose a fresh result directory.')
    if args.stage:
        recorded_stage(args.output,args.resolution,args.stage=='reference')
        return
    incident,delta_in=recorded_stage(args.output,args.resolution,True)
    arms,delta_out=recorded_stage(args.output,args.resolution,False)
    power=abs(arms/incident[0])**2
    loss=-10*np.log10(power.sum(axis=0))
    passed=bool(np.isfinite(power).all() and max(delta_in+delta_out)<1e-3 and np.max(power.sum(axis=0))<1.025)
    np.savez_compressed(args.output/'arrays.npz',incident=incident,arms=arms,power=power,wavelength_um=WAVELENGTHS,
                        polygon_xy_um=np.array(outline(5.9)))
    report={'case':'published-splitter-control','passed':passed,'version':mp.__version__,'solver':'3D Meep CPU',
            'resolution':args.resolution,'vendor_source':TIDY,'vendor_notebook_sha256':NOTEBOOK_SHA,
            'waveguide_height_um':.22,'central_permittivities':[CORE_EPS,CLAD_EPS],
            'wavelength_um':WAVELENGTHS.tolist(),'power_per_arm':power.tolist(),'total_excess_loss_db':loss.tolist(),
            'mean_five_bin_loss_db':float(loss.mean()),'late_complex_monitor_changes':delta_in+delta_out,
            'scope':'Thirteen published junction widths, 2 µm junction and sinusoidal six-micrometre access bends reconstructed using constant normal path width. Real materials frozen at the reported central wavelength. Independent straight input reference and both output amplitudes. Vendor uses a nonuniform grid, a path construction library and dispersive media.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not passed:raise SystemExit('Inspect the measured power budget and convergence.')

if __name__=='__main__': main()
