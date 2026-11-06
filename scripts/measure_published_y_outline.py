"""Measure the regenerated PhotonForge Y outline against the published upper-arm loss curve."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.calibration_checkpoint import MeasurementCheckpoints
from scripts.compare_published_splitter import CLAD_EPS, CORE_EPS, NOTEBOOK_SHA, TIDY, WAVELENGTHS, WIDTHS


def area(poly):
    p=np.asarray(poly,dtype=float)
    return .5*float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))


def vertical_span(poly,x):
    p=np.asarray(poly,dtype=float)
    a=p; b=np.roll(p,-1,axis=0)
    crossed=(a[:,0]>x)!=(b[:,0]>x)
    a,b=a[crossed],b[crossed]
    ys=a[:,1]+(x-a[:,0])*(b[:,1]-a[:,1])/(b[:,0]-a[:,0])
    if len(ys)!=2:
        raise ValueError(f'Expected two edges of one outgoing arm at x={x}.')
    return [float(ys.min()),float(ys.max())]


def render_check(notebook,upper):
    """Read the pinned notebook's stored lower-arm illustration at its original pixel scale."""
    from PIL import Image
    raw=Path(notebook).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=NOTEBOOK_SHA:
        raise ValueError('The embedded geometry must come from the pinned public notebook.')
    cell=json.loads(raw)['cells'][13]
    obj=next(o['data']['image/png'] for o in cell['outputs'] if 'image/png' in o.get('data',{}))
    pixels=np.asarray(Image.open(io.BytesIO(base64.b64decode(obj))).convert('RGB'),dtype=int)
    if pixels.shape[:2]!=(299,643):
        raise ValueError('Pinned illustrated axes have changed.')
    scale=68.875  # both axes, x=2..10 at pixel centres 73..624
    zero_y=45.8   # y=0 tick centre
    checks=[]
    for x in np.arange(3.5,9.,.5):
        col=pixels[:,round(73+scale*(x-2))]
        green=np.flatnonzero((col[:,1]>col[:,0]+35)&(col[:,1]>col[:,2]+25)&(col[:,0]<100))
        if len(green)<8:
            raise ValueError('Cannot resolve the published rendered arm.')
        measured=float((green.mean()-zero_y)/scale)
        fresh=float(np.mean(vertical_span(upper,x)))
        s=x-3; t=s/6
        prose=.35+2*(t-np.sin(2*np.pi*t)/(2*np.pi))
        arc=.35+(5-np.sqrt(25-s*s) if s<=3 else np.sqrt(25-(6-s)**2)-3)
        checks.append({'x_um':float(x),'absolute_rendered_centre_um':measured,'regenerated_span_midpoint_um':fresh,
                       'text_sine_centre_um':float(prose),'two_arc_centre_um':float(arc)})
    error=max(abs(c['regenerated_span_midpoint_um']-c['absolute_rendered_centre_um']) for c in checks)
    if error>2/scale:
        raise ValueError('Regenerated arm differs from the notebook illustration by more than two pixels.')
    return {'image_cell_index':13,'pixel_pitch_um':1/scale,'max_regenerated_midpoint_error_um':error,'samples':checks}


def extract(notebook,output):
    import photonforge as pf
    # Same calls and coordinates as cell 13 of the pinned public notebook.
    upper=pf.Path(origin=(3,.35),width=.5)
    upper.s_bend((6,2),relative=True)
    upper.segment((100,0),relative=True)
    polygons=pf.boolean([upper,upper.copy().mirror()],[],'+')
    arms=sorted([np.asarray(p.vertices,dtype=float) for p in polygons],key=lambda p:p[:,1].mean())
    if len(arms)!=2:
        raise ValueError('Expected two distinct bend polygons.')
    for x in np.linspace(3.1,9.9,21):
        low,high=(vertical_span(p,x) for p in arms)
        if not np.allclose(low,[-high[1],-high[0]],atol=1e-8,rtol=0):
            raise ValueError('PhotonForge output arms are not mirror images.')
    audit=render_check(notebook,arms[1])
    record={'model_id':'published-y-photonforge-outline-v1','notebook_sha256':NOTEBOOK_SHA,'source_url':TIDY,
            'photonforge_version':pf.__version__,'units':'µm','coordinates':'Original notebook coordinates before translating x by -5 and z by -0.11.',
            'render_check':audit,'arms_lower_upper':[p.tolist() for p in arms],
            'scope':'Vertices regenerated locally with the named Community release and default Path.s_bend. The archived vendor notebook does not pin the PhotonForge version or preserve its numeric vertices. Geometry agreement is checked against its embedded image.'}
    with Path(output).open('x') as f:
        json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    return record


def physical_geometry(mp,record,reference):
    silicon=mp.Medium(epsilon=CORE_EPS)
    if reference:
        return [mp.Block(size=mp.Vector3(mp.inf,.5,.22),material=silicon)]
    def prism(poly):
        vertices=np.asarray(poly)
        if area(vertices)<0:
            vertices=vertices[::-1]
        return mp.Prism([mp.Vector3(x-5,y,-.11) for x,y in vertices],height=.22,material=silicon)
    xs=np.linspace(1.,3.,len(WIDTHS))
    stem=np.vstack([np.c_[xs,np.asarray(WIDTHS)/2],np.c_[xs[::-1],-np.asarray(WIDTHS)[::-1]/2]])
    input_lead=mp.Block(center=mp.Vector3(-54.),size=mp.Vector3(100,.5,.22),material=silicon)
    return [input_lead,prism(stem),*[prism(p) for p in record['arms_lower_upper']]]


def native_stage(record,resolution,reference):
    import meep as mp
    mp.verbosity(0)
    frequencies=1/WAVELENGTHS
    aperture=mp.Vector3(0,2,1.32)
    source=mp.EigenModeSource(mp.GaussianSource(1/1.55,fwidth=.12,cutoff=5),center=mp.Vector3(-4.5),size=aperture,
            eig_band=1,direction=mp.X,eig_kpoint=mp.Vector3(2.5/1.55),eig_match_freq=True,
            eig_parity=mp.ODD_Y+mp.EVEN_Z,eig_resolution=resolution,eig_tolerance=1e-9)
    sim=mp.Simulation(cell_size=mp.Vector3(11.6,1.2+4+1.5*1.55+1.6,3.8),resolution=resolution,sources=[source],
            geometry=physical_geometry(mp,record,reference),default_material=mp.Medium(epsilon=CLAD_EPS),
            symmetries=[mp.Mirror(mp.Y,phase=-1),mp.Mirror(mp.Z,phase=1)],boundary_layers=[mp.PML(.8)])
    planes=[(-4.10,0.),(4.5,0.)] if reference else [(4.5,2.35),(4.5,-2.35)]
    try:
        mp.set_random_seed(0)
        sim.init_sim()
        mode=[]; dfts=[]
        for x,y in planes:
            pt=mp.Vector3(x,y)
            mode.append(sim.add_mode_monitor(frequencies.tolist(),mp.ModeRegion(center=pt,size=aperture),decimation_factor=1))
            dfts.append(sim.add_dft_fields([mp.Ey,mp.Hz],frequencies.tolist(),center=pt,size=aperture,decimation_factor=1))
        sim.run(until_after_sources=65.)
        def samples(d):
            return np.array([np.concatenate([np.ravel(sim.get_dft_array(d,component,i)) for component in (mp.Ey,mp.Hz)]) for i in range(len(frequencies))])
        # The two mirror-related native DFT buffers can have different pixel bounds.
        early=[samples(d) for d in dfts]; sim.run(until=20.)
        late=[samples(d) for d in dfts]
        change=np.stack([np.linalg.norm(b-a,axis=-1)/np.linalg.norm(b,axis=-1) for a,b in zip(early,late)])
        amps=[];indices=[]
        for m in mode:
            if not np.array_equal(mp.get_flux_freqs(m),frequencies):
                raise ValueError('Unexpected native frequency order.')
            mp.set_random_seed(0)
            coef=sim.get_eigenmode_coefficients(m,[1],direction=mp.X,eig_parity=mp.EVEN_Z,eig_resolution=resolution,
                    eig_tolerance=1e-9,kpoint_func=lambda f,b:mp.Vector3(2.5*f))
            amps.append(coef.alpha[0,:,0].copy());indices.append(np.array([kp.x for kp in coef.kpoints])/frequencies)
        guided=bool(np.all(np.asarray(indices)>np.sqrt(CLAD_EPS)) and np.all(np.asarray(indices)<np.sqrt(CORE_EPS)))
        return {'planes_um':planes,'guided':guided,'late_increment_per_plane_per_frequency':change.tolist(),
                'settled':bool(guided and np.isfinite(change).all() and np.max(change)<1e-3),'timesteps':int(sim.timestep())}, \
               {'forward_modes':np.asarray(amps),'mode_indices':np.asarray(indices),'frequency_per_um':frequencies,
                **{f'plane_{i}_Ey_Hz_samples':fields for i,fields in enumerate(late)}}
    finally:
        sim.reset_meep()


def prior_straight(folder,settings,checkpoints):
    """Read a verified pre-layout-fix straight stage from the same numerical implementation."""
    p=Path(folder)
    marker=json.loads((p/'receipt.json').read_text())
    spec=marker['specification']
    # c8158bc recorded both equal-size *straight* monitor buffers in a single array.
    # Its unrelated mirrored device buffers required separate storage. Native field
    # evolution, source, mode projection and reference planes were identical.
    provenance=dict(checkpoints.provenance,worker_sha256='7171c3fe37b8f81d21e60062305ebb06ab3f07ea7760333e6b31078c583417dd')
    if (spec!={'label':'reference','settings':settings,'provenance':provenance} or
            checkpoints.json_hash(spec)!=marker['signature'] or
            checkpoints.json_hash(marker['report'])!=marker['report_sha256'] or
            checkpoints.file_hash(p/'arrays.npz')!=marker['arrays_sha256'] or not marker['report']['settled']):
        raise ValueError('Historic straight stage has incompatible provenance, settings, settling or arrays.')
    with np.load(p/'arrays.npz',allow_pickle=False) as raw:
        data={k:raw[k] for k in raw.files}
    if data['forward_modes'].shape!=(2,len(WAVELENGTHS)) or not np.array_equal(data['frequency_per_um'],1/WAVELENGTHS):
        raise ValueError('Unexpected historic straight modes or wavelengths.')
    return {**marker['report'],'reused_native_reference_sha256':marker['arrays_sha256'],
            'reused_reference_worker_sha256':provenance['worker_sha256']},data


def measure(vertices,output,resolution,only=None,historic_reference=None):
    verification=json.loads(Path(vertices).read_text())
    if verification['model_id']!='published-y-photonforge-outline-v1' or verification['notebook_sha256']!=NOTEBOOK_SHA:
        raise ValueError('Unexpected geometry or pinned source.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():
        raise FileExistsError('This outline study is complete; choose a fresh output directory.')
    checkpoints=MeasurementCheckpoints(out/'stages',__file__)
    geometry_hash=hashlib.sha256(Path(vertices).read_bytes()).hexdigest()
    stages={}; settings={'resolution':resolution,'geometry_sha256':geometry_hash,'frequency_per_um':(1/WAVELENGTHS).tolist(),
                         'junction_widths_um':WIDTHS,'notebook_sha256':NOTEBOOK_SHA,
                         'symmetry_xyz':[0,-1,1],'core_cladding_epsilon':[CORE_EPS,CLAD_EPS]}
    for name in ([only] if only else ['reference','device']):
        if name=='reference' and historic_reference:
            stages[name]=prior_straight(historic_reference,settings,checkpoints)
        else:
            stages[name]=checkpoints.acquire(name,settings,lambda: native_stage(verification,resolution,name=='reference'))
    if only:
        return stages[only][0]
    ref,ra=stages['reference'];dev,da=stages['device']
    near,far=ra['forward_modes'];upper,lower=da['forward_modes']
    if np.min(abs(near))<1e-8 or np.min(abs(near)**2)/np.max(abs(near)**2)<1e-5:
        raise ValueError('The straight reference has insufficient incident power.')
    top=abs(upper/near)**2;bottom=abs(lower/near)**2;through=abs(far/near)**2
    checks={'valid_stages':ref['settled'] and dev['settled'],'straight_through_power':bool(np.all(abs(through-1)<.02)),
            'mirror_power':bool(np.max(abs(top-bottom))<.002),'upper_arm_passive':bool(np.all(2*top<1.025))}
    arrays=out/'arrays.npz'
    np.savez_compressed(arrays,frequency_per_um=1/WAVELENGTHS,wavelength_um=WAVELENGTHS,upper_mode=upper,lower_mode=lower,
                        incident_mode=near,straight_far_mode=far,power_upper=top,power_lower=bottom,straight_through_power=through)
    # Positive-loss readouts of the pinned vendor's stored 2*upper-arm plot, ±0.004 dB.
    vendor=np.array([.2364864865,.1594594595,.1317567568,.1371621622,.1520270270]); loss=-10*np.log10(2*top)
    report={'model_id':verification['model_id'],'geometry_sha256':geometry_hash,'source_url':TIDY,'notebook_sha256':NOTEBOOK_SHA,
            'photonforge_geometry_version':verification['photonforge_version'],'solver':checkpoints.provenance,'resolution':resolution,
            'passed':all(checks.values()),'checks':checks,'wavelength_um':WAVELENGTHS.tolist(),'upper_power_fraction':top.tolist(),
            'lower_mirrored_power_fraction':bottom.tolist(),'upper_times_two_loss_db':loss.tolist(),
            'measured_sum_loss_db':(-10*np.log10(top+bottom)).tolist(),'vendor_plot_loss_db':vendor.tolist(),
            'difference_from_vendor_db':(loss-vendor).tolist(),'vendor_readout_uncertainty_db':.004,'reference':ref,'device':dev,
            'symmetry_xyz':[0,-1,1],'straight_through_power':through.tolist(),'arrays_sha256':hashlib.sha256(arrays.read_bytes()).hexdigest(),
            'scope':'One centred fundamental TE input and its physical upper output; mirror symmetry yields the lower port. Positive loss is -10log10(2Pupper), normalized to an independent straight run in the same solver, geometry of the source region and symmetries. Vendor geometry regenerated from the named PhotonForge Community release. Meep freezes indices at 1.55 µm, has a broader reference-normalized pulse, centres the finite z cell on the core and uses a uniform mesh.'}
    with (out/'report.json').open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    extraction=subs.add_parser('extract');extraction.add_argument('--notebook',type=Path,required=True)
    calculation=subs.add_parser('run');calculation.add_argument('--vertices',type=Path,required=True)
    calculation.add_argument('--resolution',type=int,choices=(20,24),default=20)
    calculation.add_argument('--stage',choices=('reference','device'))
    calculation.add_argument('--historic-reference',type=Path,help='verified straight checkpoint from the initial outline study')
    for item in (extraction,calculation): item.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='extract':
        extract(args.notebook,args.output)
    else:
        result=measure(args.vertices,args.output,args.resolution,args.stage,args.historic_reference)
        print(json.dumps({k:result.get(k) for k in ('passed','settled','checks','upper_times_two_loss_db') if k in result},indent=2))
        if not result.get('passed',result.get('settled')):raise SystemExit(2)
