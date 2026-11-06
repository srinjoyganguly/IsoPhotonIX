"""Compare the same five physical wavelengths across full 3D grating meshes."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_grating_domain import read
from scripts.grating_phase_matching import continuous_te
from scripts.lumerical_grating3d import model
from scripts.calibrate_finite_grating import grid, settings


def material_dispersion_screen(reference):
    """Isolate the material-only phase shift in two lossless analytic slab guides."""
    m=model(16)  # Analytic slab eigenvalues are independent of mesh spacing.
    silicon=reference["material_epsilon"]["Si (Silicon) - Palik"]["real"]
    oxide=reference["material_epsilon"]["SiO2 (Glass) - Palik"]["real"]
    phases=[]
    for wave,eps_si,eps_ox in zip(reference["wavelength_um"],silicon,oxide):
        dispersive=dict(m,n_si=np.sqrt(eps_si),n_oxide=np.sqrt(eps_ox))
        heights=(m["height"],m["height"]-m["etch"])
        delta=[continuous_te(dispersive,h,wave)-continuous_te(m,h,wave) for h in heights]
        weighted=m["fill"]*delta[0]+(1-m["fill"])*delta[1]
        phases.append({"wavelength_um":wave,"delta_neff_ridge":delta[0],"delta_neff_etched_film":delta[1],
                       "estimated_shift_over_one_pitch_rad":2*np.pi*m["period"]*weighted/wave,
                       "estimated_shift_over_25_um_rad":2*np.pi*25*weighted/wave})
    return {"project_sha256":reference["project_sha256"],"samples":phases,
            "scope":"Continuous lossless TE0 in symmetric oxide-clad 220 and 120 nm slabs, with filed permittivity versus its fixed 1550 nm value. The fill-weighted phase estimates propagation in uniform sections. The full-device field acquisition determines diffraction power."}


def vendor_grid_cost(evidence):
    """Cell-scaling lower bound under the recorded policy; candidate CUDA plans may require more."""
    vendor=json.loads((evidence/"v014-grating-vendor-mesh.json").read_text())
    result={}
    for geometry,profile_name in (("lumerical","v014-grating-memory-lumerical-mirror-r20.json"),
                                  ("tidy3d","v014-grating-memory-tidy3d-mirror-r16.json")):
        profile_path=evidence/profile_name
        if not profile_path.is_file():
            continue
        profile=json.loads(profile_path.read_text())
        dz=min(vendor[geometry]["etched_part"]["intersected_steps_nm"])
        target=int(np.ceil(1000/dz))
        m=settings(geometry,12); m["resolution"]=target
        full=grid(m)[0]
        native=(full[0],full[1]//2,full[2])
        cells=int(np.prod(native))
        bound=max(1024+1.2*stage["estimate"]["baseline_native_peak_mib"]*cells/stage["base"]["cells"]
                  for stage in profile["roles"].values())
        result[geometry]={"filed_etch_dz_nm":dz,"target_uniform_mesh_per_um":target,
                          "native_mirror_shape":native,"native_cells":cells,
                          "cell_scaling_min_required_free_mib":bound,"baseline_profile":profile_name,
                          "scope":"Planning lower bound from measured r14 native allocation and cell-count scaling plus reserve. This candidate has zero measured FDTD steps and no compiled admission plan."}
    return result


def summarize(paths):
    groups = {}
    designs = {}
    evidence = Path(__file__).resolve().parents[1]/"docs/verification/v012-grating-commercial-references.json"
    references = json.loads(evidence.read_text())
    for path in paths:
        row, arrays = read(path)
        config = row["settings"]
        geometry = config["geometry"]
        if geometry not in ("lumerical", "tidy3d"):
            raise ValueError("Only a geometry-matched published benchmark is comparable.")
        design={k:v for k,v in config.items() if k not in ("grating_domain","grating_allocator","resolution")}
        if geometry in designs and designs[geometry]!=design:
            raise ValueError(f"Mesh trend requires one physical design and material model for {geometry}.")
        designs[geometry]=design
        reference = next(v for v in row["comparisons"] if v["geometry_matched"])
        waves = np.array(reference["wavelength_um"])
        published = np.asarray(reference["published_s21_power"] if geometry == "lumerical" else reference["published_power"])
        with np.load(arrays, allow_pickle=False) as data:
            np.testing.assert_array_equal(waves, data["dev_wavelength_um"])
            p = np.abs(data["output_mode"])**2/data["incident_power"]
            np.testing.assert_allclose(p, row["result"]["beam_to_guide"])
            weak_return = (abs(data["reflected_fiber_mode"])**2).tolist() if geometry == "lumerical" else None
            neff = data["dev_mode_neff"].real.tolist()
        delta = 100*(p-published)
        physical_etch_nm = 1000*config["etch"]
        sample = {"mesh":config["resolution"], "spacing_nm":1000/config["resolution"],
                  "etch_nm":physical_etch_nm,"etch_cells":config["etch"]*config["resolution"],
                  "arrays_sha256":row["arrays_sha256"], "wavelength_um":waves.tolist(),
                  "power":p.tolist(), "published_power":published.tolist(), "delta_percentage_points":delta.tolist(),
                  "mean_absolute_delta_pp":float(np.mean(abs(delta))), "max_absolute_delta_pp":float(max(abs(delta))),
                  "fiber_return_power":weak_return, "port_neff":neff,
                  "stages":{k:{"shape":v["shape"],"steps":v["time_steps"],"freshness":v.get("acquisition"),
                               "performance":v.get("performance")} for k,v in row["stages"].items()}}
        if geometry=="tidy3d":
            readout=references["tidy3d"]["calibration"]["conservative_readout_uncertainty_db"]
            db_delta=10*np.log10(p/published)
            sample["delta_db"]=db_delta.tolist()
            sample["inside_published_figure_readout"]=np.less_equal(abs(db_delta),readout).tolist()
        if weak_return is not None:
            filed = references["lumerical"]
            if reference["vendor_file_sha256"] != filed["s_file_sha256"]:
                raise ValueError("Return and transmission references belong to different S-parameter files.")
            vr = np.array(filed["samples"]["S11"]["power"])
            sample["published_fiber_return_power"] = vr.tolist()
            sample["fiber_return_minus_published_pp"] = (100*(np.asarray(weak_return)-vr)).tolist()
        groups.setdefault(geometry,[]).append(sample)
    for runs in groups.values():
        runs.sort(key=lambda r:r["mesh"])
        if len({r["mesh"] for r in runs})!=len(runs):
            raise ValueError("Each physical mesh may appear only once in a trend.")
        for r1, r2 in zip(runs,runs[1:]):
            np.testing.assert_array_equal(r1["wavelength_um"],r2["wavelength_um"])
            d=100*(np.asarray(r2["power"])-r1["power"])
            r2["previous_mesh"]=r1["mesh"]
            r2["change_vs_previous_pp"]=d.tolist()
            r2["max_absolute_change_vs_previous_pp"]=float(max(abs(d)))
            r2["within_1pp_mesh_target_every_bin"]=bool(np.all(abs(d)<=1.))
    return {"schema_version":1,"observables":"Independently illuminated full-width gratings. Power is output TE0 power divided by measured incoming finite-aperture power. Each reference belongs to its own published design.",
            "mesh_power_target_pp":1.0,"mesh_power_target_scope":"Successive physically matched meshes differ by at most one absolute percentage point at every sampled wavelength. This workbench stability screen is independent of external-solver agreement.",
            "tidy_readout_uncertainty_db":0.08,"filed_material_dispersion":material_dispersion_screen(references["lumerical"]),
            "uniform_grid_cost_for_published_etch_sampling":vendor_grid_cost(evidence.parent),"groups":groups}


def framebuffer_observations(path):
    from datetime import datetime
    rows = {}
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        if len(sample["jobs"]) != 1:
            raise ValueError("A grating memory sample identifies overlapping GPU workers.")
        geometry,mesh,stage = sample["jobs"][0]
        rows.setdefault((geometry,mesh,stage),[]).append(sample)
    grouped = []
    for (geometry,mesh,stage), samples in sorted(rows.items()):
        times = [datetime.fromisoformat(s["utc"]) for s in samples]
        grouped.append({"geometry":geometry,"mesh":mesh,"stage":stage,"samples":len(samples),
                        "sample_span_seconds":(times[-1]-times[0]).total_seconds(),
                        "maximum_observed_global_used_mib":max(s["used_mib"] for s in samples),
                        "minimum_observed_free_mib":min(s["free_mib"] for s in samples)})
    return {"scope":"Read-only 2 s polling of global framebuffer usage during selected FDTDX workers. Graphics and other allocations are included. Sampling began late in the full 16-cell incident run. These are observed high-water marks.",
            "stages":grouped}


def plot_curves(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometries=[v for v in ("lumerical","tidy3d") if v in report["groups"]]
    fig,axes=plt.subplots(2,len(geometries),figsize=(5*len(geometries),6.6),squeeze=False,sharex=True,layout="constrained")
    palette={12:"#737e8c",14:"#0072b2",16:"#d55e00",18:"#994a88",20:"#13846d"}
    for column,geometry in enumerate(geometries):
        runs=report["groups"][geometry]
        waves=1000*np.array(runs[0]["wavelength_um"])
        power,delta=axes[:,column]
        power.plot(waves,100*np.asarray(runs[0]["published_power"]),"ks--",label="Published")
        if geometry=="tidy3d":
            expected=100*np.asarray(runs[0]["published_power"])
            error=10**(report["tidy_readout_uncertainty_db"]/10)
            power.fill_between(waves,expected/error,expected*error,color="black",alpha=.12)
        for run in runs:
            mesh=run["mesh"]
            power.plot(waves,100*np.asarray(run["power"]),"o-",color=palette[mesh],label=f"{mesh}/µm")
            delta.plot(waves,run["delta_percentage_points"],"o-",color=palette[mesh])
        title="Lumerical curved fiber grating" if geometry=="lumerical" else "Tidy3D straight Gaussian grating"
        power.set_title(title)
        power.legend(frameon=False,ncol=2,fontsize=8)
        delta.axhline(0,color="#555",linewidth=.8)
        delta.set_xlabel("Wavelength (nm)")
        delta.set_xticks(waves)
        for ax in (power,delta):
            ax.grid(axis="y",linewidth=.6,alpha=.25)
            ax.spines[["top","right"]].set_visible(False)
    axes[0,0].set_ylabel("Captured TE₀ power (%)")
    axes[1,0].set_ylabel("Local − published (percentage points)")
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",type=Path)
    parser.add_argument("--framebuffer-samples",type=Path)
    args=parser.parse_args()
    report=summarize(args.receipts)
    if args.framebuffer_samples:
        report["observed_framebuffer"]=framebuffer_observations(args.framebuffer_samples)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if args.plot:
        plot_curves(report,args.plot)
