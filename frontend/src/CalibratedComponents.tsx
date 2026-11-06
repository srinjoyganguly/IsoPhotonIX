import { useEffect, useState } from 'react';
import { request } from './api';
import { PlanarPortLayoutEditor, parsePlanarLayout } from './PlanarPortLayoutEditor';
import { PhaseSensitivity,type PhaseCheck } from './PhaseSensitivity';
import { SpatialPhase,type SpatialPhaseStudy } from './SpatialPhase';
import { SpatialComparison } from './SpatialComparison';
import { GpuPerformance } from './GpuPerformance';
import { CalibrationField, type FieldPlane } from './CalibrationField';
import { CodeExport } from './CodeExport';
import { CalibrationMeasurement, Spectrum, type Series } from './CalibrationMeasurement';
import './calibratedComponents.css';

type Evidence = { summary: string; dimension: string; routes: string[]; wavelengths_um: number[]; series: Series[];
  limits: string; phase_check?:PhaseCheck; phase_studies?:SpatialPhaseStudy[]; recorded?: {label:string; receipt:string;report:unknown;field?:FieldPlane;qualification?:string}[]; comparisons: {vendor: string; description: string; url: string}[] };
type Preset = { case: string; name: string; scope: string; routes: string[]; grating_meshes?: Record<string,Record<string,Record<string,number[]>>>; evidence?: Evidence };
type Job = { id: string; status: string; can_resume?:boolean;attempt?:number; created_at: string; error?: string; result?: {actual_device?:string;device_name?:string}; config: {case: string; solver: string; device: string} };
type Target = {solver: string; device: string; available: boolean; reason?: string};

export function CalibratedComponents({active}:{active:boolean}) {
  const [catalog,setCatalog]=useState<Preset[]>([]), [key,setKey]=useState('coupler'), [target,setTarget]=useState('');
  const [record,setRecord]=useState(0);
  const [materialGrid,setMaterialGrid]=useState('yee-aligned');
  const [gratingScope,setGratingScope]=useState('tidy3d-compact');
  const [ringScope,setRingScope]=useState('lumerical');
  const [layoutText,setLayoutText]=useState('');
  const [geometry3d,setGeometry3d]=useState('lumerical');
  const [ringDuration,setRingDuration]=useState('');
  const [ringInterface,setRingInterface]=useState('legacy');
  const [ringMaterial,setRingMaterial]=useState('constant');
  const [gratingMaterial,setGratingMaterial]=useState('volume-average');
  const [gratingDomain,setGratingDomain]=useState('full');
  const [gratingSpectrum,setGratingSpectrum]=useState('standard');
  const [portDistance,setPortDistance]=useState('1.25');
  const [mesh,setMesh]=useState(''),[offset,setOffset]=useState('.125');
  const [targets,setTargets]=useState<Target[]>([]),[jobs,setJobs]=useState<Job[]>([]),[jobId,setJobId]=useState('');
  const [localField,setLocalField]=useState<FieldPlane|null>(null);
  const [log,setLog]=useState('');
  const [error,setError]=useState(''),[pending,setPending]=useState(false),[report,setReport]=useState<unknown>(null);
  const selected=catalog.find(p=>p.case===key), evidence=selected?.evidence;
  const measuredDiagnostic=report!==null && typeof report==='object' && 'measurement_complete' in report && report.measurement_complete===true && 'valid_bins' in report && Number(report.valid_bins)>0;
  const job=jobs.find(j=>j.id===jobId), busy=jobs.some(j=>j.status==='running'||j.status==='queued');
  const history=jobs.filter(j=>j.config.case===key);
  const editableLayout=key==='layout-ports-2d' || key==='layout-ports-3d';
  const gpuComponent = key==='ring-3d' || key==='grating-3d';
  const denseGrating = gratingDomain==='full' && key==='grating-3d' && geometry3d==='lumerical' && (!mesh || mesh==='12');
  const dispersiveRing = key==='ring-3d' && geometry3d==='lumerical' && ringInterface==='exact-area' && (!mesh || mesh==='16');
  const meshes = key==='layout-ports-3d' ? [12,16] : key==='layout-ports-2d' ? [24,32,40] : key==='ring-3d' ? (ringInterface==='wall-tensor'?[16,20,24]:[12,16,20,24]) : key==='grating-3d' ? (selected?.grating_meshes?.[geometry3d]?.[gratingDomain]?.[gratingMaterial] ?? [12,16]) :
    key==='multimode-tm' ? [48,64,80] : key==='grating' && gratingScope!=='full' ? [28,40,60] : target.startsWith('fdtdx:') ? [12,16,20,24] : key==='multimode' ? [32,48,64] : ['multimode-3d','coupler','splitter'].includes(key)?[16,20,24]:[];
  useEffect(()=>{
    if(!editableLayout || layoutText) return;
    let live=true;
    request('/api/calibrations/planar-layout-example').then(data=>{if(live)setLayoutText(JSON.stringify(data,null,2));})
      .catch(e=>{if(live)setError(e.message);});
    return ()=>{live=false;};
  },[editableLayout,layoutText]);
  const layoutValid=!editableLayout || parsePlanarLayout(layoutText)!==null;
  const offsetValid = !key.startsWith('multimode') || (offset.trim()!=='' && Number.isFinite(Number(offset)) && +offset>=0 && +offset<=.25);
  useEffect(()=>{
    if(!active)return;
    let live=true;
    request<Preset[]>('/api/calibrations').then(c=>{if(live)setCatalog(c);})
      .catch(e=>{if(live)setError(e.message);});
    request<{targets:Target[]}>('/api/capabilities').then(t=>{if(live)setTargets(t.targets);})
      .catch(e=>{if(live)setError(e.message);});
    return ()=>{live=false;};
  },[active]);
  useEffect(()=>{
    if(!active)return;
    let live=true;
    const refresh=()=>request<Job[]>('/api/calibrations/jobs').then(rows=>{if(live)setJobs(rows);})
      .catch(e=>{if(live)setError(e.message);});
    void refresh(); const timer=setInterval(refresh,2500);
    return ()=>{live=false;clearInterval(timer);};
  },[active]);
  useEffect(()=>{
    let live=true;setReport(null);setLocalField(null);
    if(job?.status==='completed'||job?.status==='failed') {
      request('/api/calibrations/jobs/'+job.id+'/report').then(async data=>{
          if (!live) return;
          setReport(data);
          if (data && typeof data === 'object' && ('field_preview' in data || ['layout-ports-2d','multimode-tm'].includes(job.config.case))) {
            const field = await request<FieldPlane>('/api/calibrations/jobs/'+job.id+'/field');
            if (live) setLocalField(field);
          }
        })
        .catch(e=>{if(live && job.status==='completed')setError(e.message);});
    }
    return ()=>{live=false;};
  },[job?.id,job?.status]);
  useEffect(()=>{
    setLog('');
    if(!active || !job || job.status!=='running')return;
    let live=true;
    const update=()=>request<{logs:string}>('/api/jobs/'+job.id)
      .then(result=>{if(live)setLog(result.logs);}).catch(()=>{});
    void update();
    const timer=setInterval(update,5000);
    return ()=>{live=false;clearInterval(timer);};
  },[active,job?.id,job?.status]);
  const [solver,device]=target.split(':');
  const available=targets.some(t=>t.solver===solver && t.device===device && t.available);
  const configuration={revision:2,case:key,solver,device,resolution:mesh?Number(mesh):null,
    offset_um:key.startsWith('multimode')?Number(offset):null,material_model:solver==='fdtdx'&&!gpuComponent&&!editableLayout?materialGrid:null,
    planar_layout:editableLayout?parsePlanarLayout(layoutText):null,
    duration_um_c:key==='ring-3d'&&ringDuration?Number(ringDuration):null,
    ring_interface:key==='ring-3d' && ringInterface!=='legacy'?ringInterface:null,
    ring_material:dispersiveRing && ringMaterial==='palik-ade'?'palik-ade':null,
    grating_material_model:key==='grating-3d' && geometry3d==='lumerical' && gratingMaterial!=='volume-average'?gratingMaterial:null,
    grating_domain:key==='grating-3d' && gratingDomain!=='full'?gratingDomain:null,
    grating_spectrum:denseGrating && gratingSpectrum==='dense'?'dense':null,
    geometry_3d:gpuComponent?geometry3d:null,port_distance_um:key==='multimode-tm'?Number(portDistance):null,
    grating_scope:key==='grating'?gratingScope:null,ring_scope:key==='ring'?ringScope:null};
  async function run(){
    if(!selected||!target)return;
    setPending(true);setError('');
    try {
      const created=await request<Job>('/api/calibrations/jobs',configuration);
      setJobs(rows=>[created,...rows.filter(j=>j.id!==created.id)]);setJobId(created.id);
    }catch(e){setError((e as Error).message);}finally{setPending(false);}
  }
  async function cancel(){
    if(!job)return;
    try {
      const updated=await request<Job>('/api/jobs/'+job.id+'/cancel',{});
      setJobs(rows=>rows.map(j=>j.id===updated.id?updated:j));
    }catch(e){setError((e as Error).message);}
  }
  async function resume(){
    if(!job || pending)return;
    setPending(true);setError('');
    try {
      const updated=await request<Job>('/api/calibrations/jobs/'+job.id+'/resume',{});
      setJobs(rows=>rows.map(j=>j.id===updated.id?updated:j));
    }catch(e){setError((e as Error).message);}finally{setPending(false);}
  }
  function choose(next:string) {
    setKey(next); setRecord(0); setMesh(''); setPortDistance('1.25'); setRingDuration('');
    setGeometry3d(next==='grating-3d'?'compact':'lumerical');
    setRingInterface('legacy'); setRingMaterial('constant'); setGratingMaterial('volume-average'); setGratingSpectrum('standard'); setGratingDomain('full');
    setOffset('.125'); setTarget(''); setMaterialGrid('yee-aligned');
    setGratingScope('tidy3d-compact'); setRingScope('lumerical');
    setJobId(''); setError(''); setReport(null); setLocalField(null);
  }
  return <section className="cal-lab" aria-label="Calibrated components">
    <header><h1>Calibrated components</h1><p>Inspect measured reference results or run the same physical study on this machine.</p></header>
    <div className="cal-columns"><aside aria-label="Calibrated presets">{catalog.map(p=><button key={p.case} aria-pressed={key===p.case} onClick={()=>choose(p.case)}>{p.name}</button>)}</aside>
    <main>{selected && <><h2>{selected.name}</h2><p>{selected.scope}</p>
      {evidence && <section aria-label="Recorded evidence"><h3>Recorded measurements</h3><p>{evidence.summary}</p>
        <p>{evidence.dimension} · {evidence.routes.join(' / ')}</p>

        {evidence.recorded?.length ? <><label>Recorded result<select value={record} onChange={e=>setRecord(+e.target.value)}>
          {evidence.recorded.map((r,i)=><option key={r.receipt} value={i}>{r.label}</option>)}</select></label>
          {evidence.recorded[record]?.qualification && <p>{evidence.recorded[record].qualification}</p>}
          <CalibrationMeasurement key={'measurement'+key+record} value={evidence.recorded[record]?.report}/>
          <SpatialComparison key={'compare'+key+record} value={evidence.recorded[record]?.report}/>
          {evidence.recorded[record]?.field && <CalibrationField key={'field'+key+record} field={evidence.recorded[record].field!}/>}
          <a href={'https://github.com/srinjoyganguly/photonics-workbench/blob/main/docs/verification/'+evidence.recorded[record]?.receipt}
           target="_blank" rel="noreferrer">Full verification record</a></> :
          <Spectrum key={key} wavelengths={evidence.wavelengths_um} series={evidence.series}/>}
        <h3>Commercial comparisons</h3>{evidence.comparisons.map(c=><p key={c.vendor}>{c.description} <a href={c.url} target="_blank" rel="noreferrer">{c.vendor} example</a></p>)}
        <p>{evidence.limits}</p>{evidence.phase_studies && <SpatialPhase key={key} studies={evidence.phase_studies}/>}
        {evidence.phase_check && <PhaseSensitivity check={evidence.phase_check}/>}</section>}
      {selected.routes.includes('fdtdx:gpu') && <GpuPerformance key={key} preset={key}/>}
      <section><h3>Run locally</h3>
        {editableLayout && <PlanarPortLayoutEditor text={layoutText} onChange={setLayoutText}/>}
        {gpuComponent && <label>3D geometry<select value={geometry3d} onChange={e=>{setGeometry3d(e.target.value);setGratingDomain('full');setGratingMaterial('volume-average');setMesh('');}}>
          {key==='ring-3d' ? <><option value="lumerical">Lumerical add/drop ring</option><option value="tidy3d">Tidy3D add/drop ring</option></> :
            <><option value="compact">Compact finite-width grating</option><option value="compact-narrow">Narrow finite-width control</option><option value="compact-radiation">Compact guided launch and upward radiation</option><option value="tidy3d">Tidy3D finite-width grating</option><option value="lumerical">Lumerical curved grating and fiber</option></>}
        </select></label>}
        {key==='ring-3d' && <label>Ring boundary sampling<select value={ringInterface} onChange={e=>{setRingInterface(e.target.value);if(e.target.value==='wall-tensor' && mesh==='12')setMesh('16');}}>
          <option value="legacy">6 × 6 area samples (baseline)</option><option value="exact-area">Exact lateral area</option><option value="wall-tensor">Curved wall tensor · constant materials</option>
        </select></label>}
        {dispersiveRing && <label>Ring material response<select value={ringMaterial} onChange={e=>setRingMaterial(e.target.value)}>
          <option value="constant">Constant index</option><option value="palik-ade">Palik dispersion control · 1500–1600 nm</option>
        </select></label>}
        {key==='grating-3d' && geometry3d==='lumerical' && <label>Grating material sampling<select value={gratingMaterial} onChange={e=>{setGratingMaterial(e.target.value);setGratingDomain('full');setMesh('');}}>
          <option value="volume-average">Volume averaging (baseline)</option><option value="direct-yee">Direct electric-field cell integration</option>
        </select></label>}
        {key==='grating-3d' && Boolean(selected.grating_meshes?.[geometry3d]?.['mirror-y']?.[gratingMaterial]?.length) && <label>Simulation domain
          <select value={gratingDomain} onChange={e=>{setGratingDomain(e.target.value);setGratingSpectrum('standard');setMesh('');}}>
            <option value="full">Full width</option><option value="mirror-y">Half width · checked against the full 12-cell model</option>
          </select></label>}
        {denseGrating && <label>Grating wavelengths<select value={gratingSpectrum} onChange={e=>setGratingSpectrum(e.target.value)}>
          <option value="standard">5 samples · 25 nm spacing</option><option value="dense">21 samples · 5 nm spacing</option>
        </select></label>}
        {key==='ring-3d' && <label>Observation duration<select value={ringDuration} onChange={e=>setRingDuration(e.target.value)}>
          <option value="">Preset duration</option>{[160,1600,2700,3200,4800,6000].map(t=><option value={t} key={t}>{t} µm/c</option>)}
        </select></label>}
        {key==='multimode-tm' && <label>Port distance from joint<select value={portDistance} onChange={e=>setPortDistance(e.target.value)}>
          {[.75,1,1.25,1.5,1.75].map(d=><option key={d} value={String(d)}>{d.toFixed(2)} µm</option>)}
        </select></label>}{key==='ring' && <label>Measurement<select aria-label="Measurement" value={ringScope} onChange={e=>setRingScope(e.target.value)}>
        <option value="lumerical">Lumerical loaded ring spectrum</option><option value="tidy3d">Tidy3D loaded ring spectrum</option>
        <option value="full">Both loaded rings, finite-height poles and all controls</option></select></label>}
      {key==='grating'  && <label>Measurement<select aria-label="Measurement" value={gratingScope}
        onChange={e=>{setGratingScope(e.target.value);setMesh('');}}>
        <option value="tidy3d-compact">Tidy3D compact beam coupling</option>
        <option value="lumerical-article">Lumerical article beam coupling</option>
        <option value="lumerical-project">Lumerical project source</option>
        <option value="lumerical-output">Lumerical project output aperture</option>
        <option value="tidy3d-radiation">Tidy3D grating radiated power</option>
        <option value="lumerical-radiation">Lumerical grating radiated power</option>
        <option value="full">Both gratings and all sensitivity controls</option>
      </select></label>}{meshes.length>0 && <label>Mesh resolution<select value={mesh} onChange={e=>setMesh(e.target.value)}>
        <option value="">Preset mesh</option>{meshes.map(v=><option value={v} key={v}>{v} cells/µm</option>)}</select></label>}
      {key==='multimode' && mesh==='32' && Number(offset)===.125 && <p>The recorded 32-cell mesh exceeded the complex-amplitude limit at this offset. Select 48 or 64 cells/µm for the qualified reference study.</p>}
      {key==='coupler' && target.startsWith('fdtdx:') && mesh==='12' && <p>The 125 nm rail gap spans 1.5 cells at this mesh. Cross-port power is sensitive to its sampled width.</p>}
      {key.startsWith('multimode') && <label className="cal-offset">Lateral offset (µm)<input aria-label="Lateral offset (µm)" type="number" min="0" max="0.25" step=".025" value={offset} onChange={e=>setOffset(e.target.value)}/></label>}
      <fieldset disabled={pending}><legend>Execution target</legend>{!targets.length && <small>Checking local solver availability…</small>}{selected.routes.map(route=>{
        const info=targets.find(t=>t.solver+':'+t.device===route);
        return <label key={route}><input type="radio" name="cal-target" value={route} checked={target===route}
          onChange={()=>{setTarget(route);if(route!=='fdtdx:gpu' && materialGrid==='polygon-area')setMaterialGrid('yee-aligned');if(mesh==='12' && !route.startsWith('fdtdx:'))setMesh('');}}/>{route.replace(':',' · ').toUpperCase()}
          {info && !info.available && <small>{info.reason}</small>}</label>;
      })}</fieldset>
      {solver==='fdtdx' && !gpuComponent && !editableLayout && <label>Material sampling<select value={materialGrid} onChange={e=>setMaterialGrid(e.target.value)}>
        <option value="yee-aligned">Electric-field-aligned grid</option><option value="legacy-x">Original grid</option>{device==='gpu' && <option value="polygon-area">Integrated polygon area at each electric-field site</option>}
      </select></label>}
      <button className="button primary compact" disabled={!available||pending||busy||!offsetValid||!layoutValid} onClick={()=>void run()}>Run calibrated study</button>
      <CodeExport kind="calibration" value={configuration} disabled={!target||!offsetValid||!layoutValid}/>
      {busy&&<p role="status">A local calibration is running.</p>}</section>
      <section><h3>Local runs</h3>{history.length>0?<label>Saved run<select value={jobId} onChange={e=>setJobId(e.target.value)}>
        <option value="">Select a run</option>{history.map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)} · {j.config.solver}/{j.config.device} · {j.status}</option>)}</select></label>:<p>No saved runs for this preset.</p>}
      {job && <><p role="status">Run {job.id.slice(0,8)} · {job.status}{job.attempt && job.attempt>1 && ' · attempt '+job.attempt}</p>
        {(job.status==='queued'||job.status==='running')&&<button className="button secondary compact" onClick={()=>void cancel()}>Cancel run</button>}
        {job.result?.actual_device && <p>Measured execution target {job.result.actual_device}{job.result.device_name && ` · ${job.result.device_name}`}</p>}
        {log && <details><summary>Solver progress</summary><pre>{log}</pre></details>}
        {job.can_resume && <button className="button secondary compact" disabled={pending||busy} onClick={()=>void resume()}>Resume saved study</button>}
        {job.error && <p role="alert">{job.error}</p>}
        {job.status==='failed' && job.config.case==='ring-3d' && measuredDiagnostic && <p><a href={'/api/calibrations/jobs/'+job.id+'/export'}>Export diagnostic spectrum, validity mask and replay</a></p>}
        {job.status==='completed'&&<p><a href={'/api/calibrations/jobs/'+job.id+'/export'}>Export Python, notebook and measured arrays</a></p>}
        {report!=null&&<CalibrationMeasurement key={'measurement'+job.id} value={report}/>}
        {report!=null && <SpatialComparison key={'compare'+job.id} value={report}/>}
        {job.config.device==='gpu' && job.status==='completed' && <GpuPerformance key={job.id} preset={key} job={job.id}/>}
        {localField && <CalibrationField key={'field'+job.id} field={localField}/>}
        {report!=null&&<details><summary>Numerical report</summary><pre>{JSON.stringify(report,null,2)}</pre></details>}</>}
      </section></>}
      {error && <p role="alert">{error}</p>}
    </main></div></section>;
}
