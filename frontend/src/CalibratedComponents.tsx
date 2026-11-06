import { useEffect, useState } from 'react';
import { request } from './api';
import { CodeExport } from './CodeExport';
import { CalibrationMeasurement, Spectrum, type Series } from './CalibrationMeasurement';
import './calibratedComponents.css';

type Evidence = { summary: string; dimension: string; routes: string[]; wavelengths_um: number[]; series: Series[];
  limits: string; recorded?: {label:string; receipt:string;report:unknown}[]; comparisons: {vendor: string; description: string; url: string}[] };
type Preset = { case: string; name: string; scope: string; routes: string[]; evidence?: Evidence };
type Job = { id: string; status: string; can_resume?:boolean;attempt?:number; created_at: string; error?: string; result?: {actual_device?:string;device_name?:string}; config: {case: string; solver: string; device: string} };
type Target = {solver: string; device: string; available: boolean; reason?: string};

export function CalibratedComponents({active}:{active:boolean}) {
  const [catalog,setCatalog]=useState<Preset[]>([]), [key,setKey]=useState('coupler'), [target,setTarget]=useState('');
  const [record,setRecord]=useState(0);
  const [materialGrid,setMaterialGrid]=useState('yee-aligned');
  const [gratingScope,setGratingScope]=useState('tidy3d-compact');
  const [ringScope,setRingScope]=useState('lumerical');
  const [mesh,setMesh]=useState(''),[offset,setOffset]=useState('.125');
  const [targets,setTargets]=useState<Target[]>([]),[jobs,setJobs]=useState<Job[]>([]),[jobId,setJobId]=useState('');
  const [log,setLog]=useState('');
  const [error,setError]=useState(''),[pending,setPending]=useState(false),[report,setReport]=useState<unknown>(null);
  const selected=catalog.find(p=>p.case===key), evidence=selected?.evidence;
  const job=jobs.find(j=>j.id===jobId), busy=jobs.some(j=>j.status==='running'||j.status==='queued');
  const history=jobs.filter(j=>j.config.case===key);
  const meshes = key==='grating' && gratingScope!=='full' ? [28,40,60] : target.startsWith('fdtdx:') ? [12,16,20,24] : key==='multimode' ? [32,48,64] : ['multimode-3d','coupler','splitter'].includes(key)?[16,20,24]:[];
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
    let live=true;setReport(null);
    if(job?.status==='completed'||job?.status==='failed') {
      request('/api/calibrations/jobs/'+job.id+'/report').then(data=>{if(live)setReport(data);})
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
    offset_um:key.startsWith('multimode')?Number(offset):null,material_model:solver==='fdtdx'?materialGrid:null,
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
  function choose(next:string){setKey(next);setRecord(0);setMesh('');setOffset('.125');setTarget('');setMaterialGrid('yee-aligned');setGratingScope('tidy3d-compact');setRingScope('lumerical');setJobId('');setError('');setReport(null);}
  return <section className="cal-lab" aria-label="Calibrated components">
    <header><h1>Calibrated components</h1><p>Inspect measured reference results or run the same physical study on this machine.</p></header>
    <div className="cal-columns"><aside aria-label="Calibrated presets">{catalog.map(p=><button key={p.case} aria-pressed={key===p.case} onClick={()=>choose(p.case)}>{p.name}</button>)}</aside>
    <main>{selected && <><h2>{selected.name}</h2><p>{selected.scope}</p>
      {evidence && <section aria-label="Recorded evidence"><h3>Recorded measurements</h3><p>{evidence.summary}</p>
        <p>{evidence.dimension} · {evidence.routes.join(' / ')}</p>

        {evidence.recorded?.length ? <><label>Recorded result<select value={record} onChange={e=>setRecord(+e.target.value)}>
          {evidence.recorded.map((r,i)=><option key={r.label} value={i}>{r.label}</option>)}</select></label>
          <CalibrationMeasurement key={key+record} value={evidence.recorded[record]?.report}/>
          <a href={'https://github.com/srinjoyganguly/photonics-workbench/blob/main/docs/verification/'+evidence.recorded[record]?.receipt}
           target="_blank" rel="noreferrer">Full verification record</a></> :
          <Spectrum key={key} wavelengths={evidence.wavelengths_um} series={evidence.series}/>}
        <h3>Commercial comparisons</h3>{evidence.comparisons.map(c=><p key={c.vendor}>{c.description} <a href={c.url} target="_blank" rel="noreferrer">{c.vendor} example</a></p>)}
        <p>{evidence.limits}</p></section>}
      <section><h3>Run locally</h3>{key==='ring' && <label>Measurement<select aria-label="Measurement" value={ringScope} onChange={e=>setRingScope(e.target.value)}>
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
          onChange={()=>{setTarget(route);if(mesh==='12' && !route.startsWith('fdtdx:'))setMesh('');}}/>{route.replace(':',' · ').toUpperCase()}
          {info && !info.available && <small>{info.reason}</small>}</label>;
      })}</fieldset>
      {solver==='fdtdx' && <label>Material sampling<select value={materialGrid} onChange={e=>setMaterialGrid(e.target.value)}>
        <option value="yee-aligned">Electric-field-aligned grid</option><option value="legacy-x">Original grid</option>
      </select></label>}
      <button className="button primary compact" disabled={!available||pending||busy||!offsetValid} onClick={()=>void run()}>Run calibrated study</button>
      <CodeExport kind="calibration" value={configuration} disabled={!target||!offsetValid}/>
      {busy&&<p role="status">A local calibration is running.</p>}</section>
      <section><h3>Local runs</h3>{history.length>0?<label>Saved run<select value={jobId} onChange={e=>setJobId(e.target.value)}>
        <option value="">Select a run</option>{history.map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)} · {j.config.solver}/{j.config.device} · {j.status}</option>)}</select></label>:<p>No saved runs for this preset.</p>}
      {job && <><p role="status">Run {job.id.slice(0,8)} · {job.status}{job.attempt && job.attempt>1 && ' · attempt '+job.attempt}</p>
        {(job.status==='queued'||job.status==='running')&&<button className="button secondary compact" onClick={()=>void cancel()}>Cancel run</button>}
        {job.result?.actual_device && <p>Measured execution target {job.result.actual_device}{job.result.device_name && ` · ${job.result.device_name}`}</p>}
        {log && <details><summary>Solver progress</summary><pre>{log}</pre></details>}
        {job.can_resume && <button className="button secondary compact" disabled={pending||busy} onClick={()=>void resume()}>Resume saved study</button>}
        {job.error && <p role="alert">{job.error}</p>}
        {job.status==='completed'&&<p><a href={'/api/calibrations/jobs/'+job.id+'/export'}>Export Python, notebook and measured arrays</a></p>}
        {report!=null&&<CalibrationMeasurement key={job.id} value={report}/>}
        {report!=null&&<details><summary>Numerical report</summary><pre>{JSON.stringify(report,null,2)}</pre></details>}</>}
      </section></>}
      {error && <p role="alert">{error}</p>}
    </main></div></section>;
}
