import { useEffect, useState } from 'react';
import { request } from './api';
import { CodeExport } from './CodeExport';
import { CalibrationMeasurement, Spectrum, type Series } from './CalibrationMeasurement';
import './calibratedComponents.css';

type Evidence = { summary: string; dimension: string; routes: string[]; wavelengths_um: number[]; series: Series[];
  limits: string; recorded?: {label:string; receipt:string;report:unknown}[]; comparisons: {vendor: string; description: string; url: string}[] };
type Preset = { case: string; name: string; scope: string; routes: string[]; evidence?: Evidence };
type Job = { id: string; status: string; created_at: string; error?: string; config: {case: string; solver: string; device: string} };
type Target = {solver: string; device: string; available: boolean; reason?: string};

export function CalibratedComponents({active}:{active:boolean}) {
  const [catalog,setCatalog]=useState<Preset[]>([]), [key,setKey]=useState('coupler'), [target,setTarget]=useState('');
  const [record,setRecord]=useState(0);
  const [materialGrid,setMaterialGrid]=useState('legacy-x');
  const [mesh,setMesh]=useState(''),[offset,setOffset]=useState('.125');
  const [targets,setTargets]=useState<Target[]>([]),[jobs,setJobs]=useState<Job[]>([]),[jobId,setJobId]=useState('');
  const [error,setError]=useState(''),[pending,setPending]=useState(false),[report,setReport]=useState<unknown>(null);
  const selected=catalog.find(p=>p.case===key), evidence=selected?.evidence;
  const job=jobs.find(j=>j.id===jobId), busy=jobs.some(j=>j.status==='running'||j.status==='queued');
  const history=jobs.filter(j=>j.config.case===key);
  const meshes = target.startsWith('fdtdx:') ? [12,16,20,24] : key==='multimode' ? [32,48,64] : ['multimode-3d','coupler','splitter'].includes(key)?[16,20,24]:[];
  const offsetValid = !key.startsWith('multimode') || (offset.trim()!=='' && Number.isFinite(Number(offset)) && +offset>=0 && +offset<=.25);
  useEffect(()=>{
    if(!active)return;
    let live=true;
    Promise.all([request<Preset[]>('/api/calibrations'),request<{targets:Target[]}>('/api/capabilities')])
      .then(([c,t])=>{if(live){setCatalog(c);setTargets(t.targets);}}).catch(e=>{if(live)setError(e.message);});
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
  const [solver,device]=target.split(':');
  const available=targets.some(t=>t.solver===solver && t.device===device && t.available);
  const configuration={case:key,solver,device,resolution:mesh?Number(mesh):null,
    offset_um:key.startsWith('multimode')?Number(offset):null,material_model:solver==='fdtdx'?materialGrid:null};
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
  function choose(next:string){setKey(next);setRecord(0);setMesh('');setOffset('.125');setTarget('');setMaterialGrid('legacy-x');setJobId('');setError('');setReport(null);}
  return <section className="cal-lab" aria-label="Calibrated components">
    <header><h1>Calibrated components</h1><p>Inspect measured reference results or run the same physical study on this machine.</p></header>
    <div className="cal-columns"><aside aria-label="Calibrated presets">{catalog.map(p=><button key={p.case} aria-pressed={key===p.case} onClick={()=>choose(p.case)}>{p.name}</button>)}</aside>
    <main>{selected && <><h2>{selected.name}</h2><p>{selected.scope}</p>
      {evidence && <section aria-label="Recorded evidence"><h3>Recorded measurements</h3><p>{evidence.summary}</p>
        <p>{evidence.dimension} · {evidence.routes.join(' / ')}</p>

        {evidence.recorded?.length ? <><label>Measured target<select value={record} onChange={e=>setRecord(+e.target.value)}>
          {evidence.recorded.map((r,i)=><option key={r.label} value={i}>{r.label}</option>)}</select></label>
          <CalibrationMeasurement key={key+record} value={evidence.recorded[record]?.report}/>
          <a href={'https://github.com/srinjoyganguly/photonics-workbench/blob/main/docs/verification/'+evidence.recorded[record]?.receipt}
           target="_blank" rel="noreferrer">Full verification record</a></> :
          <Spectrum key={key} wavelengths={evidence.wavelengths_um} series={evidence.series}/>}
        <h3>Commercial comparisons</h3>{evidence.comparisons.map(c=><p key={c.vendor}>{c.description} <a href={c.url} target="_blank" rel="noreferrer">{c.vendor} example</a></p>)}
        <p>{evidence.limits}</p></section>}
      <section><h3>Run locally</h3>{meshes.length>0 && <label>Mesh resolution<select value={mesh} onChange={e=>setMesh(e.target.value)}>
        <option value="">Preset mesh</option>{meshes.map(v=><option value={v} key={v}>{v} cells/µm</option>)}</select></label>}
      {key.startsWith('multimode') && <label className="cal-offset">Lateral offset (µm)<input aria-label="Lateral offset (µm)" type="number" min="0" max="0.25" step=".025" value={offset} onChange={e=>setOffset(e.target.value)}/></label>}
      <fieldset disabled={pending}><legend>Execution target</legend>{selected.routes.map(route=>{
        const info=targets.find(t=>t.solver+':'+t.device===route);
        return <label key={route}><input type="radio" name="cal-target" value={route} checked={target===route}
          onChange={()=>{setTarget(route);if(mesh==='12' && !route.startsWith('fdtdx:'))setMesh('');}}/>{route.replace(':',' · ').toUpperCase()}
          {info && !info.available && <small>{info.reason}</small>}</label>;
      })}</fieldset>
      <button className="button primary compact" disabled={!available||pending||busy||!offsetValid} onClick={()=>void run()}>Run calibrated study</button>
      <CodeExport kind="calibration" value={configuration} disabled={!target||!offsetValid}/>
      {busy&&<p role="status">A local calibration is running.</p>}</section>
      <section><h3>Local runs</h3>{history.length>0?<label>Saved run<select value={jobId} onChange={e=>setJobId(e.target.value)}>
        <option value="">Select a run</option>{history.map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)} · {j.config.solver}/{j.config.device} · {j.status}</option>)}</select></label>:<p>No saved runs for this preset.</p>}
      {job && <><p role="status">Run {job.id.slice(0,8)} · {job.status}</p>
        {(job.status==='queued'||job.status==='running')&&<button className="button secondary compact" onClick={()=>void cancel()}>Cancel run</button>}
        {job.error && <p role="alert">{job.error}</p>}
        {job.status==='completed'&&<p><a href={'/api/calibrations/jobs/'+job.id+'/export'}>Export Python, notebook and measured arrays</a></p>}
        {report!=null&&<CalibrationMeasurement key={job.id} value={report}/>}
        {report!=null&&<details><summary>Numerical report</summary><pre>{JSON.stringify(report,null,2)}</pre></details>}</>}
      </section></>}
      {error && <p role="alert">{error}</p>}
    </main></div></section>;
}
