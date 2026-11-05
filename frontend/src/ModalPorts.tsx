import { useEffect, useRef, useState } from 'react';
import { request } from './api';
import { GeometryViewport } from './GeometryViewport';
import type { GeometryDocument } from './geometryModel';
import { VolumeSlices } from './Study3DPanel';
import { RunPassport } from './RunPassport';
import { CodeExport } from './CodeExport';
import './modalPorts.css';

type Settings = { device_kind:'straight'|'gap'; gap_um:number; cladding_index:number; wavelength_min_um:number; wavelength_max_um:number; frequency_points:number; resolution:number; pml_um:number; max_after_sources_fs:number; decay_by:number; spectrum_tolerance:number };
type CommonStudy = Omit<Settings,'gap_um'|'device_kind'> & {geometry:GeometryDocument;solver:'meep'|'fdtdx';device:'cpu'|'gpu'};
type Study = CommonStudy & (
  {schema_version:4; study_type:'waveguide-ports-3d'; device_kind:'straight'|'gap'; gap_um:number} |
  {schema_version:7; study_type:'taper-ports-3d'}
);
function modalBase(type:Study['study_type']) { return type==='taper-ports-3d'?'/api/taper-ports':'/api/ports3d'; }
function isModal(type:unknown) { return type==='taper-ports-3d'||type==='waveguide-ports-3d'; }
type Job = {id:string;status:string;config:Study;logs?:string;error?:string;result?:Record<string,unknown>;port_progress?:{stage:string;message:string}};
type Target = {solver:Study['solver'];device:Study['device'];available:boolean;device_name?:string;reason?:string;preparation?:string};
type Validation = {valid:boolean;config:Study;estimate:{grid:number[];planning_memory_mib:number;note:string};layout:{port_1_x_um:number;port_2_x_um:number;width_in?:number;width_out?:number};warnings:string[]};
type Entry = 'S11'|'S21'|'S12'|'S22';
type Metric = 'power'|'magnitude'|'phase_unwrapped_rad'|'real'|'imag';
type Parameter = {real:(number|null)[];imag:(number|null)[];magnitude:(number|null)[];power:(number|null)[];phase_rad:(number|null)[];phase_unwrapped_rad:(number|null)[]};
type Spectrum = {wavelength_um:number[];valid_bin:boolean[];parameters:Record<Entry,Parameter>;reciprocity_abs:(number|null)[];mode_neff_real?:number[]; mode_neff_ports_real?:[number|null,number|null][];metadata:{actual_device:string;device_name?:string;ports:Record<string,unknown>;settlement:Record<string,{settled:boolean;energy_ratio:number|null;spectral_relative_change:number|null}>;[key:string]:unknown}};
const initial:Settings={device_kind:'gap',gap_um:.25,cladding_index:1,wavelength_min_um:1.45,wavelength_max_um:1.65,frequency_points:5,resolution:16,pml_um:.75,max_after_sources_fs:600,decay_by:1e-7,spectrum_tolerance:1e-4};
const entries:Entry[]=['S11','S21','S12','S22'];
const colors=['#b4a4f5','#74debf','#66b8ff','#f4bc7d'];
const stageIds=['reference_forward','device_forward','reference_reverse','device_reverse'];
const stageNames=['Reference · port 1','Device · port 1','Reference · port 2','Device · port 2'];
const canonical=(value:unknown)=>JSON.stringify(value,(_key,item)=>item&&typeof item==='object'&&!Array.isArray(item)?Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])):item);
const fmt=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toPrecision(6):'Undefined';
function saveJson(value:unknown,name:string){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}

type MeshRun = { id:string; solver:Study['solver']; device:Study['device']; resolution:number; name:string };
type MeshEntry = { maximum_absolute_complex_change:number; maximum_absolute_power_change:number; maximum_absolute_phase_change_rad:number|null };
type MeshReport = { coarse_run_id:string; fine_run_id:string; device_name:string; complex_matrix_relative_l2:number|null;
  compared_bins:number; wavelength_um:number[]; jointly_valid_bins:boolean[]; reference_planes_um:number[]; entries:Record<Entry,MeshEntry> };

function TaperMeshComparison({currentId, recentKey}:{currentId?:string; recentKey:string}) {
  const [expanded,setExpanded]=useState(false);
  const [runs,setRuns]=useState<MeshRun[]>([]);
  const [coarse,setCoarse]=useState(''),[fine,setFine]=useState('');
  const [result,setResult]=useState<MeshReport|null>(null),[problem,setProblem]=useState('');
  const [loading,setLoading]=useState(false),[comparing,setComparing]=useState(false),[revision,setRevision]=useState(0);
  const sequence=useRef(0);
  useEffect(()=>{
    if(!expanded)return;
    let live=true;
    setLoading(true);setProblem('');
    request<MeshRun[]>('/api/taper-ports/mesh-runs').then(values=>{
      if(!live)return;
      setRuns(values);
      setCoarse(old=>values.some(item=>item.id===old)?old:values.find(item=>item.id===currentId)?.id||'');
      setFine(old=>values.some(item=>item.id===old)?old:'');
    }).catch(e=>{if(live)setProblem(e.message);}).finally(()=>{if(live)setLoading(false);});
    return ()=>{live=false;};
  },[expanded,currentId,recentKey,revision]);
  useEffect(()=>()=>{sequence.current++;},[]);
  const coarseRun=runs.find(run=>run.id===coarse);
  const candidates=runs.filter(run=>coarseRun && run.solver===coarseRun.solver && run.device===coarseRun.device && run.resolution>coarseRun.resolution);
  const label=(run:MeshRun)=>`${run.name} · ${run.resolution} cells/µm · ${run.solver} ${run.device.toUpperCase()} · ${run.id.slice(0,8)}`;
  function resetReport() {
    sequence.current++;setResult(null);setProblem('');setComparing(false);
  }
  async function compare() {
    const order=++sequence.current;
    setComparing(true);setResult(null);setProblem('');
    try {
      const report=await request<MeshReport>('/api/taper-ports/compare-mesh',{coarse_run_id:coarse,fine_run_id:fine});
      if(order===sequence.current)setResult(report);
    } catch(e) {
      if(order===sequence.current)setProblem((e as Error).message);
    } finally {
      if(order===sequence.current)setComparing(false);
    }
  }
  return <details className="modal-mesh" onToggle={event=>setExpanded(event.currentTarget.open)}>
    <summary>Taper mesh sensitivity</summary>
    <p>Choose two completed taper runs with the same study and execution target. Increase only the grid resolution.</p>
    <div className="modal-mesh-selectors">
      <label>Coarse mesh<select aria-label="Coarse taper mesh run" value={coarse} disabled={loading}
        onChange={event=>{resetReport();setCoarse(event.target.value);setFine('');}}>
        <option value="">{loading?'Loading saved runs…':'Select a saved taper run…'}</option>
        {runs.map(run=><option value={run.id} key={run.id}>{label(run)}</option>)}</select></label>
      <label>Fine mesh<select aria-label="Fine taper mesh run" value={fine} disabled={!coarseRun||loading}
        onChange={event=>{resetReport();setFine(event.target.value);}}>
        <option value="">{coarseRun && !candidates.length?'No finer run on this target':'Select a finer run…'}</option>
        {candidates.map(run=><option value={run.id} key={run.id}>{label(run)}</option>)}</select></label>
    </div>
    <div className="modal-actions"><button className="button secondary compact" disabled={loading||comparing||!candidates.some(run=>run.id===fine)}
      onClick={()=>void compare()}>{comparing?'Comparing meshes…':'Compare taper meshes'}</button>
      <button className="button secondary compact" disabled={loading} onClick={()=>{resetReport();setRevision(n=>n+1);}}>Refresh taper runs</button></div>
    {problem&&<p className="error-banner" role="alert">{problem}</p>}
    {result&&<div className="modal-mesh-report" role="region" aria-label="Taper mesh comparison">
      <p>{result.compared_bins} of {result.wavelength_um.length} common wavelengths qualified · {result.device_name} · Reference planes {result.reference_planes_um.map(value=>value.toFixed(3)).join(' and ')} µm</p>
      <div className="modal-table"><table><caption>Maximum absolute changes</caption><thead><tr>
        <th>Entry</th><th>Complex amplitude</th><th>Modal power</th><th>Phase (rad)</th>
      </tr></thead><tbody>{entries.map(entry=><tr key={entry} data-mesh-entry={entry}><th scope="row">{entry}</th>
        <td>{fmt(result.entries[entry].maximum_absolute_complex_change)}</td><td>{fmt(result.entries[entry].maximum_absolute_power_change)}</td>
        <td>{fmt(result.entries[entry].maximum_absolute_phase_change_rad)}</td></tr>)}</tbody></table></div>
      <p>Complex matrix relative L2 {fmt(result.complex_matrix_relative_l2)}. Two meshes measure sensitivity. More meshes and independent boundary checks are needed to estimate absolute error.</p>
      <button className="button secondary compact" onClick={()=>saveJson(result,`taper-mesh-${result.coarse_run_id.slice(0,8)}-${result.fine_run_id.slice(0,8)}.json`)}>Download mesh comparison JSON</button>
    </div>}
  </details>;
}

function MatrixResults({job}:{job:Job}){
  const [data,setData]=useState<Spectrum|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0),[index,setIndex]=useState(0),[metric,setMetric]=useState<Metric>('power');
  useEffect(()=>{let active=true;setData(null);setError('');request<Spectrum>(`/api/jobs/${job.id}/sparameters`).then(value=>{
    if(!active)return;
    const n=value.wavelength_um.length;
    if(n<3||value.valid_bin.length!==n||entries.some(e=>['real','imag','magnitude','power','phase_rad','phase_unwrapped_rad'].some(k=>value.parameters[e][k as keyof Parameter].length!==n)))throw new Error('Saved complex spectrum has inconsistent dimensions.');
    if(job.config.study_type==='taper-ports-3d' && (value.mode_neff_ports_real?.length!==n || value.mode_neff_ports_real.some(row=>!Array.isArray(row)||row.length!==2))) throw new Error('The saved taper spectrum is missing its two port-mode indices.');
    setData(value);setIndex(Math.floor(n/2));
  }).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[job.id,retry]);
  if(error)return <div className="frame-error" role="alert">S-parameters unavailable: {error}<button onClick={()=>setRetry(v=>v+1)}>Retry matrix</button></div>;
  if(!data)return <p role="status">Reading measured complex amplitudes…</p>;
  const xs=data.wavelength_um, all=entries.flatMap(e=>data.parameters[e][metric]).filter((v):v is number=>v!==null&&Number.isFinite(v));
  const lower=Math.min(0,...all),upper=Math.max(metric==='power'||metric==='magnitude'?1:0,...all),pad=Math.max(.05,(upper-lower)*.07),low=lower-pad,high=upper+pad;
  const x=(v:number)=>65+(v-xs[0])/(xs.at(-1)!-xs[0])*650,y=(v:number)=>20+(high-v)/(high-low)*240;
  const path=(values:(number|null)[])=>{let previous=false;return values.map((v,i)=>{const okay=v!==null&&Number.isFinite(v);const point=okay?`${previous?'L':'M'}${x(xs[i])},${y(v!)}`:'';previous=okay;return point;}).join(' ');};
  return <div className="modal-results" data-run-id={job.id}>
    <h3>Measured complex S-matrix</h3><p>{data.valid_bin.filter(Boolean).length} / {xs.length} qualified samples · {data.metadata.device_name||data.metadata.actual_device} · {data.metadata.actual_device}</p>
    <p>Physical reference planes: port 1 at {String(data.metadata.ports.port_1_x_um)} µm, port 2 at {String(data.metadata.ports.port_2_x_um)} µm. Rows are outgoing ports; columns are excited ports.</p>
    <label className="modal-metric">Plot quantity<select aria-label="S-parameter plot quantity" value={metric} onChange={e=>setMetric(e.target.value as Metric)}><option value="power">Modal power |S|²</option><option value="magnitude">Amplitude |S|</option><option value="phase_unwrapped_rad">Unwrapped phase (rad)</option><option value="real">Real part</option><option value="imag">Imaginary part</option></select></label>
    <div className="modal-legend">{entries.map((e,i)=><span key={e} style={{color:colors[i]}}>{e} · {e==='S21'?'1 → 2':e==='S12'?'2 → 1':'reflection'}</span>)}</div>
    <svg className="modal-chart" viewBox="0 0 750 310" role="img" aria-label="Actual complex S-parameter samples, without amplitude or phase fitting" data-metric={metric}>
      <rect x="65" y="20" width="650" height="240" fill="#0c1823" stroke="#345165"/>
      {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="65" x2="715" y1={y(low+t*(high-low))} y2={y(low+t*(high-low))} stroke="#2b4153"/><text x="56" y={y(low+t*(high-low))+4} textAnchor="end">{(low+t*(high-low)).toPrecision(3)}</text><text x={65+t*650} y="283" textAnchor="middle">{(xs[0]+t*(xs.at(-1)!-xs[0])).toFixed(3)}</text></g>)}
      {entries.map((e,k)=><g key={e} data-entry={e}><path d={path(data.parameters[e][metric])} fill="none" stroke={colors[k]} strokeWidth="2" strokeDasharray={k>1?'5 4':undefined}/>{data.parameters[e][metric].map((v,i)=>v!==null&&Number.isFinite(v)?<circle key={i} cx={x(xs[i])} cy={y(v)} r={i===index?4:2.5} fill={colors[k]} data-bin={i} data-value={v}/>:null)}</g>)}
      <line x1={x(xs[index])} x2={x(xs[index])} y1="20" y2="260" stroke="#91a2af" strokeDasharray="3 5"/><text x="390" y="305" textAnchor="middle">Wavelength (µm)</text>
    </svg>
    <p className="modal-note">Dots are actual frequency samples. Lines join adjacent valid samples; no smoothing or clipping. Near-zero amplitudes have undefined phase. Unwrapping uses only contiguous valid samples.</p>
    <label className="modal-sample">Inspect sample <strong>{index+1} / {xs.length} · {xs[index].toFixed(6)} µm</strong><input aria-label="Modal spectral sample" type="range" min="0" max={xs.length-1} value={index} onChange={e=>setIndex(Number(e.target.value))}/></label>
    <div className="modal-matrix">{entries.map((e,k)=><div key={e} data-matrix-entry={e}><h4 style={{color:colors[k]}}>{e}</h4><code data-complex="real">Re {fmt(data.parameters[e].real[index])}</code><code data-complex="imag">Im {fmt(data.parameters[e].imag[index])}</code><span>|S|² {fmt(data.parameters[e].power[index])}</span><span>Phase {fmt(data.parameters[e].phase_rad[index])} rad</span></div>)}</div>
    <p>{job.config.study_type==='taper-ports-3d' ? <>
      <span data-neff-port="1">Port 1 effective index {fmt(data.mode_neff_ports_real?.[index]?.[0])}</span> · <span data-neff-port="2">Port 2 effective index {fmt(data.mode_neff_ports_real?.[index]?.[1])}</span></>
      : <>Effective index {fmt(data.mode_neff_real?.[index])}</>} · reciprocity discrepancy |S21−S12| {fmt(data.reciprocity_abs[index])}.</p>
    <div className="modal-settlement">{stageIds.map((stage,i)=>{const r=data.metadata.settlement[stage];return <div key={stage}><strong>{stageNames[i]}</strong><span>{r?.settled?'Settling criteria met':'Unsettled'}</span><small>Energy ratio {fmt(r?.energy_ratio)}<br/>Complex spectrum change {fmt(r?.spectral_relative_change)}</small></div>;})}</div>
    <details><summary>All complex samples</summary><div className="modal-table"><table><thead><tr><th>λ (µm)</th>{entries.map(e=><th key={e}>{e} · real + imaginary i</th>)}<th>Valid</th></tr></thead><tbody>{xs.map((w,i)=><tr key={w}><th>{w.toFixed(6)}</th>{entries.map(e=><td key={e}>{fmt(data.parameters[e].real[i])} + ({fmt(data.parameters[e].imag[i])})i</td>)}<td>{data.valid_bin[i]?'Yes':'No'}</td></tr>)}</tbody></table></div></details>
    <div className="modal-actions"><a className="button secondary compact" href={`/api/jobs/${job.id}/sparameters.csv`}>Export complex CSV</a><button className="button secondary compact" onClick={()=>saveJson(data,`sparameters-${job.id.slice(0,8)}.json`)}>Export complex JSON</button></div>
    <details><summary>Phase, modal normalization and numerical evidence</summary><pre>{JSON.stringify(data.metadata,null,2)}</pre></details>
    <p className="modal-note">This matrix contains one fundamental guided channel per port. Missing modal power includes other modes, radiation and numerical error; it is not absorption. Grid, PML and port-placement convergence remain necessary for quantitative design.</p>
  </div>;
}

function preview(geometry:GeometryDocument,gap:number):GeometryDocument{
  return {...geometry,objects:geometry.objects.flatMap(obj=>obj.id==='core'&&gap>0?[-1,1].map(sign=>({...obj,id:`core-${sign<0?'left':'right'}`,name:`${sign<0?'Input':'Output'} core`,visible:true,size:{...obj.size,x:(obj.size.x-gap)/2},position:{...obj.position,x:sign*(obj.size.x+gap)/4}})):[{...obj,visible:true}])};
}

export function ModalPorts({geometry,onOpenGeometry,active}:{geometry:GeometryDocument;onOpenGeometry:(g:GeometryDocument)=>void;active:boolean}){
  const [settings,setSettings]=useState<Settings>(initial),[target,setTarget]=useState(''),[targets,setTargets]=useState<Target[]>([]),[job,setJob]=useState<Job|null>(null),[recent,setRecent]=useState<Job[]>([]),[error,setError]=useState(''),[submitting,setSubmitting]=useState(false),[opening,setOpening]=useState(false),[checking,setChecking]=useState(false),[validation,setValidation]=useState<{key:string;value:Validation}|null>(null),[invalid,setInvalid]=useState(''),[selected,setSelected]=useState<string|null>(null);
  const sequence=useRef(0),file=useRef<HTMLInputElement>(null),started=useRef(false);
  const chosen=targets.find(t=>`${t.solver}:${t.device}`===target);
  const taper=geometry.schema_version===2;
  const {device_kind,gap_um,...common}=settings;
  const shared={geometry,...common,solver:chosen?.solver??'meep',device:chosen?.device??'cpu'};
  const config:Study=taper?{...shared,schema_version:7,study_type:'taper-ports-3d'}
    :{...shared,schema_version:4,study_type:'waveguide-ports-3d',device_kind,gap_um};
  const api=modalBase(config.study_type);
  useEffect(()=>setTarget(''),[taper]);
  const key=canonical(config),valid=validation?.key===key&&validation.value.valid&&!invalid,busy=submitting||opening||!!job&&['queued','running'].includes(job.status);
  useEffect(()=>()=>{sequence.current++;},[]);
  async function refreshTargets(){setChecking(true);try{const c=await request<{targets:Target[]}>('/api/ports3d/capabilities');setTargets(c.targets);}catch(e){setError((e as Error).message);}finally{setChecking(false);}}
  async function loadRecent(){try{setRecent((await request<Job[]>('/api/jobs')).filter(j=>isModal(j.config?.study_type)));}catch(e){setError(`Saved modal runs: ${(e as Error).message}`);}}
  useEffect(()=>{if(active&&!started.current){started.current=true;void refreshTargets();void loadRecent();}},[active]);
  useEffect(()=>{if(!active)return;let live=true;setInvalid('');const timer=setTimeout(()=>request<Validation>(api+'/validate',config).then(value=>{if(live)setValidation({key,value});}).catch(e=>{if(live){setValidation(null);setInvalid(e.message);}}),180);return()=>{live=false;clearTimeout(timer);};},[key,active]);
  useEffect(()=>{if(!job||!['queued','running'].includes(job.status))return;let live=true;const id=job.id;const timer=setInterval(()=>request<Job>(`/api/jobs/${id}`).then(value=>{if(live){setJob(value);if(!['queued','running'].includes(value.status))void loadRecent();}}).catch(e=>{if(live)setError(e.message);}),1000);return()=>{live=false;clearInterval(timer);};},[job?.id,job?.status]);
  function applyStudy(c:Study){const {schema_version:_,study_type:__,geometry:g,solver:___,device:____,...s}=c;setSettings({...initial,...s});onOpenGeometry(g);setTarget('');setValidation(null);setInvalid('');setError('');}
  async function example(kind:'gap'|'straight'|'taper'){if(busy)return;const token=++sequence.current;setOpening(true);try{const c=await request<Study>(kind==='taper'?'/api/taper-ports/example':`/api/ports3d/example?kind=${kind}`);if(token===sequence.current)applyStudy(c);}catch(e){if(token===sequence.current)setError((e as Error).message);}finally{if(token===sequence.current)setOpening(false);}}
  async function run(){if(!valid||!chosen?.available||busy)return;setSubmitting(true);setError('');try{setJob(await request<Job>(api+'/jobs',config));}catch(e){setError((e as Error).message);}finally{setSubmitting(false);}}
  async function openRun(id:string){if(busy||!id)return;setOpening(true);const token=++sequence.current;try{const value=await request<Job>(`/api/jobs/${id}`);if(!isModal(value.config.study_type))throw new Error('This is not a modal 3D run.');if(token===sequence.current)setJob(value);}catch(e){if(token===sequence.current)setError((e as Error).message);}finally{if(token===sequence.current)setOpening(false);}}
  async function importStudy(f?:File){if(!f||busy)return;const token=++sequence.current;setOpening(true);try{if(f.size>100000)throw new Error('Study JSON must be smaller than 100 KB.');const c=JSON.parse(await f.text());if(!isModal(c?.study_type))throw new Error('Choose a saved waveguide or taper modal study.');const normalized=await request<Validation>(modalBase(c.study_type)+'/validate',c);if(token===sequence.current)applyStudy(normalized.config);}catch(e){if(token===sequence.current)setError((e as Error).message);}finally{if(token===sequence.current)setOpening(false);}}
  function parameter(k:Exclude<keyof Settings,'device_kind'>,label:string,step:number){return <label className="parameter" key={k}><span>{label}</span><input type="number" aria-label={`Modal ${label}`} value={Number.isFinite(settings[k])?settings[k]:''} step={step} disabled={busy||(k==='gap_um'&&settings.device_kind==='straight')} onChange={e=>setSettings(s=>({...s,[k]:e.target.value===''?NaN:Number(e.target.value)}))}/></label>;}
  const stage=job?.port_progress?.stage,stageIndex=stageIds.indexOf(stage||'');
  const changed=job&&canonical(job.config)!==key;
  return <section className="modal-study panel" aria-label="Finite-height 3D modal ports">
    <header className="modal-heading"><div><span className="eyebrow">GEOMETRY → COMPLEX SCATTERING</span><h2>Finite-height 3D ports</h2><p>Four directional solves. One guided mode per port. Actual complex amplitudes.</p></div><div className="modal-actions"><button className="button secondary" disabled={busy} onClick={()=>void example('straight')}>Open 3D straight validation</button><button className="button primary" disabled={busy} onClick={()=>void example('gap')}>Open 3D gap example</button><button className="button secondary" disabled={busy} onClick={()=>void example('taper')}>Open taper port example</button></div></header>
    <div className="modal-layout"><div className="modal-setup"><h3>Study settings</h3><p>{taper ? 'The taper uses its complete editor contour. Adjust its end widths in the component library. Each uniform access guide has its own port mode and reference solve.' : 'Core dimensions and optional substrate come from the 3D Inspector above. The study cuts a centred cladding-filled gap when selected.'}</p>
      {!taper && <label>Device<select aria-label="Modal device kind" value={settings.device_kind} disabled={busy} onChange={e=>setSettings(s=>({...s,device_kind:e.target.value as Settings['device_kind'],gap_um:e.target.value==='straight'?0:.25}))}><option value="gap">Waveguide gap</option><option value="straight">Straight-guide validation</option></select></label>}
      <div className="modal-parameters">{!taper && parameter('gap_um','Gap length (µm)',.0625)}{parameter('cladding_index','Cladding index',.05)}{parameter('wavelength_min_um','Minimum wavelength (µm)',.01)}{parameter('wavelength_max_um','Maximum wavelength (µm)',.01)}{parameter('frequency_points','Frequency samples',2)}</div>
      <details><summary>Numerics and settling</summary><div className="modal-parameters">{parameter('resolution','Resolution (cells/µm)',4)}{parameter('pml_um','PML thickness (µm)',.0625)}{parameter('max_after_sources_fs','After-source time cap (fs)',50)}{parameter('decay_by','Energy decay',1e-7)}{parameter('spectrum_tolerance','Complex spectrum stability',1e-4)}</div></details>
      <div className={`modal-validation ${invalid?'invalid':''}`} role="status">{invalid|| (valid?'3D modal study validated':'Checking the current 3D modal study…')}</div>
      {valid&&<><p>{validation!.value.estimate.grid.join(' × ')} cells · four solves · memory planning allowance {validation!.value.estimate.planning_memory_mib} MiB.</p><details><summary>Source, ports and model limits</summary><p>{validation!.value.estimate.note}</p>{taper && <p>Port 1 at {fmt(validation!.value.layout.port_1_x_um)} µm, width {fmt(validation!.value.layout.width_in)} µm. Port 2 at {fmt(validation!.value.layout.port_2_x_um)} µm, width {fmt(validation!.value.layout.width_out)} µm.</p>}{validation!.value.warnings.map(w=><p key={w}>{w}</p>)}</details></>}
      <div className="modal-target-title"><h3>Explicit execution target</h3><button disabled={busy||checking} onClick={()=>void refreshTargets()}>Refresh</button></div><p>Meep is a useful CPU baseline. FDTDX prepares source modes locally on CPU and evolves fields on the selected CPU or NVIDIA GPU.</p>
      <div className="modal-targets">{targets.map(t=>{const id=`${t.solver}:${t.device}`;return <label key={id}><input type="radio" name="modal-target" value={id} checked={target===id} disabled={!t.available||busy} onChange={()=>setTarget(id)}/><strong>{t.solver==='meep'?'Meep':'FDTDX'} · {t.device==='gpu'?'NVIDIA GPU':'CPU'}</strong><span>{t.available?t.device_name:t.reason}</span></label>;})}{checking&&<p>Checking local runtimes…</p>}</div>
      <div className="modal-actions"><button className="button primary" disabled={!valid||!chosen?.available||busy} onClick={()=>void run()}>{busy?'Study in progress…':'Run 3D S-matrix'}</button>{job&&['queued','running'].includes(job.status)&&<button className="button secondary" onClick={()=>request<Job>(`/api/jobs/${job.id}/cancel`,{}).then(setJob).catch(e=>setError(e.message))}>Cancel modal run</button>}</div>
      <div className="modal-actions"><button className="button secondary compact" onClick={()=>saveJson(config,taper?'taper-ports-3d-study.json':'modal-3d-study.json')} disabled={!valid||!chosen?.available||busy}>Export modal study JSON</button><button className="button secondary compact" onClick={()=>file.current?.click()} disabled={busy}>Import modal study JSON</button><input ref={file} hidden type="file" accept=".json,application/json" aria-label="Import 3D modal study" onChange={e=>void importStudy(e.target.files?.[0])}/><CodeExport value={config} disabled={!valid||!chosen?.available||busy}/></div>
    </div><div className="modal-output">
      {valid&&<details className="modal-preview"><summary>{taper?'Interactive editor taper':'Interactive study geometry · gap applied'}</summary><div className="modal-preview-canvas"><GeometryViewport document={taper?preview(geometry,0):preview(geometry,settings.gap_um)} selectedId={selected} onSelect={setSelected}/></div><p>{taper?'The taper joins straight input and output guides in the solver cell. ':''}Actual discretized material and field slices appear after execution.</p></details>}
      <label>Saved 3D modal run<select aria-label="Saved 3D modal run" value={job?.id||''} disabled={busy} onChange={e=>void openRun(e.target.value)}><option value="">Choose a saved measurement…</option>{[...(job&&!recent.some(j=>j.id===job.id)?[job]:[]),...recent].map(j=><option key={j.id} value={j.id}>{j.id.slice(0,8)} · {j.config.study_type==='taper-ports-3d'?'taper':j.config.device_kind} · {j.config.solver} {j.config.device} · {j.status}</option>)}</select></label><button onClick={()=>void loadRecent()}>Refresh saved modal runs</button>
      <TaperMeshComparison currentId={job?.config.study_type==='taper-ports-3d'&&job.status==='completed'?job.id:undefined} recentKey={recent.map(item=>item.id).join(',')}/>
      <ol className="modal-progress">{stageIds.map((s,i)=><li key={s} className={job?.status==='completed'||stageIndex>i?'complete':stage===s?'active':''}>{stageNames[i]}<span>{job?.status==='completed'||stageIndex>i?'Recorded':stage===s?job?.status:'Pending'}</span></li>)}</ol>
      {changed&&<p className="snapshot-note">The editor differs from this saved run. Its complex amplitudes retain the exact submitted geometry and settings.</p>}
      {job&&<><p className="modal-run-status" data-status={job.status}>Run {job.id.slice(0,10)} · {job.status} · {job.port_progress?.message}</p>{job.status==='completed'&&<><MatrixResults key={job.id} job={job}/><details className="modal-fields"><summary>Inspect actual forward-device 3D fields</summary><VolumeSlices key={job.id} job={job}/></details><div className="modal-actions"><a className="button primary compact" href={`/api/jobs/${job.id}/export`}>Export 3D modal results .zip</a><a className="button secondary compact" href={`/api/jobs/${job.id}/reproduce`}>Export run Python + notebook</a><button className="button secondary compact" onClick={()=>saveJson(job.config,`modal-run-${job.id.slice(0,8)}.json`)}>Save modal run configuration</button></div><RunPassport key={job.id} jobId={job.id}/></>}{job.error&&<p className="error-banner" role="alert">{job.error}</p>}<details className="logs" open={job.status==='running'||job.status==='failed'}><summary>3D modal solver log</summary><pre>{job.logs||'Waiting for the solver…'}</pre></details></>}
      {!job&&<div className="modal-empty"><h3>Measure amplitude and phase.</h3><p>Open an example, inspect the finite-height geometry, then explicitly select an execution target. The reference and device are solved from both directions.</p></div>}
    </div></div>{error&&<div className="error-banner" role="alert">{error}<button onClick={()=>setError('')}>Dismiss modal error</button></div>}
  </section>;
}
