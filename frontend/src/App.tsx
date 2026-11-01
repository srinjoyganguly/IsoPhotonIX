import { useEffect, useRef, useState } from 'react';
import { FieldCanvas, type Field } from './FieldCanvas';
import { request } from './api';
import { EvolutionPlayer } from './EvolutionPlayer';
import { RunPassport } from './RunPassport';
import { GeometryEditor } from './GeometryEditor';
import { PortStudyWorkspace } from './PortStudyPanel';

type Solver = 'meep' | 'fdtdx';
type Device = 'cpu' | 'gpu';
type Config = {
  schema_version: number; wavelength_um: number; core_width_um: number; core_index: number;
  cladding_index: number; length_um: number; height_um: number; resolution: number;
  duration_fs: number; pml_um: number; solver: Solver; device: Device;
};
type Target = { solver: Solver; device: Device; available: boolean; reason?: string; version?: string; device_name?: string };
type Capabilities = { targets: Target[]; recommendation?: { solver: Solver; device: Device; reason: string } };
type Job = { id: string; status: string; config: Config; created_at: string; logs?: string | string[]; error?: string; result?: Record<string, unknown> };
type Validation = { valid: boolean; estimate?: Record<string, unknown>; warnings?: string[] };
type View = 'geometry' | 'field' | 'epsilon' | 'evolution';

const defaults: Config = { schema_version: 1, wavelength_um: 1.55, core_width_um: .6, core_index: 2, cladding_index: 1, length_um: 10, height_um: 5, resolution: 20, duration_fs: 80, pml_um: 1, solver: 'meep', device: 'cpu' };
const labels: Record<string, string> = { wavelength_um: 'Wavelength', core_width_um: 'Core width', core_index: 'Core index', cladding_index: 'Cladding index', length_um: 'Domain length', height_um: 'Domain height', resolution: 'Grid resolution', duration_fs: 'Simulated duration', pml_um: 'PML thickness' };
const name = (s: string) => s === 'meep' ? 'Meep' : 'FDTDX';

function downloadJson(config: Config) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2) + '\n'], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = 'waveguide-config.json'; a.click(); URL.revokeObjectURL(url);
}
function Symbol({kind, size = 18}: {kind: 'wave'|'play'|'download'|'grid'|'check'|'arrow'|'cpu'|'refresh'|'folder';size?: number}) {
  const paths = {
    wave: <><path d="M2 12c3-12 5-12 8 0s5 12 8 0 5-12 8 0"/><path d="M2 17c3-12 5-12 8 0s5 12 8 0" opacity=".4"/></>,
    play: <path d="m8 5 11 7-11 7Z"/>, download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/></>,
    grid: <><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></>,
    check:<path d="m5 12 4 4L19 6"/>, arrow:<path d="M4 12h16m-6-6 6 6-6 6"/>,
    cpu:<><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/></>,
    refresh:<><path d="M20 8a9 9 0 1 0 1 7M20 3v5h-5"/></>,
    folder:<path d="M3 7V4h6l3 3h9v13H3Z"/>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[kind]}</svg>;
}

function Geometry({config}: {config: Config}) {
  const w = 720, h = 330, x0 = 54, y0 = 30, dw = 612, dh = 255;
  const pmlX = dw * config.pml_um / config.length_um, pmlY = dh * config.pml_um / config.height_um;
  const coreH = dh * config.core_width_um / config.height_um;
  const sourceX = x0 + pmlX + dw * .5 / config.length_um;
  return <svg className="geometry" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Waveguide geometry: ${config.length_um} by ${config.height_um} micrometres. Core width ${config.core_width_um} micrometres. Source on the left, absorbing PML boundaries.`}>
    <defs><pattern id="grid" width="30.6" height="25.5" patternUnits="userSpaceOnUse"><path d="M30.6 0H0V25.5" fill="none" stroke="#243140" strokeWidth=".7"/></pattern><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#61778d" strokeWidth="1" opacity=".22"/></pattern></defs>
    <rect x={x0} y={y0} width={dw} height={dh} fill="#101a26" stroke="#405268"/>
    <rect x={x0} y={y0} width={dw} height={dh} fill="url(#grid)"/>
    <rect x={x0} y={y0} width={dw} height={pmlY} fill="url(#hatch)"/><rect x={x0} y={y0+dh-pmlY} width={dw} height={pmlY} fill="url(#hatch)"/>
    <rect x={x0} y={y0} width={pmlX} height={dh} fill="url(#hatch)"/><rect x={x0+dw-pmlX} y={y0} width={pmlX} height={dh} fill="url(#hatch)"/>
    <rect x={x0+pmlX} y={y0+pmlY} width={dw-pmlX*2} height={dh-pmlY*2} fill="none" stroke="#5a6b7c" strokeDasharray="4 5"/>
    <rect x={x0} y={y0+dh/2-coreH/2} width={dw} height={coreH} fill="#53d9c0" fillOpacity=".16" stroke="#53d9c0" strokeOpacity=".8"/>
    <line x1={sourceX} y1={y0+dh/2-coreH/2} x2={sourceX} y2={y0+dh/2+coreH/2} stroke="#f2b969" strokeWidth="4"/>
    <path d={`M${sourceX+8} ${y0+dh/2}h28m-6-5 6 5-6 5`} stroke="#f2b969" fill="none"/>
    <text x={sourceX+3} y={y0+dh/2+coreH/2+24} fill="#f2b969" fontSize="11">Ez source</text>
    <text x={x0+dw/2} y={y0+dh/2-coreH/2-12} fill="#7ee1ce" fontSize="12" textAnchor="middle">dielectric core · n = {config.core_index}</text>
    <text x={x0+dw/2} y={y0+pmlY+24} fill="#8b9bad" fontSize="11" textAnchor="middle">cladding · n = {config.cladding_index}</text>
    <text x={x0+dw/2} y={y0+15} fill="#a1adbb" fontSize="10" textAnchor="middle">ABSORBING BOUNDARY · PML</text>
    {[0,.25,.5,.75,1].map(t=><g key={t}><line x1={x0+dw*t} y1={y0+dh} x2={x0+dw*t} y2={y0+dh+5} stroke="#69809a"/><text x={x0+dw*t} y={y0+dh+20} fill="#8194a8" fontSize="10" textAnchor="middle">{((t-.5)*config.length_um).toFixed(1)}</text></g>)}
    <text x={x0+dw/2} y={h-4} fill="#9aabba" fontSize="11" textAnchor="middle">x (µm) · propagation</text>
    <text transform={`translate(17 ${y0+dh/2}) rotate(-90)`} fill="#9aabba" fontSize="11" textAnchor="middle">y (µm)</text>
    <text x={x0-8} y={y0+4} fill="#8194a8" fontSize="10" textAnchor="end">{(config.height_um/2).toFixed(1)}</text>
    <text x={x0-8} y={y0+dh/2+4} fill="#8194a8" fontSize="10" textAnchor="end">0</text>
    <text x={x0-8} y={y0+dh} fill="#8194a8" fontSize="10" textAnchor="end">{(-config.height_um/2).toFixed(1)}</text>
  </svg>;
}


export default function App() {
  const [workspace,setWorkspace]=useState<'design'|'simulation'|'ports'>('design');
  const [config,setConfig]=useState<Config>(defaults);
  const [geometryConfig,setGeometryConfig]=useState<Config>(defaults);
  const [loaded,setLoaded]=useState(false);
  const [capabilities,setCapabilities]=useState<Capabilities>({targets:[]});
  const [target,setTarget]=useState<string>('');
  const [capLoading,setCapLoading]=useState(true);
  const [validation,setValidation]=useState<Validation|null>(null);
  const [validationError,setValidationError]=useState('');
  const [error,setError]=useState('');
  const [job,setJob]=useState<Job|null>(null);
  const [field,setField]=useState<Field|null>(null);
  const [view,setView]=useState<View>('geometry');
  const [submitting,setSubmitting]=useState(false);
  const [notice,setNotice]=useState('');
  const fileRef=useRef<HTMLInputElement>(null);
  const selected=capabilities.targets.find(t=>`${t.solver}:${t.device}`===target);
  const busy=submitting || job?.status==='queued' || job?.status==='running';
  const runConfig={...config,...(selected ? {solver:selected.solver,device:selected.device}: {})};
  const currentDiffers=!!job && (Object.keys(defaults) as (keyof Config)[]).some(key=>runConfig[key]!==job.config[key]);
  const loadCapabilities=(refresh=false)=>{
    setCapLoading(true);
    request<Capabilities>(`/api/capabilities${refresh?'?refresh=true':''}`).then(setCapabilities).catch(e=>setError(`Device check: ${e.message}`)).finally(()=>setCapLoading(false));
  };
  useEffect(()=>{
    request<Config>('/api/example').then(value=>{setConfig(value);setLoaded(true);}).catch(e=>setError(`Cannot connect to the local API. Start the backend, then reload. ${e.message}`));
    loadCapabilities();
  },[]);
  useEffect(()=>{
    if(!loaded) return;
    let active=true; setValidation(null);setValidationError('');
    const timer=setTimeout(()=>{
      request<Validation>('/api/validate',runConfig).then(v=>{if(active){setValidation(v);if(v.valid)setGeometryConfig(runConfig);}}).catch(e=>{if(active)setValidationError(e.message);});
    },300);
    return()=>{active=false;clearTimeout(timer);};
  },[config,target,loaded]);
  useEffect(()=>{
    if(!job || !['queued','running'].includes(job.status))return;
    let active=true;
    const timer=setInterval(()=>{
      request<Job>(`/api/jobs/${job.id}`).then(value=>{if(active)setJob(value);}).catch(e=>{if(active)setError(e.message);});
    },700);
    return()=>{active=false;clearInterval(timer);};
  },[job?.id,job?.status]);
  useEffect(()=>{
    if(job?.status!=='completed')return;
    let active=true;
    request<Field>(`/api/jobs/${job.id}/field`).then(value=>{if(active){setField(value);setView('field');}}).catch(e=>{if(active)setError(`Result could not be loaded: ${e.message}`);});
    return()=>{active=false;};
  },[job?.id,job?.status]);
  async function startRun(){
    if(!selected?.available || busy)return;
    setSubmitting(true);setError('');setNotice('');
    try{
      await request<Validation>('/api/validate',runConfig);
      const created=await request<Job>('/api/jobs',runConfig);
      setJob(created);setField(null);setView('geometry');
    }catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  async function cancel(){
    if(!job)return;
    try{setJob(await request<Job>(`/api/jobs/${job.id}/cancel`,{}));}catch(e){setError((e as Error).message);}
  }
  async function openExample(){
    try {setConfig(await request<Config>('/api/example'));setView('geometry');setNotice('Bundled waveguide opened. Choose a compute target to run.');setTarget('');}catch(e){setError((e as Error).message);}
  }
  async function importConfig(file?: File){
    if(!file)return;
    try{
      if(file.size>100000)throw new Error('Configuration must be smaller than 100 KB.');
      const parsed=JSON.parse(await file.text());
      if(!parsed || typeof parsed!=='object' || Array.isArray(parsed))throw new Error('Expected a configuration JSON object.');
      const value={...defaults,...parsed} as Config;
      await request<Validation>('/api/validate',value);
      const normalized=Object.fromEntries(Object.entries(value).map(([key,item])=>[key,typeof defaults[key as keyof Config]==='number'?Number(item):item])) as Config;
      setConfig(normalized);setTarget('');setView('geometry');setNotice('Configuration imported. Explicitly select the execution target.');setError('');
    }catch(e){setError(`Import failed: ${(e as Error).message}`);}
    if(fileRef.current)fileRef.current.value='';
  }
  function input(key:keyof Config,unit:string,step:number,min:number,max:number){
    return <label className="parameter" key={key}><span>{labels[key]}</span><div className="input-wrap"><input aria-label={labels[key]} type="number" value={Number.isFinite(Number(config[key]))?config[key]:''} step={step} min={min} max={max} onChange={e=>setConfig(c=>({...c,[key]:e.target.value===''?NaN:Number(e.target.value)}))}/><span>{unit}</span></div></label>;
  }
  const rec=capabilities.recommendation;
  const logs=Array.isArray(job?.logs)?job.logs.join('\n'):job?.logs || '';
  const resultMetadata=field?.metadata || job?.result || {};
  const executionDevice=resultMetadata.device_name || resultMetadata.actual_device;
  const executionVersions=resultMetadata.versions as Record<string,string>|undefined;
  return <div className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark"><Symbol kind="wave" size={27}/></span><div>photonics<span className="brand-sub">WORKBENCH</span></div></div><div className="header-divider"/><div className="project-heading"><span className="muted">{workspace==='design'?'DESIGN STUDIO':workspace==='ports'?'ENGINEERING STUDIES':'EXAMPLE / 01'}</span><strong>{workspace==='design'?'Local geometry workspace':workspace==='ports'?'Reference-normalized port measurements':'Dielectric slab waveguide'}</strong></div><div className="topbar-right"><span className="local-pill"><i/>Local compute</span><span className="version">v0.5 preview · experimental</span></div></header>
    <nav className="workspace-navigation" aria-label="Workspace"><button aria-pressed={workspace==='design'} onClick={()=>setWorkspace('design')}>3D design</button><button aria-pressed={workspace==='simulation'} onClick={()=>setWorkspace('simulation')}>2D simulation</button><button aria-pressed={workspace==='ports'} onClick={()=>setWorkspace('ports')}>Ports · 2D</button><span>Open geometry. Explicit hardware. Inspectable results.</span></nav>
    <div hidden={workspace!=='design'}><GeometryEditor slabConfig={geometryConfig} onSimulation={()=>setWorkspace('simulation')}/></div>
    <div hidden={workspace!=='ports'}><PortStudyWorkspace active={workspace==='ports'}/></div>
    <div hidden={workspace!=='simulation'}>
    <div className="intro-row"><div><span className="eyebrow">FROM GEOMETRY TO FIELD</span><h1>A small waveguide. Real physics.</h1><p>Real fields on your own hardware. No solver activation fees. Results you can inspect and reproduce.</p></div><button className="button secondary open-example" onClick={openExample} disabled={!loaded || !!busy}><Symbol kind="folder"/>Open bundled example</button></div>
    <main className="workbench">
      <aside className="panel model-panel">
        <div className="panel-heading"><div><span className="step">01</span><h2>Define the model</h2></div><span className="tag">µm · fs</span></div>
        <div className="model-type"><span className="model-icon"><Symbol kind="grid"/></span><div><strong>2D dielectric slab</strong><span>Ez polarization · uniform along z</span></div></div>
        <div className="parameter-group"><h3>PHYSICAL PARAMETERS</h3>{input('wavelength_um','µm',.05,.5,3)}{input('core_width_um','µm',.05,.2,3)}<div className="two-inputs">{input('core_index','n',.05,1.01,3.5)}{input('cladding_index','n',.05,1,2.5)}</div></div>
        <details className="numerics"><summary>Domain & numerics<span>⌄</span></summary><div className="two-inputs">{input('length_um','µm',1,6,20)}{input('height_um','µm',.5,3,12)}</div>{input('resolution','cells / µm',1,10,60)}{input('duration_fs','fs',10,20,250)}{input('pml_um','µm',.1,.5,2)}</details>
        <div className={`validation ${validationError?'invalid':''}`} role="status">{validationError?<><span>!</span><p>{validationError}</p></>:validation?<><Symbol kind="check" size={16}/><p>Parameters validated{validation.estimate?.cells ? <small>{Number(validation.estimate.cells).toLocaleString()} grid cells</small>:null}</p></>:<><span className="spinner tiny"/><p>{loaded?'Checking parameters…':'Connecting to local API…'}</p></>}</div>
        {!!validation?.warnings?.length&&<details className="assumptions"><summary>Model assumptions</summary>{validation.warnings.map((warning,i)=><div className="warning-note" key={i}>{warning}</div>)}</details>}
        <div className="model-actions"><button onClick={()=>downloadJson(runConfig)} disabled={!validation}><Symbol kind="download" size={14}/>Save config</button><button onClick={()=>fileRef.current?.click()}>Import JSON</button><input type="file" hidden ref={fileRef} accept=".json,application/json" onChange={e=>importConfig(e.target.files?.[0])}/></div>
      </aside>

      <section className="center-column">
        <div className="panel viewport-panel">
          <div className="viewport-top"><div className="view-tabs" role="tablist" aria-label="Viewport"><button role="tab" aria-selected={view==='geometry'} className={view==='geometry'?'active':''} onClick={()=>setView('geometry')}>Geometry</button><button role="tab" aria-selected={view==='field'} disabled={!field} className={view==='field'?'active':''} onClick={()=>setView('field')}>Ez field{field&&<i/>}</button><button role="tab" aria-selected={view==='evolution'} disabled={!field} className={view==='evolution'?'active':''} onClick={()=>setView('evolution')}>Time evolution</button><button role="tab" aria-selected={view==='epsilon'} disabled={!field} className={view==='epsilon'?'active':''} onClick={()=>setView('epsilon')}>Material grid</button></div><span className="dimension-tag">XY PLANE</span></div>
          <div className="viewport-title"><div><h2>{view==='geometry'?'Your simulation domain':view==='field'?'Electric field, Ez':view==='evolution'?'Watch the field propagate':'Discretized material'}</h2><p>{view==='geometry'?(validationError?'Last valid geometry · fix parameters before running':'Geometry preview · no numerical field shown'):view==='field'?'Actual solver output · final time snapshot':view==='evolution'?'Recorded solver frames · one fixed amplitude scale':'Relative permittivity returned by the solver'}</p></div><span className="viewport-badge">{view==='geometry'?'DESIGN':view==='evolution'?'RECORDED':'COMPUTED'}</span></div>
          <div className="viewport-content">{view==='geometry'?<Geometry config={geometryConfig}/>:field?view==='evolution'&&job?<EvolutionPlayer key={job.id} jobId={job.id} data={field} active={workspace==='simulation'}/>:<FieldCanvas data={field} view={view==='epsilon'?'epsilon':'field'}/>:null}</div>
          <div className="viewport-footer">{view==='geometry'?<><span><i className="legend-core"/>Dielectric core</span><span><i className="legend-source"/>Finite source</span><span><i className="legend-pml"/>Absorbing PML</span></>:<><span className="computed-dot"/> {name(job?.config.solver || 'meep')} · {job?.config.device.toUpperCase()} · {field?.field[0]?.length} × {field?.field.length} samples</>}</div>
        </div>
        <div className="panel results-panel">
          <div className="panel-heading"><div><span className="step">03</span><h2>Inspect & export</h2></div>{job&&<span className={`status-tag ${job.status}`}><i/>{job.status}</span>}</div>
          {job?<><div className="run-facts"><div><span>RUN</span><strong className="mono">{job.id.slice(0,10)}</strong></div><div><span>SOLVER / TARGET</span><strong>{name(job.config.solver)} / {job.config.device.toUpperCase()}</strong></div><div><span>WAVELENGTH</span><strong>{job.config.wavelength_um} µm</strong></div><div><span>CORE WIDTH</span><strong>{job.config.core_width_um} µm</strong></div></div>
            {currentDiffers&&<div className="snapshot-note">The editor has changed. This run retains the configuration shown above.</div>}
            {job.status==='completed'&&<>{!!executionDevice&&<div className="execution-proof"><Symbol kind="check" size={13}/><span>Executed on {String(executionDevice)}{executionVersions?.[job.config.solver]&&` · ${name(job.config.solver)} ${executionVersions[job.config.solver]}`}</span></div>}<div className="result-actions"><a className="button primary compact" href={`/api/jobs/${job.id}/export`}><Symbol kind="download" size={16}/>Export results .zip</a><span>Configuration · numerical arrays · provenance</span></div></>}
            {job.status==='completed'&&<RunPassport key={job.id} jobId={job.id}/>}
            {job.error&&<div className="error-banner" role="alert">{job.error}</div>}
            {!!Object.keys(resultMetadata).length&&<details className="metadata"><summary>Run metadata & solver versions</summary><pre>{JSON.stringify(resultMetadata,null,2)}</pre></details>}
          </>:<div className="empty-results"><span className="empty-symbol"><Symbol kind="wave" size={30}/></span><div><strong>Your first field starts here.</strong><p>Run the waveguide to inspect real electromagnetic data and export a reproducible result.</p></div></div>}
          <details className="logs" open={!!busy || job?.status==='failed'}><summary><span>Solver log</span><span className="log-hint">{busy?'WORKER ACTIVE':job?'PROCESS OUTPUT':'AWAITING RUN'}</span></summary><pre>{logs || (busy?'Starting isolated solver worker…':'No simulation has run in this session.')}</pre></details>
        </div>
        <p className="science-note">A teaching-scale FDTD calculation with nondispersive materials and a finite source. Fields are not normalized transmission; convergence and device performance require separate studies.</p>
      </section>

      <aside className="panel execution-panel">
        <div className="panel-heading"><div><span className="step">02</span><h2>Choose & run</h2></div><button className="icon-button" aria-label="Refresh available targets" title="Refresh available targets" onClick={()=>loadCapabilities(true)} disabled={capLoading || !!busy}><Symbol kind="refresh" size={15}/></button></div>
        <div className="recommendation"><span className="eyebrow">SOLVER GUIDANCE</span><h3>{capLoading?'Checking your environment…':rec?`${name(rec.solver)} · ${rec.device.toUpperCase()}`:'Choose an available target'}</h3><p>{capLoading?'Verifying imports and actual compute devices.':rec?.reason || 'Availability is checked by the local service. The model uses a supported dielectric FDTD study.'}</p></div>
        <div className="target-label">EXECUTION TARGET<span>Explicit selection required</span></div>
        <div className="targets" role="radiogroup" aria-label="Execution target">{capabilities.targets.map(t=>{
          const key=`${t.solver}:${t.device}`,checked=target===key;
          return <label className={`target-card ${checked?'selected':''} ${!t.available?'unavailable':''}`} key={key}><input type="radio" name="execution-target" value={key} checked={checked} disabled={!t.available || !!busy} onChange={()=>setTarget(key)}/><div className="target-top"><span className="radio-mark"/>{name(t.solver)}<span className="target-chip">{t.device==='gpu'?'NVIDIA GPU':'CPU'}</span></div><div className="target-description">{t.device_name || (t.device==='gpu'?'JAX CUDA execution':'Local processor')}</div><div className={`readiness ${t.available?'ready':''}`}><i/>{t.available?'Ready':'Unavailable'}{t.version&&<span>v{t.version}</span>}</div>{t.reason&&<div className="target-reason">{t.reason}</div>}</label>;
        })}{!capabilities.targets.length&&<div className="target-placeholder">{capLoading?<span className="spinner"/>:'No execution targets reported. Check the local API connection.'}</div>}</div>
        <div className="run-control"><button className="button primary run-button" onClick={startRun} disabled={!loaded || !selected?.available || !validation?.valid || !!validationError || !!busy}>{busy?<><span className="spinner"/>Simulation {job?.status || 'starting'}…</>:<><Symbol kind="play" size={17}/>Run simulation<Symbol kind="arrow" size={17}/></>}</button>{busy?<button className="cancel-button" onClick={cancel} disabled={!job || submitting}>Cancel run</button>:<p>{selected?`${name(selected.solver)} on ${selected.device.toUpperCase()} · stays on this machine`:'Select a ready CPU or GPU target above.'}</p>}</div>
        <div className="execution-notes"><Symbol kind="cpu" size={18}/><p>No commercial solver license.<br/>No cloud compute. No silent fallback.</p></div>
      </aside>
    </main>
    </div>
    {error&&<div className="global-message error-banner" role="alert"><span>{error}</span><button onClick={()=>setError('')} aria-label="Dismiss error">×</button></div>}
    {notice&&<div className="global-message notice" role="status"><span>{notice}</span><button onClick={()=>setNotice('')} aria-label="Dismiss notice">×</button></div>}
    <footer className="app-footer"><span><i/> OPEN SOURCE PHOTONICS</span><span>Meep + FDTDX / JAX<span className="footer-separator">·</span>Built for exploring, learning, and inspecting.</span></footer>
  </div>;
}
