import { useEffect, useRef, useState } from 'react';
import { request } from './api';
import { RunPassport } from './RunPassport';
import { CodeExport } from './CodeExport';
import { validateGeometryDocument, type GeometryDocument } from './geometryModel';
import './study3d.css';

type Solver = 'meep' | 'fdtdx';
type Device = 'cpu' | 'gpu';
type Settings = { wavelength_um: number; resolution: number; duration_fs: number; pml_um: number; cladding_index: number };
type StudyFormat = { schema_version: 2; study_type: 'waveguide-3d' } | { schema_version: 5; study_type: 'taper-3d' } | { schema_version: 6; study_type: 'layout-3d'; source_um: {x:number;y:number;z:number} };
export type Study3D = Settings & StudyFormat & { geometry: GeometryDocument; solver: Solver; device: Device };
type Target = { solver: Solver; device: Device; available: boolean; reason?: string; device_name?: string; version?: string };
type Job = { id: string; status: string; config: Study3D; logs?: string; error?: string; result?: Record<string, unknown>; elapsed_s?: number };
type PhysicalLayout = {
  domain: { x: number; y: number; z: number };
  x_in: number; x_out: number; width_in: number; width_out: number; height: number; source_x: number;
};
type Validation = { valid: boolean; estimate: Record<string, unknown>; warnings: string[]; physical_layout?: PhysicalLayout; scene_layout?: { domain: Record<string,number>; source_um: Record<string,number>; objects: unknown[] } };
type Plane = 'xy' | 'xz' | 'yz';
type Component = 'Ex' | 'Ey' | 'Ez';
type Volume = { components: Component[]; shape: number[]; x_um: number[]; y_um: number[]; z_um: number[]; field_peak_abs: number; field_peaks_by_component?: Record<Component, number>; metadata: Record<string, unknown> };
type Slice = { field: number[][]; epsilon: number[][]; u_um: number[]; v_um: number[]; u_axis: string; v_axis: string; normal_axis: string; coordinate_um: number; index: number; plane: Plane; component: Component; metadata: Record<string, unknown> };
type Comparison = { kind: 'same-study-repeatability'; reference_run_id: string; candidate_run_id: string; components: Record<Component, { relative_l2: number | null; max_abs_error: number }>; overall_relative_l2: number | null; epsilon_identical: boolean; commercial_agreement: 'not_validated'; notes: string[]; reference_target: string; candidate_target: string };
const initialSettings: Settings = { wavelength_um: 1.55, resolution: 16, duration_fs: 60, pml_um: .75, cladding_index: 1 };
const fieldNames = {'waveguide-3d':'Finite-height waveguide','taper-3d':'Library taper','layout-3d':'Finite layout'};
const fieldRoutes = {
  'layout-3d': {example:'/api/layout/example',validate:'/api/layout/validate',jobs:'/api/layout/jobs'},
  'waveguide-3d': { example: '/api/example3d', validate: '/api/validate3d', jobs: '/api/jobs3d' },
  'taper-3d': { example: '/api/taper/example', validate: '/api/taper/validate', jobs: '/api/taper/jobs' },
};
const solverName = (solver: Solver) => solver === 'meep' ? 'Meep' : 'FDTDX';
const settingsFrom = (study: Study3D): Settings => ({ wavelength_um: study.wavelength_um, resolution: study.resolution, duration_fs: study.duration_fs, pml_um: study.pml_um, cladding_index: study.cladding_index });
const canonical = (value: unknown): string => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);

function saveStudy(study: Study3D) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(study, null, 2) + '\n'], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `${study.study_type}-study.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function StudyComparison({ job, recent }: { job: Job; recent: Job[] }) {
  const [reference, setReference] = useState('');
  const [result, setResult] = useState<Comparison | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const sequence = useRef(0);
  const compatibleCandidates = recent.filter(item => item.id !== job.id && item.status === 'completed' && item.config.solver === job.config.solver);
  useEffect(() => () => { sequence.current++; }, [job.id]);
  function choose(value: string) { sequence.current++; setReference(value); setResult(null); setError(''); setLoading(false); }
  async function compare() {
    if (!reference) return;
    const current = ++sequence.current;
    setLoading(true); setResult(null); setError('');
    try {
      const value = await request<Comparison>('/api/compare3d', { reference_run_id: reference, candidate_run_id: job.id });
      if (current !== sequence.current) return;
      if (value.reference_run_id !== reference || value.candidate_run_id !== job.id || value.kind !== 'same-study-repeatability') throw new Error('Comparison response does not match the selected runs.');
      setResult(value);
    } catch (reason) { if (current === sequence.current) setError((reason as Error).message); }
    finally { if (current === sequence.current) setLoading(false); }
  }
  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `comparison-${result.reference_run_id.slice(0, 8)}-${result.candidate_run_id.slice(0, 8)}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const numerical = (value: number | null) => value === null ? 'undefined' : Number.isFinite(value) ? value.toExponential(5) : 'unavailable';
  return <details className="study3d-comparison">
    <summary>Same-study comparison</summary>
    <div className="study3d-comparison-body">
      <p>Checks repeatability on matching studies; not commercial accuracy. The service checks geometry, source, time, sampling, and solver versions before comparing.</p>
      {compatibleCandidates.length ? <div className="study3d-comparison-select"><label>Reference run<select aria-label="3D comparison reference" value={reference} onChange={event => choose(event.target.value)}><option value="">Choose a saved {solverName(job.config.solver)} run…</option>{compatibleCandidates.map(item => <option value={item.id} key={item.id}>{item.id.slice(0, 8)} · {item.config.device.toUpperCase()} · {item.config.wavelength_um} µm</option>)}</select></label><button className="button secondary compact" onClick={compare} disabled={!reference || loading}>{loading ? 'Comparing…' : 'Compare saved runs'}</button></div> : <p>Complete another run with this solver and the same study settings to compare its saved numerical arrays.</p>}
      {error && <div className="frame-error" role="alert">Comparison unavailable: {error}</div>}
      {result && <div className="study3d-comparison-result">
        <p>Reference {result.reference_run_id.slice(0, 8)} → current run {result.candidate_run_id.slice(0, 8)}</p>
        <p>{result.reference_target} → {result.candidate_target}</p>
        <table><thead><tr><th scope="col">Component</th><th scope="col">Relative L2</th><th scope="col">Max |difference|</th></tr></thead><tbody>{(['Ex', 'Ey', 'Ez'] as const).map(component => <tr key={component}><th scope="row">{component}</th><td>{numerical(result.components[component].relative_l2)}</td><td>{numerical(result.components[component].max_abs_error)}</td></tr>)}</tbody></table>
        <p>Combined relative L2: <strong className="comparison-overall">{numerical(result.overall_relative_l2)}</strong> · Stored permittivity: {result.epsilon_identical ? 'identical' : 'different'}</p>
        {result.notes.map((note, i) => <p key={i}>{note}</p>)}
        <p>Commercial solver agreement: not validated.</p><button className="button secondary compact" onClick={download}>Export comparison JSON</button>
      </div>}
    </div>
  </details>;
}

function SliceCanvas({ slice, material, peak }: { slice: Slice; material: boolean; peak: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [point, setPoint] = useState<{ x: number; y: number } | null>(null);
  const values = material ? slice.epsilon : slice.field;
  const ny = values.length, nx = values[0]?.length ?? 0;
  let min = Infinity, max = -Infinity;
  for (const row of values) for (const value of row) { min = Math.min(min, value); max = Math.max(max, value); }
  const limit = peak > 0 && Number.isFinite(peak) ? peak : Math.max(Math.abs(min), Math.abs(max)) || 1;
  useEffect(() => {
    const element = canvas.current;
    if (!element || !nx || !ny) return;
    element.width = nx; element.height = ny;
    const context = element.getContext('2d'); if (!context) return;
    const pixels = context.createImageData(nx, ny);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const offset = ((ny - y - 1) * nx + x) * 4, value = values[y][x];
      if (material) {
        const t = (value - min) / (max - min || 1);
        pixels.data[offset] = 22 + 67 * t; pixels.data[offset + 1] = 37 + 182 * t; pixels.data[offset + 2] = 54 + 143 * t;
      } else {
        const normalized = Math.max(-1, Math.min(1, value / limit));
        const base = [19, 28, 41], color = normalized < 0 ? [73, 141, 249] : [255, 157, 104];
        base.forEach((channel, i) => { pixels.data[offset + i] = Math.round(channel + (color[i] - channel) * Math.abs(normalized)); });
      }
      pixels.data[offset + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    setPoint(null);
  }, [values, min, max, limit, nx, ny, material]);
  const hover = point && values[point.y]?.[point.x] !== undefined
    ? `${slice.u_axis} ${slice.u_um[point.x].toFixed(4)} µm · ${slice.v_axis} ${slice.v_um[point.y].toFixed(4)} µm · ${material ? 'εr' : slice.component} ${values[point.y][point.x].toExponential(5)}`
    : 'Move over the slice to inspect an actual stored sample';
  return <div className="slice3d-display" style={{ maxWidth: `${420 * nx / Math.max(ny, 1) + 60}px`, margin: '0 auto', width: '100%' }}>
    <div className="slice3d-axis-v">{slice.v_axis} (µm)</div>
    <div className="slice3d-y-ticks"><span>{slice.v_um.at(-1)?.toFixed(2)}</span><span>{slice.v_um[0]?.toFixed(2)}</span></div>
    <div className="slice3d-image-wrap"><canvas className="slice3d-canvas" ref={canvas} data-slice-index={slice.index} data-plane={slice.plane} data-component={slice.component} aria-label={`Actual 3D ${material ? 'relative permittivity' : slice.component} ${slice.plane.toUpperCase()} slice, ${nx} by ${ny} samples`}
      onMouseMove={event => { const rect = event.currentTarget.getBoundingClientRect(); setPoint({ x: Math.min(nx - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * nx))), y: Math.min(ny - 1, Math.max(0, ny - 1 - Math.floor((event.clientY - rect.top) / rect.height * ny))) }); }}/></div>
    <div className="slice3d-x-ticks"><span>{slice.u_um[0]?.toFixed(2)}</span><span>{slice.u_um.at(-1)?.toFixed(2)}</span></div>
    <div className="slice3d-axis-u">{slice.u_axis} (µm)</div>
    <div className="slice3d-scale"><span>{material ? 'Relative permittivity εr' : `${slice.component} · arbitrary source units`}</span><div><div className={`colorbar ${material ? 'epsilon' : 'field'}`}/><div className="colorbar-labels"><span>{(material ? min : -limit).toPrecision(3)}</span><span>{material ? '' : '0'}</span><span>{(material ? max : limit).toPrecision(3)}</span></div></div></div>
    <p className="slice3d-readout">{hover}</p>
  </div>;
}

export function VolumeSlices({ job }: { job: { id: string; result?: Record<string, unknown> } }) {
  const [volume, setVolume] = useState<Volume | null>(null);
  const [slice, setSlice] = useState<Slice | null>(null);
  const [plane, setPlane] = useState<Plane>('xy');
  const [component, setComponent] = useState<Component>('Ey');
  const [index, setIndex] = useState<number | null>(null);
  const [material, setMaterial] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setError(''); setLoading(true);
    request<Volume>(`/api/jobs/${job.id}/volume`).then(value => { if (active) setVolume(value); }).catch(reason => { if (active) { setError(reason.message); setLoading(false); } });
    return () => { active = false; };
  }, [job.id, retry]);
  const axis = plane === 'xy' ? volume?.z_um : plane === 'xz' ? volume?.y_um : volume?.x_um;
  const selectedIndex = index ?? Math.floor((axis?.length ?? 1) / 2);
  useEffect(() => {
    if (!volume || !axis?.length) return;
    let active = true;
    setLoading(true); setError(''); setSlice(null);
    request<Slice>(`/api/jobs/${job.id}/slice?plane=${plane}&index=${selectedIndex}&component=${component}`).then(value => {
      if (!active) return;
      if (value.plane !== plane || value.component !== component || value.index !== selectedIndex || value.field.length !== value.v_um.length || value.field.some(row => row.length !== value.u_um.length)) throw new Error('Stored slice does not match the requested plane, component, or coordinates.');
      setSlice(value); setLoading(false);
    }).catch(reason => { if (active) { setError(reason.message); setLoading(false); } });
    return () => { active = false; };
  }, [job.id, volume, plane, component, selectedIndex, retry]);
  const metadata = volume?.metadata ?? job.result ?? {};
  const time = metadata.actual_duration_fs ?? metadata.simulated_time_fs;
  // React renders before its fetch effect runs. Never pair old samples with a
  // newly selected plane/component/index, even for that intermediate render.
  const matchingSlice = slice?.plane === plane && slice.component === component && slice.index === selectedIndex ? slice : null;
  return <div className="volume3d-results">
    <div className="slice3d-header"><div><h3>Actual 3D fields</h3><p>Final solver snapshot{typeof time === 'number' ? ` · ${time.toFixed(2)} fs` : ''} · slices through the saved volume</p><p>Exploratory grid · Commercial solver agreement: not validated</p></div><span className="viewport-badge">COMPUTED</span></div>
    <div className="slice3d-controls">
      <label>Plane<select aria-label="3D slice plane" value={plane} onChange={event => { setPlane(event.target.value as Plane); setIndex(null); }}><option value="xy">XY · z normal</option><option value="xz">XZ · y normal</option><option value="yz">YZ · x normal</option></select></label>
      <label>Component<select aria-label="3D field component" value={component} onChange={event => setComponent(event.target.value as Component)}>{['Ex', 'Ey', 'Ez'].map(value => <option key={value}>{value}</option>)}</select></label>
      <button className={material ? 'selected' : ''} aria-pressed={material} onClick={() => setMaterial(value => !value)}>Material grid</button>
    </div>
    {volume && axis && <div className="slice3d-position"><label htmlFor={`slice-${job.id}`}>Slice position <strong>{plane === 'xy' ? 'z' : plane === 'xz' ? 'y' : 'x'} = {axis[selectedIndex]?.toFixed(4)} µm</strong></label><input id={`slice-${job.id}`} aria-label="3D slice index" type="range" min="0" max={axis.length - 1} step="1" value={selectedIndex} onChange={event => setIndex(Number(event.target.value))}/><span>{selectedIndex + 1} / {axis.length}</span></div>}
    {loading || (!matchingSlice && !error) ? <div className="slice3d-loading"><span className="spinner"/>Reading actual solver samples…</div> : matchingSlice && !error ? <SliceCanvas key={`${plane}:${component}:${selectedIndex}:${material}`} slice={matchingSlice} material={material} peak={volume?.field_peaks_by_component?.[matchingSlice.component] ?? volume?.field_peak_abs ?? 0}/> : null}
    {error && <div className="frame-error" role="alert"><span>3D results unavailable: {error}</span><button onClick={() => setRetry(value => value + 1)}>Retry 3D results</button></div>}
    {volume && <p className="slice3d-note">Volume: {volume.shape[2]} × {volume.shape[1]} × {volume.shape[0]} samples (x × y × z). Field scale is fixed across slices of each component. Final-time samples only; no synthesized time evolution. Permittivity shows the solver's stored material sampling.{typeof metadata.field_sampling === 'string' ? ` ${metadata.field_sampling}` : ''}</p>}
  </div>;
}

export function Study3DPanel({ geometry, familyNames, onOpenGeometry, onEditGeometry }: { geometry: GeometryDocument; familyNames: Record<string,string>; onOpenGeometry: (document: GeometryDocument) => void; onEditGeometry: (document: GeometryDocument) => void }) {
  const [studySettings, setStudySettings] = useState({
    'waveguide-3d': initialSettings,
    'layout-3d': {...initialSettings, duration_fs:35},
    'taper-3d': { ...initialSettings, resolution: 20 },
  });
  const [source, setSource] = useState({x:0,y:0,z:0});
  const [model, setModel] = useState<StudyFormat['study_type'] | 'auto'>('auto');
  const [targets, setTargets] = useState<Target[]>([]);
  const [choice, setChoice] = useState<{ kind: Study3D['study_type']; target: string } | null>(null);
  const [checkingTargets, setCheckingTargets] = useState(true);
  const [validation, setValidation] = useState<{ key: string; data: Validation } | null>(null);
  const [validationError, setValidationError] = useState('');
  const [error, setError] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [opening, setOpening] = useState(false);
  const [retryValidation, setRetryValidation] = useState(0);
  const [recent, setRecent] = useState<Job[]>([]);
  const instance = geometry.components?.[0];
  const presetFamily = geometry.components?.length === 1 && instance && Object.hasOwn(familyNames,instance.family) ? instance.family : null;
  const currentGeometry = useRef(geometry); currentGeometry.current = geometry;
  const singleTaper = geometry.components?.length === 1 && geometry.objects.length === 1 &&
    instance?.family === 'taper' && Object.values(instance.placement).every(value => value === 0);
  const automatic = geometry.schema_version === 1 ? 'waveguide-3d' : singleTaper ? 'taper-3d' : 'layout-3d';
  const kind = model === 'auto' ? automatic : model;
  const format: StudyFormat = kind === 'layout-3d' ? {schema_version:6, study_type:kind, source_um:source}
    : kind === 'taper-3d' ? {schema_version:5, study_type:kind} : {schema_version:2, study_type:kind};
  const libraryDraft = kind === 'taper-3d';
  const finite = kind === 'layout-3d';
  const settings = studySettings[kind];
  const routes = fieldRoutes[kind];
  const target = choice?.kind === kind ? choice.target : '';
  const selected = targets.find(item => `${item.solver}:${item.device}` === target);
  const config: Study3D = { ...format, geometry, ...settings, solver: selected?.solver ?? 'meep', device: selected?.device ?? 'cpu' };
  const configKey = canonical(config);
  const busy = submitting || job?.status === 'queued' || job?.status === 'running';
  const valid = validation?.key === configKey && validation.data.valid;
  const hasChanged = job && canonical(job.config) !== configKey;
  const core = geometry.objects.find(object => object.id === 'core');
  const substrate = geometry.objects.find(object => object.id === 'substrate');
  function checkTargets(refresh = false) {
    setCheckingTargets(true);
    request<{ targets: Target[] }>(`/api/capabilities${refresh ? '?refresh=true' : ''}`).then(value => setTargets(value.targets)).catch(reason => setError(`Target check: ${reason.message}`)).finally(() => setCheckingTargets(false));
  }
  function loadRecent() {
    request<Job[]>('/api/jobs').then(value => setRecent(value.filter(item =>
      item.config?.study_type in fieldRoutes))).catch(() => {});
  }
  useEffect(() => { checkTargets(); loadRecent(); }, []);
  useEffect(() => {
    let active = true;
    setValidation(null); setValidationError('');
    const timer = setTimeout(() => request<Validation>(routes.validate, config).then(data => { if (active) setValidation({ key: configKey, data }); }).catch(reason => { if (active) setValidationError(reason.message); }), 300);
    return () => { active = false; clearTimeout(timer); };
  }, [configKey, retryValidation]);
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let active = true;
    const timer = setInterval(() => request<Job>(`/api/jobs/${job.id}`).then(value => { if (active) { setJob(value); if (!['queued', 'running'].includes(value.status)) loadRecent(); } }).catch(reason => { if (active) setError(`Run status: ${reason.message}`); }), 800);
    return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status]);
  async function openExample(type: Study3D['study_type'], family?: string) {
    setOpening(true); setError('');
    try {
      const url = fieldRoutes[type].example + (family ? '?family=' + encodeURIComponent(family) : '');
      const example = await request<Study3D>(url);
      if (currentGeometry.current !== geometry) throw new Error('The editor changed while opening this example. Review the current design and try again.');
      if (example.study_type !== type) throw new Error('Returned example uses a different study type.');
      const document = validateGeometryDocument(example.geometry);
      if (family && (document.components?.length !== 1 || document.components[0].family !== family)) throw new Error('Returned field setup has a different component family.');
      setModel(type);
      if (example.study_type === 'layout-3d') setSource(example.source_um);
      onOpenGeometry(document);
      setStudySettings(current => ({ ...current, [type]: settingsFrom(example) }));
      setChoice(null);
    }
    catch (reason) { setError(`3D example could not be opened: ${(reason as Error).message}`); }
    finally { setOpening(false); }
  }
  function fitSubstrate() {
    if (!core || core.kind !== 'box' || !substrate || substrate.kind !== 'box') return;
    const top = core.position.z - core.size.z / 2, bottom = -geometry.domain.z / 2;
    onEditGeometry({ ...geometry, mode: 'draft-3d', objects: geometry.objects.map(object => object.id === 'substrate' ? { ...object, position: { x: 0, y: 0, z: (top + bottom) / 2 }, size: { x: geometry.domain.x, y: geometry.domain.y, z: top - bottom }, rotation_deg: 0 } : object) });
  }
  async function run() {
    if (!valid || !selected?.available || busy) return;
    const snapshot = JSON.parse(configKey) as Study3D;
    setSubmitting(true); setError('');
    try { await request<Validation>(routes.validate, snapshot); setJob(await request<Job>(routes.jobs, snapshot)); }
    catch (reason) { setError(`3D run could not start: ${(reason as Error).message}`); }
    finally { setSubmitting(false); }
  }
  async function cancel() { if (job) try { setJob(await request<Job>(`/api/jobs/${job.id}/cancel`, {})); loadRecent(); } catch (reason) { setError((reason as Error).message); } }
  async function openRun(id: string) { if (!id) return; try { setJob(await request<Job>(`/api/jobs/${id}`)); } catch (reason) { setError((reason as Error).message); } }
  function parameter(key: keyof Settings, label: string, unit: string, step: number) {
    return <label className="parameter"><span>{label}</span><div className="input-wrap"><input aria-label={`3D ${label}`} type="number" value={Number.isFinite(settings[key]) ? settings[key] : ''} step={step} onChange={event => {
      const value = event.target.value === '' ? NaN : Number(event.target.value);
      setStudySettings(current => ({ ...current, [kind]: { ...current[kind], [key]: value } }));
    }}/><span>{unit}</span></div></label>;
  }
  const estimate = validation?.data.estimate;
  const layout = valid ? validation.data.physical_layout : undefined;
  const actualDevice = job?.result?.device_name ?? job?.result?.actual_device;
  return <section className="study3d panel" aria-label="Finite-height 3D simulation">
    <div className="study3d-heading">
      <div><h2>{fieldNames[kind]} field study</h2>
        <p>{libraryDraft ? 'Validate the current editor design, then compute its electric field and material grid.' : 'A real 3D calculation, with absorbing boundaries on all six faces.'}</p></div>
      <div className="study3d-examples">
        {presetFamily && <div className="study3d-preset"><button className="button primary compact" onClick={() => openExample('layout-3d',presetFamily)} disabled={opening || !!busy}>Open tested {familyNames[presetFamily]} field setup</button><small>Opens a new design. Undo restores the previous layout.</small></div>}
        <button className="button secondary compact" onClick={() => openExample('waveguide-3d')} disabled={opening || !!busy}>Open runnable 3D waveguide</button>
        <button className="button secondary compact" onClick={() => openExample('layout-3d')} disabled={opening || !!busy}>Open finite coupler example</button>
        <button className="button secondary compact" onClick={() => openExample('taper-3d')} disabled={opening || !!busy}>Open runnable taper</button>
      </div>
    </div>
    <div className="study3d-layout">
      <div className="study3d-setup">
        <h3>STUDY SETTINGS</h3>
        <label className="parameter">Excitation model<select aria-label="3D excitation model" value={model} disabled={!!busy} onChange={e=>{setModel(e.target.value as typeof model);setChoice(null);}}>
          <option value="auto">From current geometry</option><option value="layout-3d">Finite layout · local Ey dipole</option>
          <option value="taper-3d">Taper · matched access guides</option><option value="waveguide-3d">Straight waveguide study</option>
        </select></label>
        {finite && <fieldset className="study3d-parameters"><legend>Ey dipole position (µm)</legend>{(['x','y','z'] as const).map(a=>
          <label className="parameter" key={a}>{a.toUpperCase()}<input type="number" aria-label={`Dipole ${a} (µm)`} step=".05" value={Number.isFinite(source[a]) ? source[a] : ''} onChange={e=>setSource({...source,[a]:e.target.value === '' ? NaN : Number(e.target.value)})}/></label>)}</fieldset>}
        <div className="study3d-parameters">{parameter('wavelength_um', 'Wavelength', 'µm', .05)}{parameter('cladding_index', 'Cladding index', 'n', .05)}{parameter('resolution', 'Resolution', 'cells / µm', 1)}{parameter('duration_fs', 'Duration', 'fs', 10)}{parameter('pml_um', 'PML thickness', 'µm', .0625)}</div>
        <div className="study3d-model-note">{finite ? 'Drive the finite structure as drawn with a local electric dipole. Solids end inside homogeneous cladding. The last solid wins where materials overlap.' : libraryDraft
          ? 'Use one centred, unrotated linear taper from the component library. Its finite-height polygon and lossless constant core material define the device. The study adds matched straight access guides extending through the X absorbing layers.'
          : 'Use one centred rectangular core spanning the domain along X. An optional full-width substrate can meet its bottom face. Dimensions and indices come from the inspector. Visibility affects only the preview.'}</div>
        {kind==='waveguide-3d' && core && substrate && <button className="study3d-fit" onClick={fitSubstrate}>Fit substrate to core</button>}
        <div className={`study3d-validation ${validationError ? 'invalid' : ''}`} role="status">{validationError ? <><strong>Study needs attention</strong><p>{validationError}</p><button onClick={() => setRetryValidation(value => value + 1)}>Recheck study</button></> : valid ? <><strong>3D study validated</strong>{estimate && <p>{Array.isArray(estimate.grid) ? `${estimate.grid.join(' × ')} cells` : estimate.cells ? `${Number(estimate.cells).toLocaleString()} cells` : 'Bounded study ready for execution'}{typeof estimate.planning_memory_mib === 'number' ? ` · memory planning allowance ${estimate.planning_memory_mib.toFixed(0)} MiB` : ''}</p>}</> : <><span className="spinner tiny"/> Checking geometry and numerical settings…</>}</div>
        {valid && !!validation?.data.warnings?.length && <details className="study3d-assumptions"><summary>Study assumptions & limits</summary>{typeof estimate?.note === 'string' && <p>{estimate.note}</p>}{validation.data.warnings.map((value, i) => <p key={i}>{value}</p>)}</details>}
        {finite ? valid && validation.data.scene_layout && <p className="study3d-source">Applied dipole: {Object.values(validation.data.scene_layout.source_um).map(n=>n.toFixed(4)).join(', ')} µm · {validation.data.scene_layout.objects.length} sampled solids · cell {Object.values(validation.data.scene_layout.domain).map(n=>n.toFixed(3)).join(' × ')} µm</p> : libraryDraft ? layout && <div className="study3d-taper-layout" aria-label="Validated taper simulation layout">
          <h4>Simulation domain and excitation</h4>
          <dl>
            <div><dt>FDTD cell (X × Y × Z)</dt><dd>{[layout.domain.x, layout.domain.y, layout.domain.z].map(n => n.toFixed(3)).join(' × ')} µm</dd></div>
            <div><dt>Taper endpoints (X)</dt><dd>{layout.x_in.toFixed(3)} → {layout.x_out.toFixed(3)} µm</dd></div>
            <div><dt>Input → output width</dt><dd>{layout.width_in.toFixed(3)} → {layout.width_out.toFixed(3)} µm</dd></div>
            <div><dt>Core height</dt><dd>{layout.height.toFixed(3)} µm</dd></div>
            <div><dt>Ey source on input lead</dt><dd>x = {layout.source_x.toFixed(3)} µm</dd></div>
          </dl>
        </div> : <p className="study3d-source">Fixed source: finite Ey excitation at x = {( -geometry.domain.x / 2 + settings.pml_um + .5).toFixed(3)} µm, spanning the core cross-section. Four-period turn-on. This is not a solved eigenmode or normalized transmission study.</p>}
        <div className="study3d-target-title"><h3>EXPLICIT EXECUTION TARGET</h3><button onClick={() => checkTargets(true)} disabled={checkingTargets || !!busy}>Refresh targets</button></div>
        <p className="study3d-guidance">Meep CPU provides the reference CPU route. FDTDX offers JAX CPU or NVIDIA GPU; compilation can dominate small runs. Availability is probed separately for each target.</p>
        <div className="study3d-targets" role="radiogroup" aria-label="3D execution target">{targets.map(item => { const key = `${item.solver}:${item.device}`; return <label className={`study3d-target ${target === key ? 'selected' : ''} ${!item.available ? 'unavailable' : ''}`} key={key}><input type="radio" name="study3d-target" value={key} checked={target === key} disabled={!item.available || !!busy} onChange={() => setChoice({ kind, target: key })}/><strong>{solverName(item.solver)} · {item.device === 'gpu' ? 'NVIDIA GPU' : 'CPU'}</strong><span>{item.available ? item.device_name ?? 'Ready' : item.reason ?? 'Unavailable'}</span></label>; })}{checkingTargets && !targets.length && <p>Checking installed solvers and devices…</p>}</div>
        <div className="study3d-run-buttons"><button className="button primary" onClick={run} disabled={!valid || !selected?.available || !!busy}>{busy ? '3D simulation running…' : 'Run 3D simulation'}</button>{busy ? <button className="button secondary" onClick={cancel} disabled={submitting || !job}>Cancel 3D run</button> : <button className="button secondary" onClick={() => saveStudy(config)} disabled={!valid || !selected?.available}>Export study JSON</button>}</div>
        <p className="study3d-explicit">{selected ? `${solverName(selected.solver)} / ${selected.device.toUpperCase()} · local process · no CPU fallback` : 'Choose an available target to enable Run.'}</p>
        <CodeExport value={config} disabled={!valid||!selected?.available||!!busy}/>
      </div>
      <div className="study3d-output">
        <div className="study3d-output-heading"><h3>INSPECT & EXPORT</h3>{job && <span className={`status-tag ${job.status}`}><i/>{job.status}</span>}</div>
        {!!recent.length && <label className="study3d-recent">Recent 3D runs<select aria-label="Recent 3D run" value={job && recent.some(item => item.id === job.id) ? job.id : ''} disabled={!!busy} onChange={event => openRun(event.target.value)}><option value="">Select a saved run…</option>{recent.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {fieldNames[item.config.study_type]} · {solverName(item.config.solver)} / {item.config.device.toUpperCase()} · {item.status}</option>)}</select></label>}
        {job ? <><div className="study3d-run-facts"><span>Run <strong>{job.id.slice(0, 10)}</strong></span><span><strong>{solverName(job.config.solver)} / {job.config.device.toUpperCase()}</strong></span><span>{job.config.wavelength_um} µm · {job.config.resolution} cells / µm</span></div>
          {hasChanged && <div className="snapshot-note">The editor differs from this run. Results retain their saved geometry and study settings.</div>}
          {job.status === 'completed' && <><div className="study3d-execution-proof">Executed on {String(actualDevice ?? 'recorded target')} · finite 3D volume</div><VolumeSlices key={job.id} job={job}/><div className="study3d-export"><a className="button primary compact" href={`/api/jobs/${job.id}/export`}>Export 3D results .zip</a><button className="button secondary compact" onClick={() => saveStudy(job.config)}>Save run configuration</button>{job.config.study_type !== 'waveguide-3d' && <a className="button secondary compact" href={`/api/jobs/${job.id}/reproduce`}>Export {job.config.study_type === 'taper-3d' ? 'taper' : 'layout'} run Python + notebook</a>}</div><RunPassport key={job.id} jobId={job.id}/><StudyComparison key={job.id} job={job} recent={recent.filter(item => item.config.study_type === job.config.study_type)}/></>}
          {job.error && <div className="error-banner" role="alert">{job.error}</div>}
          <details className="logs" open={!!busy || job.status === 'failed'}><summary>3D solver log</summary><pre>{job.logs || 'Waiting for the isolated solver worker…'}</pre></details>
        </> : <div className="study3d-empty"><span>↗</span><h3>Inspect a real volume.</h3><p>Open the runnable example, edit its geometry, choose CPU or GPU, then run. Completed results expose Ex, Ey, Ez and material slices on XY, XZ and YZ planes.</p></div>}
      </div>
    </div>
    {error && <div className="study3d-error error-banner" role="alert"><span>{error}</span><button aria-label="Dismiss 3D study error" onClick={() => setError('')}>×</button></div>}
  </section>;
}
