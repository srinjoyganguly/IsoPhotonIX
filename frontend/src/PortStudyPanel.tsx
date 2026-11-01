import { useEffect, useRef, useState, type ReactNode } from 'react';
import { request } from './api';
import { RunPassport } from './RunPassport';
import { CodeExport } from './CodeExport';
import { PortSpectrum } from './PortSpectrum';
import './portStudy.css';

// Presentation boundary while the port-study API is qualified. The controller
// supplies actual catalog entries, validation and execution states; this panel
// never invents defaults, solver support, progress or numerical results.
export type PortTemplate = { id: string; title: string; description: string };
export type PortParameter = {
  key: string; label: string; value: number | ''; unit: string;
  group: 'geometry' | 'spectrum' | 'numerics'; step?: number; min?: number; max?: number;
  description?: string; disabled?: boolean;
};
export type PortTarget = {
  id: string; label: string; available: boolean; reason: string;
  device_name?: string; version?: string;
};
export type PortStage = {
  id: 'reference' | 'device' | 'measurements';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  detail?: string; progress?: number;
};
export type PortValidation = {
  valid: boolean; summary: string; warnings?: string[];
  source_description?: string; monitor_description?: string; estimate_description?: string;
};
export type PortStudyPanelProps = {
  templates?: PortTemplate[]; templateId?: string; parameters?: PortParameter[];
  targets?: PortTarget[]; selectedTargetId?: string; targetLoading?: boolean;
  validation?: PortValidation | null; validationPending?: boolean;
  recommendation?: string; stages?: PortStage[]; busy?: boolean;
  immutableRunNote?: string; executionDescription?: string; error?: string;
  onTemplateChange?: (id: string) => void;
  onParameterChange?: (key: string, value: number | '') => void;
  onTargetChange?: (id: string) => void;
  onRun?: () => void; onCancel?: () => void; onRefreshTargets?: () => void;
  onOpenExample?: () => void; onDismissError?: () => void;
  results?: ReactNode; configurationActions?: ReactNode; outputNavigation?: ReactNode;
};

const stageLabels: Record<PortStage['id'], { title: string; description: string }> = {
  reference: { title: 'Incident reference', description: 'Measure the same source in a straight guide.' },
  device: { title: 'Device study', description: 'Measure reflection and transmission around the device.' },
  measurements: { title: 'Normalized spectra', description: 'Check incident power and run diagnostics before interpreting each sample.' },
};
const groupLabels: Record<PortParameter['group'], string> = {
  geometry: 'Geometry and materials', spectrum: 'Wavelength sampling', numerics: 'Numerical settings',
};

function PortProgress({ stages = [] }: { stages?: PortStage[] }) {
  return <ol className="port-progress" aria-label="Port study stages">
    {(Object.keys(stageLabels) as PortStage['id'][]).map((id, index) => {
      const stage = stages.find(item => item.id === id);
      const status = stage?.status ?? 'pending';
      const fraction = stage?.progress;
      return <li key={id} className={`port-stage ${status}`} aria-current={status === 'running' ? 'step' : undefined}>
        <span className="port-stage-number" aria-hidden="true">{index + 1}</span>
        <div><div className="port-stage-heading"><h3>{stageLabels[id].title}</h3><span>{status === 'pending' ? 'Not started' : status}</span></div>
          <p>{stage?.detail || stageLabels[id].description}</p>
          {status === 'running' && typeof fraction === 'number' && Number.isFinite(fraction) && fraction >= 0 && fraction <= 1
            && <progress max={1} value={fraction} aria-label={`${stageLabels[id].title} progress`}/>}</div>
      </li>;
    })}
  </ol>;
}

export function PortStudyPanel({
  templates = [], templateId = '', parameters = [], targets = [], selectedTargetId = '', targetLoading = false,
  validation = null, validationPending = false, recommendation, stages, busy = false,
  immutableRunNote, executionDescription, error, onTemplateChange, onParameterChange, onTargetChange,
  onRun, onCancel, onRefreshTargets, onOpenExample, onDismissError, results, configurationActions, outputNavigation,
}: PortStudyPanelProps) {
  const selectedTarget = targets.find(target => target.id === selectedTargetId);
  const canRun = !!onRun && !!validation?.valid && !validationPending && !!selectedTarget?.available && !busy;
  return <section className="port-workspace" aria-label="2D port measurements">
    <div className="port-intro"><div><span className="eyebrow">FROM FIELDS TO MEASUREMENTS</span><h1>Port measurements</h1><p>A reference and a device run. Spectra with an explicit normalization.</p></div>
      <div className="port-intro-actions"><span className="port-dimension">2D · INVARIANT Z</span>{onOpenExample && <button className="button secondary compact" onClick={onOpenExample} disabled={busy}>Open straight-guide validation</button>}</div></div>
    <div className="port-scope"><strong>Bounded guided-wave study</strong><p>These studies measure a two-dimensional slab guide. They do not apply calibrated ports to the editable 3D geometry. A matched incident-reference run is part of every measurement.</p></div>
    <div className="port-layout">
      <aside className="panel port-setup" aria-label="Port study setup">
        <div className="panel-heading"><h2>Study setup</h2><span className="tag">REFERENCE + DEVICE</span></div>
        <div className="port-setup-body">
          <h3 className="port-section-title">Start from a study</h3>
          {templates.length ? <div className="port-templates" role="group" aria-label="Port study examples">{templates.map(template => <button key={template.id} className={template.id === templateId ? 'selected' : ''} aria-pressed={template.id === templateId} disabled={busy || !onTemplateChange} onClick={() => onTemplateChange?.(template.id)}><strong>{template.title}</strong><span>{template.description}</span></button>)}</div>
            : <p className="port-muted">No supported port-study catalog has been loaded.</p>}
          {(Object.keys(groupLabels) as PortParameter['group'][]).map(group => {
            const fields = parameters.filter(parameter => parameter.group === group);
            const fieldset = <fieldset className="port-parameter-group" disabled={busy || !onParameterChange}><legend>{groupLabels[group]}</legend><div className="port-parameters">{fields.map(parameter => <label className="parameter" key={parameter.key}><span>{parameter.label}</span><div className="input-wrap"><input aria-label={`Port ${parameter.label}`} type="number" value={parameter.value} step={parameter.step ?? 'any'} min={parameter.min} max={parameter.max} disabled={parameter.disabled} onChange={event => onParameterChange?.(parameter.key, event.target.value === '' ? '' : Number(event.target.value))}/><span>{parameter.unit}</span></div>{parameter.description && <small>{parameter.description}</small>}</label>)}</div></fieldset>;
            return !fields.length ? null : group === 'numerics' ? <details key={group} className="port-advanced"><summary>Advanced numerics and settling</summary>{fieldset}</details> : <div key={group}>{fieldset}</div>;
          })}
          {validationPending && <p className="port-validation" role="status">Checking the current study…</p>}
          {!validationPending && validation && <div className={`port-validation ${validation.valid ? 'valid' : 'invalid'}`} role="status"><strong>{validation.summary}</strong>{!!validation.warnings?.length && <details><summary>Model assumptions and limits</summary>{validation.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</details>}</div>}
          {validation?.source_description && <p className="port-method"><strong>Mode excitation</strong>{validation.source_description}</p>}
          {validation?.monitor_description && <p className="port-method"><strong>Measurement planes</strong>{validation.monitor_description}</p>}
          {validation?.estimate_description && <p className="port-muted">{validation.estimate_description}</p>}
          <div className="port-target-heading"><h3 className="port-section-title">Choose an execution target</h3>{onRefreshTargets && <button onClick={onRefreshTargets} disabled={targetLoading || busy}>Refresh</button>}</div>
          {recommendation && <p className="port-recommendation">{recommendation}</p>}
          <p className="port-muted">Select an available target explicitly. The reference and device use the same target; GPU requests never switch to CPU automatically.</p>
          {targetLoading ? <p className="port-muted" role="status">Checking support for port studies…</p> : <div className="port-targets">{targets.map(target => <label key={target.id} className={`port-target ${target.id === selectedTargetId ? 'selected' : ''} ${target.available ? '' : 'unavailable'}`}><input type="radio" name="port-target" value={target.id} checked={target.id === selectedTargetId} disabled={!target.available || busy || !onTargetChange} onChange={() => onTargetChange?.(target.id)}/><strong>{target.label}</strong>{target.device_name && <span>{target.device_name}</span>}<span>{target.reason}</span>{target.version && <small>Solver version {target.version}</small>}</label>)}</div>}
          {!targetLoading && !targets.length && <p className="port-muted">No qualified execution targets have been loaded for this study.</p>}
          <div className="port-run-actions"><button className="button primary" onClick={onRun} disabled={!canRun}>{busy ? 'Please wait…' : 'Run reference + device'}</button>{busy && onCancel && <button className="button secondary" onClick={onCancel}>Cancel measurement</button>}</div>
          <p className="port-run-note">Each run keeps its own configuration and normalization evidence.</p>
          {configurationActions}
        </div>
      </aside>
      <div className="panel port-output" aria-label="Port measurement results">
        <div className="panel-heading"><h2>Measurement sequence</h2><span className="tag">LOCAL COMPUTATION</span></div>
        <div className="port-output-body">{outputNavigation}<PortProgress stages={stages}/>
          {immutableRunNote && <p className="snapshot-note">{immutableRunNote}</p>}
          {executionDescription && <p className="port-execution">{executionDescription}</p>}
          {results || <div className="port-empty"><span aria-hidden="true">∿</span><h3>Inspect power, with its reference.</h3><p>Run the matched reference and device studies to view actual reflection and transmission samples. Numerical diagnostics will identify unsupported or unsettled samples.</p></div>}
          <div className="port-interpretation"><strong>Read the residual carefully</strong><p>1 − R − T is uncollected flux. It can include radiation outside the measurement planes and numerical or unsettled-field error. It is not an absorption measurement.</p><p>Negative values and values above one remain visible. Normalized flux is not a complex S-parameter or a mode-resolved power measurement unless the result explicitly records that method.</p></div>
        </div>
      </div>
    </div>
    {error && <div className="port-error error-banner" role="alert"><span>{error}</span>{onDismissError && <button aria-label="Dismiss port measurement error" onClick={onDismissError}>×</button>}</div>}
  </section>;
}

type Solver = 'meep' | 'fdtdx';
type Device = 'cpu' | 'gpu';
export type PortConfig = {
  schema_version: 3; study_type: 'waveguide-ports-2d'; device_kind: 'straight' | 'gap';
  gap_um: number; core_width_um: number; core_index: number; cladding_index: number;
  wavelength_min_um: number; wavelength_max_um: number; frequency_points: number;
  length_um: number; height_um: number; pml_um: number; resolution: number;
  max_after_sources_fs: number; decay_by: number; spectrum_tolerance: number;
  solver: Solver; device: Device;
};
type NumericKey = Exclude<keyof PortConfig, 'schema_version' | 'study_type' | 'device_kind' | 'solver' | 'device'>;
type Layout = { source_x_um: number; reflection_x_um: number; transmission_x_um: number; monitor_aperture_um: number; gap_center_x_um: number; field_component: string; dimensions: string };
type ValidationResponse = { valid: boolean; estimate: Record<string, unknown>; warnings: string[]; layout: Layout };
type TargetResponse = { solver: Solver; device: Device; available: boolean; reason?: string; preparation?: string; device_name?: string; version?: string };
type Capabilities = { targets: TargetResponse[]; recommendation?: { solver?: Solver; device?: Device; reason: string } };
type PortJob = { id: string; status: string; config: PortConfig; logs?: string | string[]; error?: string; result?: Record<string, unknown>; port_progress?: { stage: 'reference' | 'device' | 'normalization' | 'complete'; message?: string } };
const canonical = (value: unknown) => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
const solverName = (value: Solver) => value === 'meep' ? 'Meep' : 'FDTDX';
const templates: PortTemplate[] = [
  { id: 'straight', title: 'Straight-guide validation', description: 'Repeat the guide as its own device and inspect measured T and R.' },
  { id: 'gap', title: 'Waveguide gap', description: 'A centred cladding-filled gap between identical slab guides.' },
];
const parameterDefinitions: (Omit<PortParameter, 'value'> & { key: NumericKey })[] = [
  { key: 'gap_um', label: 'Gap length', unit: 'µm', group: 'geometry', step: .03125, min: 0, max: 2 },
  { key: 'core_width_um', label: 'Core width', unit: 'µm', group: 'geometry', step: .03125, min: .2, max: 1.5 },
  { key: 'core_index', label: 'Core index', unit: 'n', group: 'geometry', step: .05, min: 1.01, max: 3.5 },
  { key: 'cladding_index', label: 'Cladding index', unit: 'n', group: 'geometry', step: .05, min: 1, max: 2.5 },
  { key: 'wavelength_min_um', label: 'Minimum wavelength', unit: 'µm', group: 'spectrum', step: .01, min: .8, max: 2.8 },
  { key: 'wavelength_max_um', label: 'Maximum wavelength', unit: 'µm', group: 'spectrum', step: .01, min: .8, max: 2.8 },
  { key: 'frequency_points', label: 'Frequency samples', unit: 'bins', group: 'spectrum', step: 2, min: 3, max: 61, description: 'Uniform frequency samples appear at nonuniform wavelengths.' },
  { key: 'resolution', label: 'Resolution', unit: 'cells/µm', group: 'numerics', step: 4, min: 20, max: 48 },
  { key: 'length_um', label: 'Domain length', unit: 'µm', group: 'numerics', step: .5, min: 10, max: 16 },
  { key: 'height_um', label: 'Domain height', unit: 'µm', group: 'numerics', step: .5, min: 4, max: 8 },
  { key: 'pml_um', label: 'PML thickness', unit: 'µm', group: 'numerics', step: .125, min: .75, max: 1.5 },
  { key: 'max_after_sources_fs', label: 'After-source time cap', unit: 'fs', group: 'numerics', step: 100, min: 300, max: 1500, description: 'Each solve must settle within this cap after the pulse ends.' },
  { key: 'decay_by', label: 'Energy decay', unit: 'ratio', group: 'numerics', step: 1e-8, min: 1e-8, max: 1e-4 },
  { key: 'spectrum_tolerance', label: 'Spectrum stability', unit: 'relative', group: 'numerics', step: 1e-5, min: 1e-6, max: 1e-2, description: 'Separate spectral-change criterion; not a mesh-convergence tolerance.' },
];

function saveConfig(value: PortConfig, filename = 'port-study.json') {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + '\n'], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function PortGeometry({ config, layout }: { config: PortConfig; layout: Layout }) {
  const left = 55, top = 28, width = 660, height = 210;
  const x = (coordinate: number) => left + (coordinate / config.length_um + .5) * width;
  const y = (coordinate: number) => top + (.5 - coordinate / config.height_um) * height;
  const coreTop = y(config.core_width_um / 2), coreHeight = config.core_width_um / config.height_um * height;
  const gap = config.device_kind === 'gap' ? config.gap_um : 0;
  const apertureTop = y(layout.monitor_aperture_um / 2), apertureBottom = y(-layout.monitor_aperture_um / 2);
  return <details className="port-geometry" open><summary>Current validated geometry · 2D schematic</summary><svg viewBox="0 0 760 292" role="img" aria-label={`${config.device_kind === 'gap' ? `${config.gap_um} micrometre gap` : 'Straight waveguide'}, ${config.core_width_um} micrometre core. Source x ${layout.source_x_um}, reflection x ${layout.reflection_x_um}, transmission x ${layout.transmission_x_um} micrometres. This is a geometry schematic, not a field result.`}>
    <rect x={left} y={top} width={width} height={height} fill="#0b1723" stroke="#41586b"/>
    <rect x={left} y={top} width={width} height={config.pml_um / config.height_um * height} fill="#3c506944"/><rect x={left} y={y(-config.height_um / 2 + config.pml_um)} width={width} height={config.pml_um / config.height_um * height} fill="#3c506944"/><rect x={left} y={top} width={config.pml_um / config.length_um * width} height={height} fill="#3c506944"/><rect x={x(config.length_um / 2 - config.pml_um)} y={top} width={config.pml_um / config.length_um * width} height={height} fill="#3c506944"/>
    <rect x={left} y={coreTop} width={x(-gap / 2) - left} height={coreHeight} fill="#24534c" stroke="#77cdb7"/><rect x={x(gap / 2)} y={coreTop} width={left + width - x(gap / 2)} height={coreHeight} fill="#24534c" stroke="#77cdb7"/>
    {[{ value: layout.source_x_um, color: '#eebc79', label: 'Mode source' }, { value: layout.reflection_x_um, color: '#baa8ef', label: 'R plane' }, { value: layout.transmission_x_um, color: '#80ddbe', label: 'T plane' }].map(marker => <g key={marker.label}><line x1={x(marker.value)} x2={x(marker.value)} y1={apertureTop} y2={apertureBottom} stroke={marker.color} strokeWidth="2" strokeDasharray={marker.label === 'Mode source' ? undefined : '4 4'}/><text x={x(marker.value)} y={top + height + 19} textAnchor="middle" fill={marker.color} fontSize="10">{marker.label}</text><text x={x(marker.value)} y={top + height + 33} textAnchor="middle" fill="#93abbd" fontSize="9">{marker.value.toFixed(2)} µm</text></g>)}
    <text x={left + width / 2} y={top + 17} textAnchor="middle" fill="#99aebe" fontSize="10">Absorbing PML · all X/Y edges</text><text x={x(0)} y={coreTop - 12} textAnchor="middle" fill="#bed8d5" fontSize="11">{gap ? `Core n = ${config.core_index} · gap ${gap} µm (n = ${config.cladding_index})` : `Continuous core · n = ${config.core_index}`}</text><text transform={`translate(17 ${top + height / 2}) rotate(-90)`} textAnchor="middle" fill="#9eb5c7" fontSize="11">y (µm) · invariant z</text><text x={left + width / 2} y={289} textAnchor="middle" fill="#9eb5c7" fontSize="10">x (µm) · propagation →</text>
  </svg><p>Source and monitor positions come from the validated study. The reference always uses the continuous guide. Gap material: cladding, n = {config.cladding_index}. Aperture: {layout.monitor_aperture_um} µm.</p></details>;
}

function jobStages(job: PortJob | null): PortStage[] {
  if (!job) return [];
  const order = ['reference', 'device', 'normalization', 'complete'];
  const position = order.indexOf(job.port_progress?.stage ?? '');
  return (['reference', 'device', 'measurements'] as const).map((id, index) => {
    let status: PortStage['status'] = 'pending';
    if (job.status === 'completed' || position > index) status = 'completed';
    else if (position === index && job.status === 'running') status = 'running';
    else if ((job.status === 'failed' || job.status === 'cancelled') && (position === index || (position < 0 && index === 0))) status = job.status;
    return { id, status, ...(position === index && job.port_progress?.message ? { detail: job.port_progress.message } : {}) };
  });
}

export function PortStudyWorkspace({ active = true }: { active?: boolean }) {
  const [config, setConfig] = useState<PortConfig | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities>({ targets: [] });
  const [targetId, setTargetId] = useState('');
  const [targetLoading, setTargetLoading] = useState(false);
  const [validation, setValidation] = useState<{ key: string; value: ValidationResponse } | null>(null);
  const [validationPending, setValidationPending] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [error, setError] = useState('');
  const [job, setJob] = useState<PortJob | null>(null);
  const [recent, setRecent] = useState<PortJob[]>([]);
  const [recentError, setRecentError] = useState('');
  const [loadingExample, setLoadingExample] = useState(false);
  const [openingRun, setOpeningRun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const started = useRef(false);
  const tokens = useRef({ alive: true, example: 0, targets: 0, recent: 0, opening: 0 });
  useEffect(() => { tokens.current.alive = true; return () => { tokens.current.alive = false; }; }, []);
  const selectedTarget = capabilities.targets.find(target => `${target.solver}:${target.device}` === targetId);
  const currentConfig = config ? { ...config, ...(selectedTarget ? { solver: selectedTarget.solver, device: selectedTarget.device } : {}) } : null;
  const currentKey = currentConfig ? canonical(currentConfig) : '';
  const currentValidation = validation?.key === currentKey ? validation.value : null;
  const busy = submitting || openingRun || loadingExample || !!job && ['queued', 'running'].includes(job.status);

  async function loadExample(kind: 'straight' | 'gap') {
    const token = ++tokens.current.example; setLoadingExample(true); setError('');
    try {
      const value = await request<PortConfig>(`/api/ports/example?kind=${kind}`);
      if (tokens.current.alive && tokens.current.example === token) {
        if (value.schema_version !== 3 || value.study_type !== 'waveguide-ports-2d' || value.device_kind !== kind) throw new Error('The example is not the requested 2D port study.');
        setConfig(value); setTargetId('');
      }
    } catch (reason) { if (tokens.current.alive && tokens.current.example === token) setError(`Example: ${(reason as Error).message}`); }
    finally { if (tokens.current.alive && tokens.current.example === token) setLoadingExample(false); }
  }
  async function checkTargets(refresh = false) {
    const token = ++tokens.current.targets; setTargetLoading(true);
    try { const value = await request<Capabilities>(`/api/ports/capabilities${refresh ? '?refresh=true' : ''}`); if (tokens.current.alive && tokens.current.targets === token) setCapabilities(value); }
    catch (reason) { if (tokens.current.alive && tokens.current.targets === token) { setCapabilities({ targets: [] }); setError(`Port targets: ${(reason as Error).message}`); } }
    finally { if (tokens.current.alive && tokens.current.targets === token) setTargetLoading(false); }
  }
  async function loadRecent() {
    const token = ++tokens.current.recent;
    try { const value = await request<PortJob[]>('/api/jobs'); if (tokens.current.alive && tokens.current.recent === token) { setRecent(value.filter(item => item.config?.study_type === 'waveguide-ports-2d')); setRecentError(''); } }
    catch (reason) { if (tokens.current.alive && tokens.current.recent === token) setRecentError((reason as Error).message); }
  }
  useEffect(() => {
    if (!active || started.current) return;
    started.current = true; void loadExample('straight'); void checkTargets(); void loadRecent();
  }, [active]);
  useEffect(() => {
    if (!currentConfig) return;
    let enabled = true; setValidation(null); setValidationError(''); setValidationPending(true);
    const snapshot = currentConfig, key = currentKey;
    const timer = setTimeout(() => {
      request<ValidationResponse>('/api/ports/validate', snapshot).then(value => { if (enabled) setValidation({ key, value }); }).catch(reason => { if (enabled) setValidationError(reason.message); }).finally(() => { if (enabled) setValidationPending(false); });
    }, 250);
    return () => { enabled = false; clearTimeout(timer); };
  }, [currentKey]);
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let enabled = true; let timer: ReturnType<typeof setTimeout>;
    const id = job.id;
    const poll = async () => {
      try {
        const value = await request<PortJob>(`/api/jobs/${id}`);
        if (!enabled) return;
        if (value.id !== id || value.config?.study_type !== 'waveguide-ports-2d') throw new Error('Run status does not match the requested port study.');
        setJob(value);
        if (!['queued', 'running'].includes(value.status)) { void loadRecent(); return; }
      } catch (reason) { if (enabled) setError(`Run status: ${(reason as Error).message}`); }
      if (enabled) timer = setTimeout(poll, 900);
    };
    timer = setTimeout(poll, 500);
    return () => { enabled = false; clearTimeout(timer); };
  }, [job?.id, job?.status]);

  async function run() {
    if (!currentConfig || !currentValidation?.valid || !selectedTarget?.available || busy) return;
    const snapshot: PortConfig = JSON.parse(JSON.stringify(currentConfig));
    setSubmitting(true); setError('');
    try {
      await request<ValidationResponse>('/api/ports/validate', snapshot);
      const created = await request<PortJob>('/api/ports/jobs', snapshot);
      if (tokens.current.alive) { setJob(created); void loadRecent(); }
    } catch (reason) { if (tokens.current.alive) setError(`Measurement: ${(reason as Error).message}`); }
    finally { if (tokens.current.alive) setSubmitting(false); }
  }
  async function cancel() {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    const id = job.id;
    try { const value = await request<PortJob>(`/api/jobs/${id}/cancel`, {}); if (tokens.current.alive) { setJob(old => old?.id === id ? value : old); void loadRecent(); } }
    catch (reason) { if (tokens.current.alive) setError(`Cancel: ${(reason as Error).message}`); }
  }
  async function openRun(id: string) {
    if (!id || busy) return;
    const token = ++tokens.current.opening; setOpeningRun(true); setError('');
    try {
      const value = await request<PortJob>(`/api/jobs/${id}`);
      if (tokens.current.alive && tokens.current.opening === token) {
        if (value.id !== id || value.config?.study_type !== 'waveguide-ports-2d') throw new Error('This saved run is not the selected port study.');
        setJob(value);
      }
    } catch (reason) { if (tokens.current.alive && tokens.current.opening === token) setError(`Open run: ${(reason as Error).message}`); }
    finally { if (tokens.current.alive && tokens.current.opening === token) setOpeningRun(false); }
  }
  async function importConfig(file?: File) {
    if (!file || busy) return;
    const token = ++tokens.current.example;
    setLoadingExample(true); setError('');
    try {
      if (file.size > 100_000) throw new Error('Port configurations must be smaller than 100 KB.');
      const value = JSON.parse(await file.text()) as PortConfig;
      if (value.schema_version !== 3 || value.study_type !== 'waveguide-ports-2d') throw new Error('Select a schema 3 waveguide-ports-2d configuration.');
      await request<ValidationResponse>('/api/ports/validate', value);
      if (tokens.current.alive && tokens.current.example === token) { setConfig(value); setTargetId(''); setError(''); }
    } catch (reason) { if (tokens.current.alive && tokens.current.example === token) setError(`Import: ${(reason as Error).message}`); }
    finally { if (tokens.current.alive && tokens.current.example === token) setLoadingExample(false); if (fileInput.current) fileInput.current.value = ''; }
  }

  const parameters: PortParameter[] = config ? parameterDefinitions.filter(parameter => parameter.key !== 'gap_um' || config.device_kind === 'gap').map(parameter => ({ ...parameter, value: Number.isFinite(config[parameter.key]) ? config[parameter.key] : '' })) : [];
  const targets: PortTarget[] = capabilities.targets.map(target => ({ id: `${target.solver}:${target.device}`, label: `${solverName(target.solver)} · ${target.device === 'gpu' ? 'NVIDIA GPU' : 'CPU'}`, available: target.available, reason: [target.reason, target.preparation].filter(Boolean).join(' ') || 'Availability reported by the port-study service.', device_name: target.device_name, version: target.version }));
  const estimate = currentValidation?.estimate;
  const panelValidation: PortValidation | null = validationError ? { valid: false, summary: validationError } : currentValidation ? {
    valid: currentValidation.valid, summary: currentValidation.valid ? 'Port study validated' : 'Port study is invalid', warnings: currentValidation.warnings,
    source_description: `Fundamental even-y ${currentValidation.layout.field_component} mode pulse; source x = ${currentValidation.layout.source_x_um} µm. The fixed profile is normalized using a separate straight-guide solve.`,
    monitor_description: `Reflection x = ${currentValidation.layout.reflection_x_um} µm; transmission x = ${currentValidation.layout.transmission_x_um} µm; aperture ${currentValidation.layout.monitor_aperture_um} µm. ${currentValidation.layout.dimensions}.`,
    estimate_description: estimate ? `${Number(estimate.xy_cells).toLocaleString()} XY cells · two solves · ${estimate.planning_memory_mib} MiB memory planning allowance. ${estimate.note || ''}` : undefined,
  } : null;
  const logs = Array.isArray(job?.logs) ? job.logs.join('\n') : job?.logs || '';
  const currentDiffers = !!job && currentKey !== canonical(job.config);
  const output = <>
    {currentConfig && currentValidation?.valid && <PortGeometry config={currentConfig} layout={currentValidation.layout}/>}
    {job ? <div className="port-job" data-run-id={job.id}><div className="port-job-facts"><span>Run <strong>{job.id.slice(0, 10)}</strong></span><span>Saved study <strong>{job.config.device_kind === 'straight' ? 'Straight guide' : `Gap ${job.config.gap_um} µm`}</strong></span><span>Requested <strong>{solverName(job.config.solver)} · {job.config.device.toUpperCase()}</strong></span><span>Status <strong>{job.status}</strong></span></div>
      {job.status === 'completed' && <><PortSpectrum key={job.id} jobId={job.id}/><div className="port-result-actions"><a className="button primary compact" href={`/api/jobs/${job.id}/export`}>Export port results .zip</a><button className="button secondary compact" onClick={() => saveConfig(job.config, `port-study-${job.id.slice(0, 8)}.json`)}>Save run configuration</button></div><RunPassport key={job.id} jobId={job.id}/></>}
      {job.status === 'queued' && <p className="port-running" role="status">Waiting for the local worker. Both solves use the selected execution target.</p>}
      {job.status === 'running' && <p className="port-running" role="status">{job.port_progress?.message || 'Local port study is running. Read the solver log for preparation and time-stepping details.'}</p>}
      {job.status === 'cancelled' && <p className="port-running">Measurement cancelled. No completed normalized spectrum is claimed.</p>}
      {job.error && <div className="frame-error" role="alert">{job.error}</div>}
      <details className="port-logs" open={job.status === 'running' || job.status === 'failed'}><summary>Port solver log</summary><pre>{logs || 'No worker output recorded yet.'}</pre></details>
    </div> : <div className="port-empty"><span aria-hidden="true">∿</span><h3>Measure a guide against its reference.</h3><p>Choose a target to run the matched reference and device. Actual spectral samples will appear here after the local worker completes.</p></div>}
  </>;
  return <PortStudyPanel templates={templates} templateId={config?.device_kind} parameters={parameters} targets={targets} selectedTargetId={targetId} targetLoading={targetLoading} validation={panelValidation} validationPending={validationPending} recommendation={capabilities.recommendation?.reason} stages={jobStages(job)} busy={busy}
    immutableRunNote={currentDiffers ? 'The editor differs from this saved run. Its spectrum and exports retain the original configuration; changing these inputs does not alter the recorded result.' : undefined}
    error={error} onTemplateChange={kind => { if (kind === 'straight' || kind === 'gap') void loadExample(kind); }} onParameterChange={(key, value) => setConfig(previous => previous ? { ...previous, [key]: value === '' ? NaN : value } : previous)} onTargetChange={setTargetId} onRun={() => void run()} onCancel={job && ['queued', 'running'].includes(job.status) ? () => void cancel() : undefined} onRefreshTargets={() => void checkTargets(true)} onOpenExample={() => void loadExample('straight')} onDismissError={() => setError('')} results={output}
    configurationActions={<div className="port-config-actions"><button onClick={() => currentConfig && saveConfig(currentConfig)} disabled={!currentValidation?.valid || !selectedTarget?.available || busy}>Export study JSON</button><button onClick={() => fileInput.current?.click()} disabled={busy}>Import study JSON</button><CodeExport value={currentConfig} disabled={!currentValidation?.valid||!selectedTarget?.available||busy}/><input ref={fileInput} type="file" aria-label="Import port configuration" accept=".json,application/json" hidden onChange={event => void importConfig(event.target.files?.[0])}/></div>}
    outputNavigation={<div className="port-recent"><label>Saved measurement<select aria-label="Recent port run" value={job?.id || ''} disabled={busy} onChange={event => void openRun(event.target.value)}><option value="">Choose a recorded port study…</option>{[...(job && !recent.some(item => item.id === job.id) ? [job] : []), ...recent].map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.config.device_kind} · {solverName(item.config.solver)} {item.config.device.toUpperCase()} · {item.status}</option>)}</select></label><button onClick={() => void loadRecent()}>Refresh runs</button>{recentError && <p role="alert">Saved runs unavailable: {recentError}</p>}</div>}/>
  ;
}
