import { BeamZReference } from './BeamZReference';
import { PassiveCavityReference } from './PassiveCavityReference';
import ComponentComparisons from './ComponentComparisons';
import { useEffect, useState } from 'react';
import './publicBenchmarks.css';

type Refinement = { resolution_cells_per_um: number; computed_neff: number; relative_error?: number; grid_shape?: number[] };
type BenchmarkCase = {
  id: string; title: string; source_url: string; reference_kind: string;
  reference_neff: number | null; computed_neff: number | null; relative_error: number | null;
  absolute_error?: number; resolution_cells_per_um?: number; status?: string;
  refinements?: Refinement[]; last_refinement_relative_change?: number;
  domain_check?: { domain_scale?: number; computed_neff?: number; relative_change?: number; note?: string };
  physics?: Record<string, unknown>; mode_info?: Record<string, unknown>;
  reproduction_command?: string; notes?: string[];
};
type BenchmarkReport = {
  schema_version: 1; scope: 'mode-benchmarks'; status: 'available' | 'not_run';
  generated_at?: string; solver?: string; versions?: Record<string, string | number>;
  cases: BenchmarkCase[]; notes?: string[]; reproduction_command?: string; report_sha256?: string;
};
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const neff = (value: unknown) => finite(value) ? String(Number(value.toFixed(7))) : 'Not available';
const percent = (value: unknown) => finite(value) ? `${(100 * value).toPrecision(4)}%` : 'Not available';
function sourceLink(value: string) {
  try { const url = new URL(value); return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : undefined; }
  catch { return undefined; }
}
function referenceLabel(kind: string) {
  if (kind === 'analytic_reference') return 'Analytic reference';
  if (kind === 'published_solver_value') return 'Published solver value';
  return 'Reference type unclassified';
}
function parseReport(value: unknown): BenchmarkReport {
  const report = value as BenchmarkReport;
  if (!report || report.schema_version !== 1 || report.scope !== 'mode-benchmarks'
    || !['available', 'not_run'].includes(report.status) || !Array.isArray(report.cases)
    || report.cases.some(item => !item || typeof item.id !== 'string' || typeof item.title !== 'string'
      || typeof item.reference_kind !== 'string' || typeof item.source_url !== 'string')) {
    throw new Error('The local benchmark report has an unsupported format. No numerical claims are displayed.');
  }
  return report;
}
function downloadReport(report: BenchmarkReport) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2) + '\n'], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = 'isophotonix-public-mode-benchmarks.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function EvidenceCard({ item, solver }: { item: BenchmarkCase; solver: string }) {
  const source = sourceLink(item.source_url);
  const computed = finite(item.computed_neff);
  return <article className="benchmark-card" aria-label={item.title} data-benchmark-id={item.id}>
    <div className="benchmark-card-heading"><span className="benchmark-reference-kind">{referenceLabel(item.reference_kind)}</span><span className="benchmark-measure">n<sub>eff</sub></span></div>
    <h3>{item.title}</h3>
    <p className="benchmark-reference-note">{item.reference_kind === 'analytic_reference'
      ? 'Compared with an analytic reference for the published example; no commercial solver was run.'
      : item.reference_kind === 'published_solver_value'
        ? 'Compared with a scalar reported in the linked solver example; no commercial solver was run.'
        : 'Inspect the evidence report for this reference’s origin before interpreting the comparison.'}</p>
    <dl className="benchmark-values">
      <div><dt>Local {solver} n<sub>eff</sub></dt><dd>{computed ? neff(item.computed_neff) : 'Not run'}</dd></div>
      <div><dt>{item.reference_kind === 'analytic_reference' ? 'Analytic' : 'Published'} n<sub>eff</sub></dt><dd>{neff(item.reference_neff)}</dd></div>
      <div className="benchmark-difference"><dt>Relative difference</dt><dd>{computed ? percent(item.relative_error) : 'Not available'}</dd></div>
    </dl>
    {finite(item.last_refinement_relative_change) && <p className="benchmark-mesh-note">Last mesh refinement change: <strong>{percent(item.last_refinement_relative_change)}</strong>. Convergence is not certified.</p>}
    <p className="benchmark-card-scope">Effective-index scalar only. This does not validate driven 3D fields, ports, transmission, or a complete device.</p>
    <div className="benchmark-card-meta"><span>{finite(item.resolution_cells_per_um) ? `${item.resolution_cells_per_um} cells/µm` : 'See numerical settings'}</span><span>{computed ? 'Computed · not certified' : 'No computed result'}</span></div>
    {source && <a className="benchmark-source" href={source} target="_blank" rel="noreferrer">Read the official example <span aria-hidden="true">↗</span></a>}
    <details className="benchmark-case-details"><summary>Inspect numerical evidence</summary>
      {!!item.refinements?.length && <div className="benchmark-refinements"><table><caption>Recorded mesh refinements</caption><thead><tr><th scope="col">Cells/µm</th><th scope="col">Local n<sub>eff</sub></th><th scope="col">Difference</th></tr></thead><tbody>{item.refinements.map((row, index) => <tr key={index}><td>{row.resolution_cells_per_um}</td><td>{neff(row.computed_neff)}</td><td>{percent(row.relative_error)}</td></tr>)}</tbody></table></div>}
      {item.domain_check && <p>Domain-size check{finite(item.domain_check.domain_scale) ? ` (${item.domain_check.domain_scale}×)` : ''}: n<sub>eff</sub> {neff(item.domain_check.computed_neff)}; relative change {percent(item.domain_check.relative_change)}.</p>}
      {item.domain_check?.note && <p>{item.domain_check.note}</p>}
      {item.notes?.map((note, index) => <p key={index}>{note}</p>)}
      {item.physics && <details><summary>Physics and geometry</summary><pre>{JSON.stringify(item.physics, null, 2)}</pre></details>}
      {item.mode_info && <details><summary>Mode identification</summary><pre>{JSON.stringify(item.mode_info, null, 2)}</pre></details>}
      {item.reproduction_command && <><span className="benchmark-command-label">REPRODUCE THIS CASE LOCALLY</span><code className="benchmark-command">{item.reproduction_command}</code></>}
    </details>
  </article>;
}

export function PublicBenchmarks() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoading(true); setError(''); setReport(null);
    (async () => {
      try {
        const response = await fetch('/api/benchmarks', { signal: controller.signal });
        if (!response.ok) throw new Error(`The local benchmark report could not be loaded (HTTP ${response.status}).`);
        const value = parseReport(await response.json());
        if (active) setReport(value);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : 'The local benchmark report could not be loaded.');
      } finally { if (active) setLoading(false); }
    })();
    return () => { active = false; controller.abort(); };
  }, [refresh]);

  const available = report?.status === 'available' && report.cases.length > 0;
  return <section className="panel public-benchmarks" aria-label="Public mode benchmark evidence" aria-busy={loading}>
    <div className="benchmark-heading"><div><span className="eyebrow">TRACEABLE NUMERICAL EVIDENCE</span><h2>Public mode benchmarks</h2><p>Inspect recorded results and their references. Refreshing reads saved evidence; use the reproduction command to run again.</p></div>
      <div className="benchmark-actions"><button className="button secondary compact" onClick={() => setRefresh(value => value + 1)} disabled={loading}>Refresh evidence</button>{report && <button className="button secondary compact" onClick={() => downloadReport(report)}>Download evidence JSON</button>}</div></div>
    <div className="benchmark-scope"><span aria-hidden="true">⌑</span><p>These are independent waveguide mode checks using local MPB. They do not certify the 3D FDTD study above or establish general agreement with a commercial solver.</p></div>
    {loading && <p className="benchmark-state" role="status">Loading saved local benchmark evidence…</p>}
    {error && <p className="benchmark-state benchmark-error" role="alert">{error} Refresh after the local report is available.</p>}
    {!loading && !error && report && <>
      {available ? <div className="benchmark-cards">{report.cases.map(item => <EvidenceCard key={item.id} item={item} solver={report.solver || 'solver'}/>)}</div>
        : <div className="benchmark-state"><strong>Benchmarks not run</strong><p>No computed benchmark evidence is available from this installation. A saved report will appear here after the local benchmark command completes.</p></div>}
      <details className="benchmark-report"><summary>Report, reproduction, and provenance{report.generated_at ? <span>{report.generated_at}</span> : null}</summary><div className="benchmark-report-body">
        {report.notes?.map((note, index) => <p key={index}>{note}</p>)}
        {report.reproduction_command && <><span className="benchmark-command-label">RUN FROM THE PROJECT DIRECTORY</span><code className="benchmark-command">{report.reproduction_command}</code></>}
        {report.versions && <div className="benchmark-versions">{Object.entries(report.versions).map(([name, version]) => <span key={name}>{name} <strong>{version}</strong></span>)}</div>}
        {report.report_sha256 && <p className="benchmark-hash">Report SHA-256 <code>{report.report_sha256}</code></p>}
        <details><summary>Complete API evidence JSON</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
      </div></details>
    </>}
    <ComponentComparisons/>
    <PassiveCavityReference/>
    <BeamZReference/>
  </section>;
}
