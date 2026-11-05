import { useEffect, useState } from 'react';
import { request } from './api';

type Pole = {
  longitudinal_order: number;
  frequency_MHz: number;
  published_frequency_MHz: number;
  quality_factor: number;
  published_quality_factor: number;
};
type Report = { case_id: string; source_url: string; solver: string; reference_comparison: Pole[] };

export function PassiveCavityReference() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open || report) return;
    let active = true;
    setError('');
    request<Report>('/api/cavity-benchmark').then(value => {
      if (value.case_id !== 'comsol-fabry-perot' || value.reference_comparison?.length !== 4) {
        throw new Error('The four measured cavity modes are unavailable.');
      }
      if (active) setReport(value);
    }).catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [open, report]);
  return <details className="benchmark-report benchmark-comparisons" aria-label="Passive cavity reference"
    onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>Passive cavity frequencies and Q · COMSOL reference</summary>
    <div className="benchmark-report-body">
      {error && <p role="alert">{error}</p>}
      {!report && !error && <p>Reading measured cavity modes…</p>}
      {report && <article>
        <h3>Fabry–Pérot dielectric slab</h3>
        <p>Meep recorded the decaying electric field after the source switched off. The fitted modes belong to a one-metre, index-four slab in air. Q measures how slowly each mode loses its stored energy.</p>
        <div style={{overflowX:'auto'}}><table className="benchmark-ref-table">
          <caption>Meep ringdown and published COMSOL eigenfrequencies</caption>
          <thead><tr><th>Mode</th><th>Meep (MHz)</th><th>COMSOL (MHz)</th><th>Meep Q</th><th>COMSOL Q</th></tr></thead>
          <tbody>{report.reference_comparison.map(mode =>
            <tr key={mode.longitudinal_order}><th scope="row">{mode.longitudinal_order}</th>
              <td>{mode.frequency_MHz.toFixed(4)}</td><td>{mode.published_frequency_MHz}</td>
              <td>{mode.quality_factor.toFixed(4)}</td><td>{mode.published_quality_factor}</td></tr>)}</tbody>
        </table></div>
        <p><a href={report.source_url} target="_blank" rel="noreferrer">COMSOL model, Table 1</a></p>
        <a href="/api/cavity-benchmark" download="passive-cavity-reference.json">Download mode fits and refinement checks</a>
        <p>Reproduce locally with <code>photonics-workbench benchmark-cavity --output cavity-study</code>.</p>
      </article>}
    </div>
  </details>;
}
