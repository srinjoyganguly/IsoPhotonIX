import { useEffect, useState } from 'react';
import { request } from './api';

type Mesh = { points_per_material_wavelength: number; phase_velocity_m_s: number; continuum_relative_error: number };
type Receipt = { actual_device: string; release: string; propagation: Mesh[] };
type Report = { cpu: Receipt; gpu: Receipt };

export function BeamZReference() {
  const [open, setOpen] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open || report) return;
    let active = true;
    setError('');
    request<Report>('/api/beamz-benchmark').then(value => {
      if (active) setReport(value);
    }).catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [open, report]);
  return <details className="benchmark-report benchmark-comparisons" aria-label="BeamZ propagation reference"
    onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>BeamZ propagation · CPU and NVIDIA</summary>
    <div className="benchmark-report-body">
      {error && <p role="alert">{error}</p>}
      {!report && !error && <p>Reading local BeamZ reference runs…</p>}
      {report && <article>
        <h3>Phase velocity in a uniform dielectric</h3>
        <p>BeamZ {report.cpu.release} measured the propagation of a plane wave on three meshes. The reference velocity is the speed of light divided by the material index.</p>
        <div style={{overflowX:'auto'}}><table className="benchmark-ref-table">
          <thead><tr><th>Device</th><th>Cells per material wavelength</th><th>Velocity (m/s)</th><th>Difference from theory</th></tr></thead>
          <tbody>{[report.cpu, report.gpu].flatMap(item => item.propagation.map(mesh =>
            <tr key={item.actual_device + mesh.points_per_material_wavelength}><th scope="row">{item.actual_device}</th>
              <td>{mesh.points_per_material_wavelength}</td><td>{mesh.phase_velocity_m_s.toExponential(5)}</td>
              <td>{(100 * mesh.continuum_relative_error).toFixed(4)}%</td></tr>))}</tbody>
        </table></div>
        <a href="/api/beamz-benchmark" download="beamz-references.json">Download field checks and CPU slab-mode comparison</a>
        <p>Reproduce locally with <code>isophotonix benchmark-beamz --device cpu --output beamz-study</code>.</p>
      </article>}
    </div>
  </details>;
}
