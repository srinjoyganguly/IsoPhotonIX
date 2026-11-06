import { useEffect, useState } from 'react';
import { request } from './api';

type Stage = {
  receipt: string; stage: string; geometry: string; origin: string; cells: number | null; steps: number | null; steps_lower_bound: number|null;purpose:string;
  mesh_cells_per_um: number | null; synchronized_step_seconds: number | null;
  cuda_event_elapsed_seconds: number | null; kernel_active_seconds: number | null; compile_seconds: number | null; historical_elapsed_seconds: number | null;
  historical_basis: string | null; kernel_timing_note: string; device_kind: string | null;
};
const seconds = (value: number | null) => value === null ? '—' : value.toLocaleString(undefined,{maximumFractionDigits:3})+' s';

export function GpuPerformance({preset, job}:{preset:string; job?:string}) {
  const [rows,setRows]=useState<Stage[]>([]), [error,setError]=useState(''), [open,setOpen]=useState(false);
  const [receipt,setReceipt]=useState('');
  useEffect(()=>{
    if (!open) return;
    let live=true;
    const endpoint=job ? '/api/calibrations/jobs/'+job+'/performance' : '/api/calibrations/performance?case='+encodeURIComponent(preset);
    request<Stage[]>(endpoint).then(result=>{
      if (!live) return;
      setRows(result);
      const latest=[...result].sort((a,b)=>b.receipt.localeCompare(a.receipt))[0];
      setReceipt(latest?.receipt||'');
    }).catch(e=>{if(live)setError(e.message);});
    return ()=>{live=false;};
  },[preset,job,open]);
  const selected=rows.filter(row=>row.receipt===receipt);
  const receipts=[...new Set(rows.map(row=>row.receipt))].sort().reverse();
  return <details className="cal-runtime" onToggle={e=>setOpen(e.currentTarget.open)}>
    <summary>{job?'GPU timing for this run':'GPU timing records'}</summary>
    {error && <p role="alert">{error}</p>}
    {rows.length>0 && <>{!job && <label>Timing record<select value={receipt} onChange={e=>setReceipt(e.target.value)}>
      {receipts.map(name=><option key={name} value={name}>{name}</option>)}</select></label>}
      <p>{selected[0]?.device_kind} · {selected[0]?.mesh_cells_per_um} cells/µm · {selected[0]?.purpose}</p>
      <div className="cal-time-scroll"><table><caption>Measured time by solver stage</caption>
        <thead><tr>{['Stage','Cells','Steps','Compile','Stepping','GPU clock','Earlier timer'].map(h=><th key={h}>{h}</th>)}</tr></thead>
        <tbody>{selected.map(row=><tr key={row.stage}><th>{row.stage}{row.origin==='reused_verified_receipt'?' (cached)':''}</th>
          <td>{row.cells?.toLocaleString()||'—'}</td><td>{row.steps?.toLocaleString() ?? (row.steps_lower_bound?'≥ '+row.steps_lower_bound.toLocaleString():'—')}</td><td>{seconds(row.compile_seconds)}</td>
          <td>{seconds(row.synchronized_step_seconds)}</td><td>{seconds(row.cuda_event_elapsed_seconds)}</td>
          <td title={row.historical_basis||undefined}>{seconds(row.historical_elapsed_seconds)}</td></tr>)}</tbody>
      </table></div><p>Stepping measures elapsed time waiting for the GPU. The GPU clock brackets the same batches on the device, including launch gaps. A dash means that timer was unavailable.</p>
      {[...new Set(selected.map(row=>row.historical_basis).filter(Boolean))].map(note=><p key={note}>Earlier timer covered {note}.</p>)}
    </>}
    {!rows.length && open && !error && <p>No recorded GPU stage timings for this selection.</p>}
    {!job && <a href={'/api/calibrations/performance.csv?case='+encodeURIComponent(preset)}>Download timing CSV</a>}
  </details>;
}
