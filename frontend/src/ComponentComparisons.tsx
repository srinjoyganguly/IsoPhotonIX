import { useEffect, useState } from 'react';
import { request } from './api';

type Metric = {key:string; label:string; unit:string; reference:number; computed:number; relative_error:number; last_mesh_change:number; larger_domain_change:number};
type Case = {id:string; title:string; vendor:string; source_url:string; source_kind:string; metrics:Metric[]; limitations:string[]};
type Report = {schema_version:number; generated_at:string; solver:string; cases:Case[]};
const number = (n:number) => n.toLocaleString('en-GB',{maximumSignificantDigits:8});

export default function ComponentComparisons() {
  const [report,setReport]=useState<Report|null>(null);
  const [error,setError]=useState('');
  useEffect(()=>{
    let active=true;
    request<Report>('/api/component-benchmarks').then(data=>{
      if(data.schema_version!==1 || !Array.isArray(data.cases))throw new Error('Unrecognized comparison report.');
      if(active)setReport(data);
    }).catch(reason=>{if(active)setError(String(reason.message));});
    return ()=>{active=false;};
  },[]);
  return <details className="benchmark-report benchmark-comparisons" aria-label="Published component comparisons">
    <summary>Compare with Lumerical and Tidy3D examples</summary>
    <div className="benchmark-report-body">
      {error && <p role="alert">{error}</p>}
      {!report && !error && <p>Reading local comparison data…</p>}
      {report && <><p>Published example values and independent local MPB calculations. Each row compares the named observable.</p>
        {report.cases.map(item=><article key={item.id}>
          <h3>{item.title}</h3><p>{item.source_kind} <a href={item.source_url} target="_blank" rel="noreferrer">{item.vendor} example</a></p>
          <div style={{overflowX:'auto'}}><table className="benchmark-ref-table"><caption>Dimensionless indices, lengths in µm and areas in µm²</caption>
            <thead><tr><th>Observable</th><th>Published</th><th>Local MPB</th><th>Relative difference</th></tr></thead>
            <tbody>{item.metrics.map(m=><tr key={m.key}><th scope="row">{m.label} {m.unit!=='1' && m.unit}</th>
              <td>{number(m.reference)}</td><td>{number(m.computed)}</td><td>{number(100*m.relative_error)}%</td></tr>)}</tbody></table></div>
          <details><summary>Model and numerical limits</summary>{item.limitations.map((note,i)=><p key={i}>{note}</p>)}</details>
        </article>)}
        <a href="/api/component-benchmarks" download="component-comparisons.json">Download values, sources and refinement evidence</a></>}
    </div>
  </details>;
}
