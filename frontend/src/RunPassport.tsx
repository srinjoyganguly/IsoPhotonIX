import {useEffect,useState} from 'react';
import {request} from './api';

type Passport = {
  run_id:string;solver:string;requested_device:string;actual_device:string;device_name?:string;
  versions:Record<string,string>;source:Record<string,unknown>;boundaries:unknown;
  numerical:Record<string,unknown>;checks:{label:string;passed:boolean;detail:string}[];
  files:{name:string;sha256:string;size_bytes:number}[];reproduce_command:string;
  reproduce:{command:string;notes:string[]};source_assumptions:string[];limitations:string[];
};
const text=(value:unknown)=>typeof value==='string'?value:JSON.stringify(value);

export function RunPassport({jobId}:{jobId:string}) {
  const [passport,setPassport]=useState<Passport|null>(null);
  const [error,setError]=useState('');
  const [retry,setRetry]=useState(0);
  useEffect(()=>{
    let active=true;setError('');
    request<Passport>(`/api/jobs/${jobId}/passport`).then(value=>{if(active)setPassport(value);}).catch(e=>{if(active)setError(e.message);});
    return()=>{active=false;};
  },[jobId,retry]);
  const solverKey=passport?.solver.toLowerCase();
  const solverLabel=solverKey==='meep'?'Meep':solverKey==='fdtdx'?'FDTDX':passport?.solver;
  return <details className="run-passport">
    <summary><span className="passport-symbol">⌑</span><span><strong>Run passport</strong><small>What ran, where, and how to reproduce it</small></span><span className="passport-tag">{passport?'INSPECTABLE':'PROVENANCE'}</span><span className="passport-chevron">⌄</span></summary>
    {error?<div className="frame-error" role="alert"><span>Passport could not be loaded: {error}</span><button onClick={()=>setRetry(value=>value+1)}>Retry</button></div>:passport?<div className="passport-body">
      <div className="passport-facts"><div><span>SOLVER</span><strong>{solverLabel} {passport.versions[solverKey || ''] ?? passport.versions[passport.solver] ?? ''}</strong></div><div><span>ACTUAL EXECUTION</span><strong>{passport.device_name || passport.actual_device}</strong></div></div>
      <div className="passport-checks">{passport.checks.map((check,i)=><span key={i} className={check.passed?'passed':'check-failed'} title={check.detail}>{check.passed?'✓':'!'} {check.label}</span>)}</div>
      <div className="passport-section"><h3>THE PHYSICAL MODEL</h3><p>{text(passport.source.type || passport.source)}</p><p>{text(passport.boundaries)}</p><ul>{[...(passport.source_assumptions || []),...passport.limitations].map((item,i)=><li key={i}>{item}</li>)}</ul></div>
      <div className="passport-section"><h3>FILE FINGERPRINTS · SHA-256</h3><p className="passport-help">Compare these hashes to check that a configuration or result file is unchanged.</p>{passport.files.filter(file=>['config.json','arrays.npz','spectra.npz','spectrum.csv','sparameters.npz','sparameters.csv','port-modes.npz','implementation.zip'].includes(file.name)).map(file=><div className="passport-file" key={file.name}><span>{file.name}<small>{file.size_bytes.toLocaleString()} bytes</small></span><code>{file.sha256}</code></div>)}</div>
      <div className="passport-section"><h3>REPRODUCE THIS RUN</h3><code className="reproduce-command">{passport.reproduce_command}</code>{(passport.reproduce?.notes || []).map((note,i)=><p className="passport-help" key={i}>{note}</p>)}</div>
      <a className="button primary compact" href={`/api/jobs/${jobId}/reproduce`}>Export run Python + notebook</a><p className="passport-help">Includes this exact configuration, saved reference arrays, and a replay comparison. {passport.files.some(file=>file.name==='implementation.zip')?'Original numerical source is preserved from execution.':'This older run uses export-time code; replay verifies compatibility.'} Runs locally after installing the recorded dependencies.</p>
      <div className="passport-versions">{Object.entries(passport.versions).map(([key,value])=><span key={key}>{key} <strong>{value}</strong></span>)}</div>
      <p className="passport-export-note">The results ZIP includes a standalone HTML passport and manifest.</p>
    </div>:<div className="passport-loading"><span className="spinner tiny"/>Reading the saved run’s provenance…</div>}
  </details>;
}
