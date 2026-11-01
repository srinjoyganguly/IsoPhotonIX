import { useEffect, useRef, useState } from 'react';
import { FieldCanvas, type Field } from './FieldCanvas';
import { request } from './api';

type FrameInfo = {count:number;times_fs:number[];global_peak_abs:number};
type Frame = {field:number[][];time_fs:number;index:number};

export function EvolutionPlayer({jobId,data,active=true}:{jobId:string;data:Field;active?:boolean}) {
  const [info,setInfo]=useState<FrameInfo|null>(null);
  const [frame,setFrame]=useState<Frame|null>(null);
  const [index,setIndex]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState('');
  const [retry,setRetry]=useState(0);
  const cache=useRef(new Map<number,Frame>());

  useEffect(()=>{
    let active=true;setLoading(true);setError('');
    request<FrameInfo>(`/api/jobs/${jobId}/frames`).then(value=>{
      if(!active)return;
      if(value.count<1 || value.times_fs.length!==value.count || !Number.isFinite(value.global_peak_abs) || value.global_peak_abs<=0)throw new Error('Saved frame metadata is incomplete.');
      setInfo(value);
    }).catch(e=>{if(active){setError(e.message);setLoading(false);setPlaying(false);}});
    return()=>{active=false;};
  },[jobId,retry]);

  useEffect(()=>{
    if(!info)return;
    let active=true;setError('');
    const cached=cache.current.get(index);
    if(cached){setFrame(cached);setLoading(false);return;}
    setLoading(true);
    request<Frame>(`/api/jobs/${jobId}/frames/${index}`).then(value=>{
      if(!active)return;
      if(value.index!==index || !Number.isFinite(value.time_fs) || value.field.length!==data.y_um.length || value.field.some(row=>row.length!==data.x_um.length || row.some(x=>!Number.isFinite(x))))throw new Error('Saved frame does not match this run’s numerical grid.');
      cache.current.set(index,value);setFrame(value);setLoading(false);
    }).catch(e=>{if(active){setError(e.message);setLoading(false);setPlaying(false);}});
    return()=>{active=false;};
  },[jobId,index,info,retry,data.x_um.length,data.y_um.length]);

  useEffect(()=>{
    if(!active){setPlaying(false);return;}
    if(!playing || !info || !frame || loading || error || frame.index!==index)return;
    if(index>=info.count-1){setPlaying(false);return;}
    const timer=setTimeout(()=>setIndex(current=>current+1),300);
    return()=>clearTimeout(timer);
  },[active,playing,info,frame,index,loading,error]);

  function toggle(){
    if(!info)return;
    if(!playing && index===info.count-1)setIndex(0);
    setPlaying(current=>!current);
  }

  return <div className="evolution-player">
    {frame&&info?<FieldCanvas data={{...data,field:frame.field}} view="field" fixedPeak={info.global_peak_abs}/>:<div className="frame-empty">{loading?<><span className="spinner"/>Loading recorded solver frames…</>:<>Final fields remain available in the Ez field tab.</>}</div>}
    {error&&<div className="frame-error" role="alert"><span>Time evolution unavailable: {error}</span><button onClick={()=>setRetry(value=>value+1)}>Retry</button></div>}
    {info&&<div className="frame-controls">
      <button className="frame-play" onClick={toggle} disabled={!!error || !frame} aria-label={playing?'Pause playback':'Play recorded frames'}>{playing?'Ⅱ Pause':'▶ Play'}</button>
      <div className="frame-scrubber"><label htmlFor={`frame-${jobId}`}>Recorded frame <strong>{frame?frame.index+1:'–'} / {info.count}</strong></label><input id={`frame-${jobId}`} aria-label="Recorded frame" type="range" min="0" max={info.count-1} step="1" value={index} onChange={e=>{setPlaying(false);setIndex(Number(e.target.value));}}/></div>
      <div className="frame-time"><strong>{frame?frame.time_fs.toFixed(2):'–'} <span>fs</span></strong><small>{loading?'Loading next frame…':'Simulated time'}</small></div>
      <p className="frame-disclosure">{loading?'Loading a saved numerical frame. ':'Each frame is recorded solver output. '}No interpolated fields. Scale fixed across all frames.</p>
    </div>}
  </div>;
}
