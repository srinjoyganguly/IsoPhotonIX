import { useState } from 'react';
import { request } from './api';
import type { MaterialRecord } from './geometryModel';
import './thinFilm.css';

type Sample = { wavelength_um:number; n:number; k:number; R:number; T:number; A:number; r:[number,number]; t:[number,number] };
type Spectrum = { samples: Sample[] };
const curves = [{key:'R',name:'Reflected'}, {key:'T',name:'Transmitted'}, {key:'A',name:'Absorbed'}] as const;
const number = (value:number) => Number(value.toPrecision(6)).toString();
const phasor = ([re,im]:[number,number]) => `${number(re)} ${im < 0 ? '−' : '+'} ${number(Math.abs(im))}i`;

export function ThinFilm({material, thickness}:{material:MaterialRecord;thickness:number}) {
  const bounds = material.model === 'tabulated-nk' ? [material.wavelength_um[0],material.wavelength_um.at(-1)!]
    : [Math.max(.001,material.reference_wavelength_um*.9), Math.min(1000,material.reference_wavelength_um*1.1)];
  const [start,setStart] = useState(String(bounds[0])), [end,setEnd] = useState(String(bounds[1]));
  const [incident,setIncident] = useState('1'), [substrate,setSubstrate] = useState('1.45');
  const [samples,setSamples] = useState('51'), [selected,setSelected] = useState(0);
  const [result,setResult] = useState<{key:string;spectrum:Spectrum;config:string}|null>(null);
  const [error,setError] = useState(''), [busy,setBusy] = useState(false);
  const signature = JSON.stringify([material,thickness,start,end,incident,substrate,samples]);
  const current = result?.key === signature ? result : null;
  const row = current?.spectrum.samples[selected];

  async function calculate() {
    setBusy(true); setError('');
    try {
      if (![start,end,incident,substrate,samples].every(value=>value.trim() && Number.isFinite(Number(value)))) throw new Error('Enter finite indices, wavelength limits and a sample count.');
      const lo=Number(start), hi=Number(end), count=Number(samples);
      if (!Number.isInteger(count) || count < 1 || count > 201 || (count===1 ? hi!==lo : hi<=lo)) throw new Error('Choose 2–201 samples in an increasing band, or 1 sample with equal endpoints.');
      const waves=Array.from({length:count}, (_,i)=>i===0 ? lo : i===count-1 ? hi : lo+(hi-lo)*i/(count-1));
      const config={material,thickness_um:thickness,incident_index:Number(incident),substrate_index:Number(substrate),wavelengths_um:waves};
      const spectrum=await request<Spectrum>('/api/materials/film', config);
      setResult({key:signature,spectrum,config:JSON.stringify(config)});
      setSelected(Math.floor(count/2));
    } catch(reason) {setResult(null);setError((reason as Error).message);}
    finally {setBusy(false);}
  }
  async function download() {
    if (!current) return;
    setBusy(true); setError('');
    try {
      const response=await fetch('/api/materials/film/export',{method:'POST',headers:{'Content-Type':'application/json'},body:current.config});
      if (!response.ok) throw new Error((await response.json()).detail ?? 'Film export failed.');
      const url=URL.createObjectURL(await response.blob());
      const link=document.createElement('a');link.href=url;link.download='planar-film.zip';link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(reason) {setError((reason as Error).message);}
    finally {setBusy(false);}
  }
  return <details className="film-study"><summary>Planar film spectrum</summary>
    <p>{material.name} · {number(thickness)} µm thick. A plane wave enters along Z. The film and surrounding media extend infinitely in XY.</p>
    <div className="film-inputs">{([
      ['Incident index',incident,setIncident],['Substrate index',substrate,setSubstrate],
      ['Start λ (µm)',start,setStart],['End λ (µm)',end,setEnd],['Samples',samples,setSamples],
    ] as const).map(([label,value,update])=><label key={String(label)}>{String(label)}
      <input aria-label={'Film '+label} type="number" step={label==='Samples' ? 1 : 'any'} value={String(value)}
        onChange={event=>{update(event.target.value);setError('');}}/></label>)}</div>
    <button className="button primary compact" onClick={calculate} disabled={busy}>Calculate film spectrum</button>
    {error && <p role="alert" className="library-error">{error}</p>}
    {current && row && <div className="film-output" role="region" aria-label="Film spectrum results">
      <svg viewBox="0 0 220 111" role="img" aria-label="Reflected, transmitted and absorbed fractions versus vacuum wavelength">
        {[0,.5,1].map(t=><g key={t}><line x1="24" x2="212" y1={88-78*t} y2={88-78*t}/><text x="20" y={91-78*t} textAnchor="end">{number(t)}</text></g>)}
        {curves.map(({key})=>{
          const y=(s:Sample)=>88-78*s[key];
          const points=current.spectrum.samples.map((s,i)=>[24+188*i/Math.max(1,current.spectrum.samples.length-1),y(s)].join(',')).join(' ');
          return <g className={'film-'+key} key={key}><polyline points={points}/><circle cx={24+188*selected/Math.max(1,current.spectrum.samples.length-1)} cy={y(row)} r="2"/></g>;
        })}
        <text x="24" y="106">{number(current.spectrum.samples[0].wavelength_um)} µm</text>
        {current.spectrum.samples.length>1 && <text x="212" y="106" textAnchor="end">{number(current.spectrum.samples.at(-1)!.wavelength_um)} µm</text>}
      </svg>
      <label className="film-wavelength">Inspect λ · {number(row.wavelength_um)} µm
        <input type="range" aria-label="Film sample" min="0" max={current.spectrum.samples.length-1} value={selected}
          disabled={current.spectrum.samples.length===1} onChange={event=>setSelected(Number(event.target.value))}/></label>
      <dl>{curves.map(({key,name})=><div key={key} className={'film-'+key}><dt>{name}</dt><dd>{number(100*row[key])}%</dd></div>)}
        <div><dt>Entrance r</dt><dd>{phasor(row.r)}</dd></div><div><dt>Exit t</dt><dd>{phasor(row.t)}</dd></div>
      </dl>
      <button className="button secondary compact" onClick={download} disabled={busy}>Export film Python + notebook</button>
    </div>}
  </details>;
}
