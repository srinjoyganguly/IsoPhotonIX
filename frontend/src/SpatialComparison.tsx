import { useState } from 'react';

type Paired = {wavelength_um:number[];local_power:number[];published_power:number[]};
type LossPeak = {qualified:boolean;sampled_drop_peak_um:number;field_decay?:{loaded_q?:number};
  qualified_equal_bus_loss?:{loaded_q?:number;apparent_external_q_each_bus?:number;apparent_internal_q?:number;apparent_internal_q_lower_bound?:number}};
type SpatialReport = {vendor?:string;case_id?:string;settings?:{geometry:string};result?:{wavelength_um:number[];beam_to_guide:number[]};
  vendor_comparison?:{channels:Record<string,{paired_spectrum:Paired}>}; loaded_loss?:{peaks:LossPeak[]};
  comparisons?:{vendor:string;geometry_matched?:boolean;wavelength_um?:number[];published_power?:number[];published_s21_power?:number[]}[];
  radiation?:{polar_edges_deg?:number[];fraction_of_incident?:number[][];fiber_return_power?:number[]};
};

function PairedSpectrum({series,vendor}: {series:Record<string,Paired>;vendor:string}) {
  const [selection,setSelection]=useState('');
  const [sample,setSample]=useState<number|null>(null);
  const labels=Object.keys(series);
  if (!labels.length) return null;
  const channel=labels.includes(selection)?selection:labels[0];
  const data=series[channel];
  if (!data?.wavelength_um.length) return null;
  const waves=data.wavelength_um, local=data.local_power, reference=data.published_power;
  const i=Math.min(sample ?? Math.floor(waves.length/2),waves.length-1);
  const left=Math.min(...waves), right=Math.max(...waves);
  const top=Math.max(.01,...local,...reference);
  const x=(w:number)=>38+456*(w-left)/(right-left || 1);
  const y=(p:number)=>122-107*p/top;
  return <section aria-label="Commercial spectrum comparison" className="cal-spectrum">
    <h3>Published spectrum comparison</h3>
    <label>Compared output<select value={channel} onChange={e=>{setSelection(e.target.value);setSample(null);}}>
      {labels.map(name=><option key={name}>{name}</option>)}</select></label>
    <p><span className="cal-native-key">● Workbench</span> · <span className="cal-reference-key">● {vendor}</span></p>
    <svg viewBox="0 0 528 152" role="img" aria-label={channel+' power on the same wavelength axis'}>
      <path d="M38 12V122H500" stroke="#65808c" fill="none"/>
      <line x1={x(waves[i])} x2={x(waves[i])} y1={12} y2={122} stroke="#b6c5cb" strokeDasharray="3 3"/>
      {waves.map((w,j)=><g key={j}><circle cx={x(w)} cy={y(local[j])} r={waves.length<10?3:1.8} fill="#6de2c4"/>
        <circle cx={x(w)} cy={y(reference[j])} r={waves.length<10?3:1.8} fill="#ffbf78"/></g>)}
      <g fill="#aec7cf" fontSize={10}><text x={2} y={16}>{top.toPrecision(3)}</text><text x={16} y={125}>0</text>
        <text x={38} y={145}>{left.toFixed(3)} µm</text><text x={448} y={145}>{right.toFixed(3)} µm</text></g>
    </svg>
    <label>Compared wavelength<select value={i} onChange={e=>setSample(Number(e.target.value))}>
      {waves.map((w,j)=><option key={j} value={j}>{w.toFixed(5)} µm</option>)}</select></label>
    <p><output aria-label="Compared powers">Workbench {(100*local[i]).toFixed(3)}% · {vendor} {(100*reference[i]).toFixed(3)}% · Difference {(100*(local[i]-reference[i])).toFixed(3)} percentage points</output></p>
  </section>;
}

export function SpatialComparison({value}:{value:unknown}) {
  const [angleSample,setAngleSample]=useState(2);
  if(!value || typeof value!=='object') return null;
  const r=value as SpatialReport;
  if(!['loaded-ring-3d','finite-grating-3d'].includes(r.case_id || ''))return null;
  const pairs:Record<string,Paired>={};
  for(const [name,channel] of Object.entries(r.vendor_comparison?.channels || {})) {
    if(channel.paired_spectrum) pairs[name]=channel.paired_spectrum;
  }
  const matched=r.comparisons?.find(c=>c.geometry_matched);
  const ref=matched?.published_power || matched?.published_s21_power;
  if(matched && ref && r.result && ref.length===r.result.wavelength_um.length) {
    pairs['Guided output']={wavelength_um:r.result.wavelength_um,local_power:r.result.beam_to_guide,published_power:ref};
  }
  const vendor=r.vendor?.toLowerCase()==='lumerical'?'Lumerical':'Tidy3D';
  const qualified=r.loaded_loss?.peaks.filter(p=>p.qualified && p.qualified_equal_bus_loss);
  const angles=r.radiation;
  const wavelengths=r.result?.wavelength_um || [];
  const angleIndex=Math.min(angleSample,wavelengths.length-1);
  return <>
    <PairedSpectrum series={pairs} vendor={vendor}/>
    {qualified && <section aria-label="Loaded resonance measurements"><h3>Loaded resonances</h3>
      {qualified.length ? <><table><thead><tr><th>Peak (µm)</th><th>Loaded Q from linewidth</th><th>Loaded Q from decay</th><th>External Q per bus</th></tr></thead>
        <tbody>{qualified.map(p=><tr key={p.sampled_drop_peak_um}><th>{p.sampled_drop_peak_um.toFixed(4)}</th>
          <td>{p.qualified_equal_bus_loss?.loaded_q?.toFixed(1)}</td><td>{p.field_decay?.loaded_q?.toFixed(1)}</td>
          <td>{p.qualified_equal_bus_loss?.apparent_external_q_each_bus?.toFixed(1)}</td></tr>)}</tbody></table>
        <p>The external Q estimate assumes equal bus coupling.</p></>:<p>This observation window has no qualified linewidth and decay pair.</p>}
    </section>}
    {angles && wavelengths.length>0 && <section aria-label="Reflected power distribution"><h3>{r.settings?.geometry==='compact-radiation'?'Upward radiation':'Reflected power'}</h3>
      <label>Radiation wavelength<select value={angleIndex} onChange={e=>setAngleSample(Number(e.target.value))}>
        {wavelengths.map((w,i)=><option key={i} value={i}>{w.toFixed(4)} µm</option>)}</select></label>
      {angles.fiber_return_power && <p>Power returned to the incident fiber mode: {(100*angles.fiber_return_power[angleIndex]).toPrecision(4)}%</p>}
      {angles.fraction_of_incident && angles.polar_edges_deg && <table><thead><tr><th>Polar angle in air</th><th>Fraction of incident power</th></tr></thead>
        <tbody>{angles.fraction_of_incident[angleIndex].map((f,i)=><tr key={i}><th>{angles.polar_edges_deg![i]}–{angles.polar_edges_deg![i+1]}°</th><td>{f.toPrecision(5)}</td></tr>)}</tbody></table>}
    </section>}
  </>;
}
