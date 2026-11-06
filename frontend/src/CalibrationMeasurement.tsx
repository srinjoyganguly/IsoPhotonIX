import { useState } from 'react';

type Complex = [number, number];
export type Series = {label:string;values:(number|null)[]};
export type SpectrumData = {wavelengths:number[];series:Series[]};
type MatrixReport = {
  ports?: string[]; port_order?: string[]; channels?: string[]; wavelength_um: number[];
  s?: Complex[][][]; s_real?: number[][][]; s_imag?:number[][][];
};
type GratingRow = {wavelength_um:number[];beam_to_guide?:number[];guide_radiation_up?:number[];guide_radiation_down?:number[];valid_beam_bins?:boolean[];valid_energy_bins?:boolean[];direct_output_fraction?:number[];valid_direct_bins?:boolean[]};
type RingRow = {wavelength_um:number[];through:number[];drop:number[];reflection:number[];add_side?:number[];valid_bins?:boolean[]};
type Report = MatrixReport & {case_id?:string;vendor?:string;result?:GratingRow;series?:Record<string,RingRow>;studies?:Record<string,Record<string,GratingRow>>;compact_tidy_domain?:GratingRow;isolated_finite_height_ring?:Record<string,{measured_bare_ring_fsr_nm:number;published_loaded_ring_fsr_nm:number}>};

export function Spectrum({wavelengths,series}:{wavelengths:number[];series:Series[]}) {
  const [line,setLine]=useState(0);
  const data=series[line] ?? series[0];
  if (!data || wavelengths.length!==data.values.length || !wavelengths.length) return null;
  const lo=Math.min(...wavelengths), hi=Math.max(...wavelengths);
  const valid=data.values.filter((v):v is number=>v!==null && Number.isFinite(v));
  const top=Math.max(.01,...valid), bottom=Math.min(0,...valid);
  const segments:number[][][]=[];
  let current:number[][]=[];
  wavelengths.forEach((w,i)=>{
    const value=data.values[i];
    if(value===null || !Number.isFinite(value)){
      current=[];
      return;
    }
    if(!current.length)segments.push(current);
    current.push([36+466*(w-lo)/(hi-lo || 1),125-110*(value-bottom)/(top-bottom)]);
  });
  return <div className="cal-spectrum"><label>Observable<select value={series.indexOf(data)} onChange={e=>setLine(Number(e.target.value))}>
    {series.map((s,i)=><option key={s.label} value={i}>{s.label}</option>)}</select></label>
    {valid.length===0?<p>No qualified samples for {data.label}.</p>:<svg viewBox="0 0 528 156" role="img" aria-label={data.label+' versus vacuum wavelength'}>
      <path d="M36 12V125H506" stroke="#65808c" fill="none"/>
      {segments.map((line,i)=><g key={i}>
        <polyline points={line.map(p=>p.join(',')).join(' ')} stroke="#6de2c4" strokeWidth="2" fill="none"/>
        {(valid.length<20 || line.length===1) && line.map(([x,y],j)=><circle key={j} cx={x} cy={y} r="3" fill="#a2f6dd"/>)}
      </g>)}
      <g fill="#a9c0cb" fontSize="10"><text x="0" y="15">{top.toPrecision(3)}</text><text x="17" y="127">{bottom===0?'0':bottom.toPrecision(3)}</text>
        <text x="36" y="146">{lo.toFixed(3)} µm</text><text x="453" y="146">{hi.toFixed(3)} µm</text></g>
    </svg>}{valid.length>0 && valid.length<data.values.length && <small>{valid.length} of {data.values.length} wavelength samples qualified.</small>}</div>;
}

export function CalibrationMeasurement({value}:{value:unknown}) {
  const [launch,setLaunch]=useState(0),[wavelength,setWavelength]=useState(0),[vendor,setVendor]=useState('tidy3d');
  if(!value || typeof value!=='object')return null;
  const r=value as Report;
  const labels=r.ports ?? r.port_order ?? r.channels;
  if(labels && r.wavelength_um && (r.s || (r.s_real && r.s_imag))){
    const sample = (f:number,i:number):Complex => r.s ? r.s[f][i][launch] : [r.s_real![f][i][launch],r.s_imag![f][i][launch]];
    const series=labels.map((label,i)=>({label,values:r.wavelength_um.map((_,f)=>{const [a,b]=sample(f,i);return a*a+b*b;})}));
    return <section aria-label="Measured scattering"><h3>Measured scattering</h3>
      <label>Input channel<select value={launch} onChange={e=>setLaunch(+e.target.value)}>{labels.map((label,i)=><option key={label} value={i}>{label}</option>)}</select></label>
      <Spectrum wavelengths={r.wavelength_um} series={series}/>
      <label>Sample wavelength<select value={wavelength} onChange={e=>setWavelength(+e.target.value)}>{r.wavelength_um.map((w,i)=><option value={i} key={i}>{w.toFixed(5)} µm</option>)}</select></label>
      <table><thead><tr><th>Output channel</th><th>Re(S)</th><th>Im(S)</th><th>Power |S|²</th></tr></thead>
        <tbody>{labels.map((label,i)=>{const [a,b]=sample(wavelength,i);return <tr key={label}><th>{label}</th><td>{a.toPrecision(5)}</td><td>{b.toPrecision(5)}</td><td>{series[i].values[wavelength].toPrecision(5)}</td></tr>;})}</tbody>
      </table></section>;
  }
  const options=Object.keys(r.series ?? r.studies ?? {});
  const compactLabel='Tidy3D compact aperture';
  const beamLabel=r.vendor==='lumerical'?'Lumerical beam coupling':'Tidy3D beam coupling';
  if(r.case_id==='grating-gaussian-profile' && r.result)options.push(beamLabel);
  if(r.compact_tidy_domain)options.push(compactLabel);
  if(!options.length && r.isolated_finite_height_ring){
    return <section aria-label="Isolated ring spacing"><h3>Isolated ring spacing</h3>
      <table><thead><tr><th>Geometry</th><th>Bare ring FSR (nm)</th><th>Published loaded-ring FSR (nm)</th></tr></thead>
      <tbody>{Object.entries(r.isolated_finite_height_ring).map(([name,measurement])=><tr key={name}>
        <th>{name}</th><td>{measurement.measured_bare_ring_fsr_nm.toFixed(4)}</td><td>{measurement.published_loaded_ring_fsr_nm.toFixed(4)}</td>
      </tr>)}</tbody></table></section>;
  }
  if(!options.length)return null;
  const selected=options.includes(vendor)?vendor:options[0];
  const ring=r.series?.[selected], grating=selected===beamLabel?r.result:selected===compactLabel?r.compact_tidy_domain:r.studies?.[selected]?.fine ?? r.studies?.[selected]?.guided;
  const pole=r.isolated_finite_height_ring?.[selected];
  const qualified=(values:number[],mask?:boolean[])=>
    values.map((v,i)=>mask && !mask[i]?null:v);
  const spectrum: SpectrumData | null=ring?{wavelengths:ring.wavelength_um,series:[{label:'Through',values:qualified(ring.through,ring.valid_bins)},{label:'Drop',values:qualified(ring.drop,ring.valid_bins)},{label:'Reflection',values:qualified(ring.reflection,ring.valid_bins)},...(ring.add_side?[{label:'Add-side outgoing',values:qualified(ring.add_side,ring.valid_bins)}]:[])]}:
    grating?{wavelengths:grating.wavelength_um,series:[...(grating.beam_to_guide?[{label:'Gaussian to TE guide',values:qualified(grating.beam_to_guide,grating.valid_beam_bins)}]:[]),...(grating.direct_output_fraction?[{label:'Total output aperture flux',values:qualified(grating.direct_output_fraction,grating.valid_direct_bins)}]:[]),...(grating.guide_radiation_up?[{label:'Upward radiation fraction',values:qualified(grating.guide_radiation_up,grating.valid_energy_bins)}]:[]),...(grating.guide_radiation_down?[{label:'Downward radiation fraction',values:qualified(grating.guide_radiation_down,grating.valid_energy_bins)}]:[])]}:null;
  return spectrum && <section aria-label="Measured spectrum"><h3>Measured spectrum</h3>
    <label>Reference geometry<select value={selected} onChange={e=>setVendor(e.target.value)}>{options.map(v=><option key={v}>{v}</option>)}</select></label>
    <Spectrum key={selected} {...spectrum}/>
    {pole && <p>Finite-height isolated-ring mode spacing {pole.measured_bare_ring_fsr_nm.toFixed(3)} nm.
      The published loaded-ring spacing is {pole.published_loaded_ring_fsr_nm.toFixed(3)} nm.</p>}
    </section>;
}
