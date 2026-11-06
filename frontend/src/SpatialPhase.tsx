import { useState } from 'react';

type NativePhase = {power:number; complex:number; phase:number|null; strong:number};
type CommercialPhase = {native_power:number;vendor_power:number; signed_native_minus_vendor_phase_rad:number|null};
export type SpatialPhaseStudy = {
  receipt:string;vendor:string;amplitude_floor:number;
  samples:{mesh:number;passed:boolean;valid:number;total:number;band:Record<string,{count:number;rms:number|null;max_abs:number|null}>;central_wavelength_um:number|null;absolute:Record<string,CommercialPhase>;
    commercial:Record<string,{max_abs:number|null;count:number}>}[];
  pairs:{meshes:[number,number];channels:Record<string,NativePhase>}[];
};
const value=(n:number|null)=>n===null?'—':n.toPrecision(4);

export function SpatialPhase({studies}:{studies:SpatialPhaseStudy[]}) {
  const [index,setIndex]=useState(0);
  const study=studies[index];
  if (!study) return null;
  return <details className="cal-runtime"><summary>Phase and weak reflections</summary>
    <label>Comparison<select value={index} onChange={e=>setIndex(+e.target.value)}>
      {studies.map((s,i)=><option key={s.receipt} value={i}>{s.vendor} · {s.receipt.replace(/^v014-|\.json$/g,'')}</option>)}
    </select></label>
    {study.pairs.map(pair=><div className="cal-time-scroll" key={pair.meshes.join('-')}><table>
      <caption>{pair.meshes[0]===pair.meshes[1]
        ? `Largest change between the two controls at ${pair.meshes[0]} cells/µm`
        : `Largest change from ${pair.meshes[0]} to ${pair.meshes[1]} cells/µm`} at common port planes</caption>
      <thead><tr><th>Port</th><th>Power fraction</th><th>Complex amplitude</th><th>Phase (rad)</th><th>Usable phase samples</th></tr></thead>
      <tbody>{Object.entries(pair.channels).map(([name,v])=><tr key={name}><th>{name}</th><td>{value(v.power)}</td>
        <td>{value(v.complex)}</td><td>{value(v.phase)}</td><td>{v.strong}</td></tr>)}</tbody></table></div>)}
    <p>Phase requires both amplitudes to reach {study.amplitude_floor}, with stable monitored fields. Power and complex-amplitude changes remain measurable below that floor.</p>
    {study.samples.map((s,i)=><div key={i}>{!s.passed && <p>At {s.mesh} cells/µm, {s.valid} of {s.total} native wavelength samples passed. The tables use qualified samples.</p>}{s.central_wavelength_um!==null && <div className="cal-time-scroll"><table>
      <caption>Workbench and Lumerical at {(s.central_wavelength_um*1000).toFixed(3)} nm · {s.mesh} cells/µm</caption>
      <thead><tr><th>Port</th><th>Workbench power</th><th>Lumerical power</th><th>Signed phase difference (rad)</th></tr></thead>
      <tbody>{Object.entries(s.absolute).map(([name,v])=><tr key={name}><th>{name}</th><td>{value(v.native_power)}</td>
        <td>{value(v.vendor_power)}</td><td>{value(v.signed_native_minus_vendor_phase_rad)}</td></tr>)}</tbody></table></div>}
      {Object.keys(s.band).length>0 && <div className="cal-time-scroll"><table><caption>Absolute phase difference across the Lumerical band · {s.mesh} cells/µm</caption>
      <thead><tr><th>Port</th><th>RMS (rad)</th><th>Maximum (rad)</th><th>Usable samples</th></tr></thead>
      <tbody>{Object.entries(s.band).map(([name,v])=><tr key={name}><th>{name}</th><td>{value(v.rms)}</td>
        <td>{value(v.max_abs)}</td><td>{v.count}</td></tr>)}</tbody></table></div>}</div>)}
    {study.vendor.toLowerCase()==='tidy3d' && <p>The published Tidy3D example supplies power. The phase changes above compare local GPU runs.</p>}
    <p><a href={'/api/calibrations/phase.csv?receipt='+encodeURIComponent(study.receipt)}>Download registered complex samples (CSV)</a></p>
    <a href={'https://github.com/srinjoyganguly/photonics-workbench/blob/main/docs/verification/'+study.receipt}
       target="_blank" rel="noreferrer">Frequency samples, reference planes and qualification masks</a>
  </details>;
}
