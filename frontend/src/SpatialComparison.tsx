import { useState } from 'react';

type Paired = {wavelength_um:number[];local_power:number[];published_power:number[]};
type ReturnWindows = {
  max_absolute_return_change:number[];max_phase_change_rad:(number|null)[];
  amplitude_qualified:boolean[];phase_qualified:boolean[];absolute_amplitude_limit:number;
  phase_amplitude_floor:number;phase_limit_rad:number;reference_l2_limit:number;consecutive_changes:number;
};
type LossPeak = {qualified:boolean;sampled_drop_peak_um:number;field_decay?:{loaded_q?:number};
  qualified_equal_bus_loss?:{loaded_q?:number;apparent_external_q_each_bus?:number;apparent_internal_q?:number;apparent_internal_q_lower_bound?:number}};
type PowerDiagnostic =
  | {observable:'ring'; nearest_published_drop_line:{published_sampled_center_um:number;local_sampled_center_um?:number;
      local_fwhm_nm?:number;published_fwhm_nm?:number|null;local_target_distance_in_linewidths?:number;published_target_distance_in_linewidths?:number}}
  | {observable:'grating';wavelength_um:number[];local_best_sample_um:number;published_best_sample_um:number; mean_absolute_difference:number;
      gain_only:{feasible:boolean;tolerance_power_fraction:number}};
type SpatialReport = {vendor?:string;case_id?:string;settings?:{geometry:string};result?:{wavelength_um:number[];beam_to_guide:number[]};
  power_diagnostic?:PowerDiagnostic;accuracy_criteria?:Record<string,boolean|null>;
  figure_readout?:{wavelength_um?:number;drop_power_bounds?:[number,number];uncertainty_db?:number};
  vendor_comparison?:{channels:Record<string,{paired_spectrum:Paired}>}; loaded_loss?:{peaks:LossPeak[]};
  comparisons?:{vendor:string;geometry_matched?:boolean;wavelength_um?:number[];published_power?:number[];published_s21_power?:number[]}[];
  radiation?:{polar_edges_deg?:number[];fraction_of_incident?:number[][];fiber_return_power?:number[];reciprocal_return_settling?:ReturnWindows};
};

const criteriaLabels:Record<string,string>={
  native_full_spectrum_qualified:'Numerical checks over the full measured spectrum',
  nearest_drop_center_within_one_published_linewidth:'Drop peak within one published linewidth',
  through_at_target_within_five_points:'Through power at 1550 nm within 5 percentage points',
  drop_at_target_within_five_points:'Drop power at 1550 nm within 5 percentage points',
  through_band_mean_within_five_points:'Mean through-power difference within 5 percentage points',
  drop_band_mean_within_five_points:'Mean drop-power difference within 5 percentage points',
  target_within_five_points:'Guided power at 1550 nm within 5 percentage points',
  band_mean_within_five_points:'Mean guided-power difference within 5 percentage points',
};

function PowerExplanation({data,vendor}:{data:PowerDiagnostic;vendor:string}) {
  if(data.observable==='ring') {
    const line=data.nearest_published_drop_line;
    return <section aria-label="Ring detuning"><h3>Why the power at 1550 nm differs</h3>
      <p>{vendor}’s nearest published drop peak is at {(line.published_sampled_center_um*1000).toFixed(2)} nm.
        {line.local_sampled_center_um!==undefined && <> The nearest measured workbench peak is at {(line.local_sampled_center_um*1000).toFixed(2)} nm.</>}</p>
      {line.local_target_distance_in_linewidths!==undefined && line.published_target_distance_in_linewidths!==undefined &&
        <p>1550 nm lies {line.local_target_distance_in_linewidths.toFixed(2)} linewidths from the fitted workbench peak
        and {line.published_target_distance_in_linewidths.toFixed(2)} from the published peak.
        Their widths are {line.local_fwhm_nm?.toFixed(2)} and {line.published_fwhm_nm?.toFixed(2)} nm respectively.</p>}
    </section>;
  }
  return <section aria-label="Grating power difference"><h3>Coupling across the measured band</h3>
    <p>Across {data.wavelength_um.length} sampled wavelengths, the strongest workbench coupling is at {(data.local_best_sample_um*1000).toFixed(0)} nm
      and the strongest published coupling is at {(data.published_best_sample_um*1000).toFixed(0)} nm.
      The mean absolute difference is {(100*data.mean_absolute_difference).toFixed(2)} percentage points.</p>
    {!data.gain_only.feasible && <p>A single power multiplier cannot bring all samples within
      {' '}{(100*data.gain_only.tolerance_power_fraction).toFixed(0)} percentage points of {vendor}.</p>}
  </section>;
}

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
  const figure=r.figure_readout;
  const qualified=r.loaded_loss?.peaks.filter(p=>p.qualified && p.qualified_equal_bus_loss);
  const angles=r.radiation;
  const settling=angles?.reciprocal_return_settling;
  const wavelengths=r.result?.wavelength_um || [];
  const angleIndex=Math.min(angleSample,wavelengths.length-1);
  return <>{r.accuracy_criteria && <details className="cal-runtime"><summary>Numerical and commercial-reference checks</summary>
    <table><thead><tr><th>Criterion</th><th>Result</th></tr></thead><tbody>{Object.entries(r.accuracy_criteria).map(([key,pass])=>
      <tr key={key}><th>{criteriaLabels[key] || key}</th><td>{pass===null?'Unresolved':pass?'Passed':'Outside limit'}</td></tr>)}</tbody></table></details>}
    <PairedSpectrum series={pairs} vendor={vendor}/>
    {figure?.drop_power_bounds && figure.wavelength_um && <p>At {(figure.wavelength_um*1000).toFixed(0)} nm the published Tidy3D ring trace spans
      {' '}{(100*figure.drop_power_bounds[0]).toFixed(2)}–{(100*figure.drop_power_bounds[1]).toFixed(2)}% drop power when its plotted line thickness is included.</p>}
    {figure?.uncertainty_db && <p>The Tidy3D grating values were read from the published figure with a ±{figure.uncertainty_db.toFixed(2)} dB readout uncertainty.</p>}
    {r.power_diagnostic && <PowerExplanation data={r.power_diagnostic} vendor={vendor}/>}
    {qualified && <section aria-label="Loaded resonance measurements"><h3>Loaded resonances</h3>
      {qualified.length ? <><table><thead><tr><th>Peak (µm)</th><th>Loaded Q from linewidth</th><th>Loaded Q from decay</th><th>External Q per bus</th><th>Apparent internal Q</th></tr></thead>
        <tbody>{qualified.map(p=><tr key={p.sampled_drop_peak_um}><th>{p.sampled_drop_peak_um.toFixed(4)}</th>
          <td>{p.qualified_equal_bus_loss?.loaded_q?.toFixed(1)}</td><td>{p.field_decay?.loaded_q?.toFixed(1)}</td>
          <td>{p.qualified_equal_bus_loss?.apparent_external_q_each_bus?.toFixed(1)}</td>
          <td>{p.qualified_equal_bus_loss?.apparent_internal_q?.toFixed(0) ??
            (p.qualified_equal_bus_loss?.apparent_internal_q_lower_bound ? '≥ '+p.qualified_equal_bus_loss.apparent_internal_q_lower_bound.toFixed(0) : 'Unresolved')}</td></tr>)}</tbody></table>
        <p>The external Q estimate assumes equal bus coupling.</p></>:<p>This observation window has no qualified linewidth and decay pair.</p>}
    </section>}
    {settling && <details className="cal-runtime"><summary>Weak reflection over the final time windows</summary>
      <p>The amplitude limit is {settling.absolute_amplitude_limit}. Phase requires |S11| ≥ {settling.phase_amplitude_floor}
        {' '}and a change ≤ {settling.phase_limit_rad} rad across each of the last {settling.consecutive_changes} increments.
        The incident field’s relative L2 change must be ≤ {settling.reference_l2_limit}.</p>
      <table><thead><tr><th>Wavelength (nm)</th><th>Max |ΔS11|</th><th>Amplitude</th><th>Max |Δphase| (rad)</th><th>Phase</th></tr></thead>
        <tbody>{wavelengths.map((w,i)=><tr key={w}><th>{(1000*w).toFixed(1)}</th>
          <td>{settling.max_absolute_return_change[i].toPrecision(3)}</td><td>{settling.amplitude_qualified[i]?'Passed':'Outside limit'}</td>
          <td>{settling.max_phase_change_rad[i]?.toPrecision(3) ?? '—'}</td><td>{settling.phase_qualified[i]?'Passed':'Unqualified'}</td>
        </tr>)}</tbody></table></details>}
    {angles && wavelengths.length>0 && <section aria-label="Reflected power distribution"><h3>{r.settings?.geometry==='compact-radiation'?'Upward radiation':'Reflected power'}</h3>
      <label>Radiation wavelength<select value={angleIndex} onChange={e=>setAngleSample(Number(e.target.value))}>
        {wavelengths.map((w,i)=><option key={i} value={i}>{w.toFixed(4)} µm</option>)}</select></label>
      {angles.fiber_return_power && <p>Power returned to the incident fiber mode: {(100*angles.fiber_return_power[angleIndex]).toPrecision(4)}%</p>}
      {angles.fraction_of_incident && angles.polar_edges_deg && <table><thead><tr><th>Polar angle in air</th><th>Fraction of incident power</th></tr></thead>
        <tbody>{angles.fraction_of_incident[angleIndex].map((f,i)=><tr key={i}><th>{angles.polar_edges_deg![i]}–{angles.polar_edges_deg![i+1]}°</th><td>{f.toPrecision(5)}</td></tr>)}</tbody></table>}
    </section>}
  </>;
}
