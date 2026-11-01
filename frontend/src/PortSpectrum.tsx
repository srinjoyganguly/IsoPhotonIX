import { useEffect, useState, type MouseEvent } from 'react';
import { request } from './api';

type Settlement = {
  settled?: boolean; energy_ratio?: number; spectral_relative_change?: number;
  decay_by?: number; spectrum_tolerance?: number; elapsed_fs?: number;
  thresholds?: { energy_decay?: number; spectral_relative_change?: number };
  [key: string]: unknown;
};
export type PortSpectrumData = {
  wavelength_um: number[]; frequency_per_um: number[];
  incident_flux: number[]; reflected_flux: number[]; transmitted_flux: number[]; reference_transmitted_flux: number[];
  reflectance: (number | null)[]; transmittance: (number | null)[]; uncollected_fraction: (number | null)[];
  valid_bin: boolean[];
  bin_reasons?: (string | null)[]; ratio_definition?: string;
  metadata: {
    spectra_valid?: boolean; settlement?: { reference?: Settlement; device?: Settlement };
    actual_device?: string; device_name?: string; versions?: Record<string, string>;
    [key: string]: unknown;
  };
};
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const numerical = (value: unknown, digits = 5) => finite(value) ? value.toExponential(digits) : 'Not recorded';
const ratio = (value: unknown) => finite(value) ? value.toPrecision(6) : 'Masked';
const series = [
  { key: 'transmittance', label: 'T', title: 'Transmission', color: '#74debf' },
  { key: 'reflectance', label: 'R', title: 'Reflection', color: '#b4a4f5' },
  { key: 'uncollected_fraction', label: '1 − R − T', title: 'Uncollected flux', color: '#edbc80' },
] as const;

function parseSpectrum(value: PortSpectrumData) {
  const count = value?.wavelength_um?.length;
  if (!count || !value.metadata || !Array.isArray(value.valid_bin) || value.valid_bin.length !== count) throw new Error('The saved spectrum is missing its wavelengths, validity mask or metadata.');
  for (const key of ['wavelength_um', 'frequency_per_um', 'incident_flux', 'reflected_flux', 'transmitted_flux', 'reference_transmitted_flux'] as const) {
    if (!Array.isArray(value[key]) || value[key].length !== count || value[key].some(item => !finite(item))) throw new Error(`The saved ${key} array is incomplete or non-finite.`);
  }
  if (value.wavelength_um.some((wavelength, index) => wavelength <= 0 || (index > 0 && wavelength <= value.wavelength_um[index - 1]))) throw new Error('Saved wavelengths must be positive and strictly increasing.');
  if (value.valid_bin.some(item => typeof item !== 'boolean')) throw new Error('The saved spectrum has an invalid sample-validity mask.');
  for (const { key } of series) {
    if (!Array.isArray(value[key]) || value[key].length !== count || value[key].some((item, index) => value.valid_bin[index] ? !finite(item) : item !== null)) throw new Error(`The saved ${key} samples do not match the validity mask.`);
  }
  return value;
}

function saveJson(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + '\n'], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function SpectrumGraph({ data, selected, onSelect }: { data: PortSpectrumData; selected: number; onSelect: (index: number) => void }) {
  const width = 780, height = 330, left = 64, top = 20, plotWidth = 690, plotHeight = 250;
  const wavelengths = data.wavelength_um;
  const xMin = wavelengths[0], xMax = wavelengths.at(-1)!;
  const values = series.flatMap(({ key }) => data[key].filter((value, index): value is number => data.valid_bin[index] && finite(value)));
  const low = Math.min(0, ...values), high = Math.max(1, ...values), padding = Math.max(.03, (high - low) * .07);
  const yMin = low - padding, yMax = high + padding;
  const x = (value: number) => left + (value - xMin) / (xMax - xMin || 1) * plotWidth;
  const y = (value: number) => top + (yMax - value) / (yMax - yMin) * plotHeight;
  function path(key: typeof series[number]['key']) {
    let previousValid = false;
    return data[key].map((value, index) => {
      const valid = data.valid_bin[index] && finite(value);
      const point = valid ? `${previousValid ? 'L' : 'M'} ${x(wavelengths[index])} ${y(value!)}` : '';
      previousValid = valid;
      return point;
    }).join(' ');
  }
  function chooseFromPointer(event: MouseEvent<SVGSVGElement>) {
    const rectangle = event.currentTarget.getBoundingClientRect();
    const coordinate = (event.clientX - rectangle.left) / rectangle.width * width;
    const wavelength = xMin + (coordinate - left) / plotWidth * (xMax - xMin);
    let closest = 0;
    for (let index = 1; index < wavelengths.length; index++) if (Math.abs(wavelengths[index] - wavelength) < Math.abs(wavelengths[closest] - wavelength)) closest = index;
    onSelect(closest);
  }
  return <div className="port-spectrum-plot">
    <div className="port-spectrum-legend">{series.map(item => <span key={item.key}><i style={{ background: item.color }}/>{item.label} · {item.title}</span>)}</div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Actual incident-normalized reflection, transmission and uncollected-flux spectral samples. Lines join adjacent valid samples without smoothing." onMouseMove={chooseFromPointer} onClick={chooseFromPointer} data-sample-count={wavelengths.length}>
      <rect x={left} y={top} width={plotWidth} height={plotHeight} fill="#0d1924" stroke="#3c5366"/>
      {[0, .25, .5, .75, 1].map(fraction => {
        const value = yMin + fraction * (yMax - yMin);
        return <g key={fraction}><line x1={left} x2={left + plotWidth} y1={y(value)} y2={y(value)} stroke="#2b4153"/><text x={left - 10} y={y(value) + 4} fill="#93adbf" fontSize="11" textAnchor="end">{Number(value.toPrecision(3))}</text></g>;
      })}
      {[0, .25, .5, .75, 1].map(fraction => {
        const value = xMin + fraction * (xMax - xMin);
        return <g key={fraction}><line x1={x(value)} x2={x(value)} y1={top} y2={top + plotHeight} stroke="#2b4153"/><text x={x(value)} y={top + plotHeight + 22} fill="#93adbf" fontSize="11" textAnchor="middle">{value.toFixed(3)}</text></g>;
      })}
      <line x1={left} x2={left + plotWidth} y1={y(0)} y2={y(0)} stroke="#758695" strokeDasharray="3 4"/>
      {series.map(item => <g key={item.key} data-series={item.key}><path d={path(item.key)} fill="none" stroke={item.color} strokeWidth="2" strokeDasharray={item.key === 'uncollected_fraction' ? '5 4' : undefined}/>{data[item.key].map((value, index) => data.valid_bin[index] && finite(value) ? <circle key={index} cx={x(wavelengths[index])} cy={y(value)} r={selected === index ? 4.5 : 2.2} fill={item.color} data-bin={index} data-value={value}/> : null)}</g>)}
      <line x1={x(wavelengths[selected])} x2={x(wavelengths[selected])} y1={top} y2={top + plotHeight} stroke="#b8c5d1" strokeDasharray="3 5" opacity=".65"/>
      {data.valid_bin.map((valid, index) => valid ? null : <text key={index} x={x(wavelengths[index])} y={top + plotHeight - 7} textAnchor="middle" fill="#e9ad85" fontSize="12">×</text>)}
      <text x={left + plotWidth / 2} y={height - 9} fill="#abc1d1" fontSize="12" textAnchor="middle">Wavelength (µm)</text>
      <text transform={`translate(16 ${top + plotHeight / 2}) rotate(-90)`} fill="#abc1d1" fontSize="11" textAnchor="middle">Incident-normalized flux</text>
    </svg>
    <p className="port-chart-note">Dots are recorded frequency samples. Lines connect adjacent valid samples; no smoothing or clipping. × marks a masked bin.</p>
  </div>;
}

function SettlementEvidence({ data }: { data: PortSpectrumData }) {
  return <div className="port-settlement"><h3>Settlement diagnostics</h3><p>A completed worker is not proof of convergence. Energy decay and spectral stability are recorded separately for both solves.</p><div className="port-settlement-stages">{(['reference', 'device'] as const).map(stage => {
    const record = data.metadata.settlement?.[stage];
    return <div key={stage} className="port-settlement-card"><h4>{stage === 'reference' ? 'Incident reference' : 'Device study'}<span>{record?.settled === true ? 'Criteria met' : record?.settled === false ? 'Criteria not met' : 'Not recorded'}</span></h4><dl><div><dt>Energy ratio</dt><dd>{numerical(record?.energy_ratio)}</dd></div><div><dt>Energy threshold</dt><dd>{numerical(record?.decay_by ?? record?.thresholds?.energy_decay)}</dd></div><div><dt>Spectrum change</dt><dd>{numerical(record?.spectral_relative_change)}</dd></div><div><dt>Spectrum threshold</dt><dd>{numerical(record?.spectrum_tolerance ?? record?.thresholds?.spectral_relative_change)}</dd></div>{finite(record?.elapsed_fs) && <div><dt>Simulated time</dt><dd>{record.elapsed_fs.toFixed(3)} fs</dd></div>}</dl>{record && <details><summary>Complete {stage} diagnostics</summary><pre>{JSON.stringify(record, null, 2)}</pre></details>}</div>;
  })}</div></div>;
}

export function PortSpectrum({ jobId }: { jobId: string }) {
  const [spectrum, setSpectrum] = useState<PortSpectrumData | null>(null);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState(0);
  useEffect(() => {
    let active = true; setError(''); setSpectrum(null);
    request<PortSpectrumData>(`/api/jobs/${jobId}/spectrum`).then(value => {
      if (!active) return;
      const parsed = parseSpectrum(value); setSpectrum(parsed); setSelected(Math.floor(parsed.wavelength_um.length / 2));
    }).catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [jobId, retry]);
  if (error) return <div className="frame-error" role="alert"><span>Spectrum unavailable: {error}</span><button onClick={() => setRetry(value => value + 1)}>Retry spectrum</button></div>;
  if (!spectrum) return <p className="port-spectrum-loading" role="status">Reading the saved numerical spectrum…</p>;
  const validCount = spectrum.valid_bin.filter(Boolean).length;
  const binValid = spectrum.valid_bin[selected];
  return <div className="port-spectrum" data-run-id={jobId}>
    <div className="port-spectrum-title"><div><h3>Incident-normalized spectra</h3><p>Signed total flux through finite monitor apertures · no commercial accuracy certification</p></div><span className="port-spectrum-validity">{spectrum.metadata.spectra_valid === true ? 'Recorded validity checks met' : 'Spectrum validity not established'}</span></div>
    <p className="port-spectrum-count">{validCount} / {spectrum.wavelength_um.length} valid bins. Masked ratios remain unavailable; raw flux and diagnostics are retained.</p>
    {(spectrum.metadata.actual_device || spectrum.metadata.device_name) && <p className="port-actual-execution">Recorded execution: {spectrum.metadata.device_name || spectrum.metadata.actual_device}{spectrum.metadata.actual_device && spectrum.metadata.device_name ? ` · ${spectrum.metadata.actual_device}` : ''}</p>}
    <SpectrumGraph data={spectrum} selected={selected} onSelect={setSelected}/>
    <label className="port-sample-slider"><span>Inspect spectral sample <strong>{selected + 1} / {spectrum.wavelength_um.length} · {spectrum.wavelength_um[selected].toFixed(6)} µm</strong></span><input aria-label="Port spectral sample" type="range" min={0} max={spectrum.wavelength_um.length - 1} step={1} value={selected} onChange={event => setSelected(Number(event.target.value))}/></label>
    <div className="port-selected-sample" data-bin={selected} data-valid={String(binValid)}>{series.map(item => <div key={item.key}><span>{item.title} · {item.label}</span><strong data-result={item.key}>{ratio(spectrum[item.key][selected])}</strong></div>)}</div>
    {!binValid && <p className="port-masked-note">{spectrum.bin_reasons?.[selected] || 'This sample is masked by the recorded validity checks. Inspect the incident flux and settlement evidence before interpreting it.'}</p>}
    <SettlementEvidence data={spectrum}/>
    <details className="port-samples"><summary>Inspect all numerical samples</summary><div className="port-samples-scroll"><table><caption>Saved signed flux samples, normalized to the separate incident reference. Invalid ratios are masked.</caption><thead><tr><th scope="col">λ (µm)</th><th scope="col">f (1/µm)</th><th scope="col">T</th><th scope="col">R</th><th scope="col">1−R−T</th><th scope="col">Incident flux</th><th scope="col">Reflected flux</th><th scope="col">Transmitted flux</th><th scope="col">Validity</th></tr></thead><tbody>{spectrum.wavelength_um.map((wavelength, index) => <tr key={index} data-bin={index} className={index === selected ? 'selected' : ''}><th scope="row">{wavelength.toFixed(6)}</th><td>{numerical(spectrum.frequency_per_um[index])}</td><td>{ratio(spectrum.transmittance[index])}</td><td>{ratio(spectrum.reflectance[index])}</td><td>{ratio(spectrum.uncollected_fraction[index])}</td><td>{numerical(spectrum.incident_flux[index])}</td><td>{numerical(spectrum.reflected_flux[index])}</td><td>{numerical(spectrum.transmitted_flux[index])}</td><td>{spectrum.valid_bin[index] ? 'Valid' : 'Masked'}</td></tr>)}</tbody></table></div></details>
    <div className="port-result-actions"><a className="button secondary compact" href={`/api/jobs/${jobId}/spectrum.csv`} download>Export spectrum CSV</a><button className="button secondary compact" onClick={() => saveJson(spectrum, `port-spectrum-${jobId.slice(0, 8)}.json`)}>Export spectrum JSON</button></div>
    <p className="port-chart-note">CSV retains raw ratios even for unqualified bins: always read its valid_bin column. The chart and JSON mask invalid ratios.</p>
    <details className="port-spectrum-metadata"><summary>Normalization, source, and numerical metadata</summary><pre>{JSON.stringify(spectrum.metadata, null, 2)}</pre></details>
  </div>;
}
