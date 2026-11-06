type Output = {
  port:string;power_change_max:number;absolute_complex_change_max:number;
  physical_phase_change_rad:(number|null)[];straight_relative_phase_change_rad?:(number|null)[];
  physical_phase_within_0_10_rad?:boolean;return_within_0_001_power_and_0_01_amplitude?:boolean;
};
export type PhaseCheck = {launch:string;lower_mesh:number;higher_mesh:number;ports:Output[];receipt:string};

function largestAngle(values?: (number|null)[]) {
  const measured=(values||[]).filter((v):v is number=>v!==null);
  return measured.length ? Math.max(...measured.map(Math.abs)).toFixed(3) : '—';
}

export function PhaseSensitivity({check}:{check:PhaseCheck}){
  return <details className="cal-phase"><summary>Phase and weak-return mesh checks</summary>
    <p>{check.launch} excitation · {check.lower_mesh} to {check.higher_mesh} cells/µm on NVIDIA. Values are the largest observed changes across the three wavelengths.</p>
    <table><thead><tr><th>Output</th><th>Power change</th><th>Physical phase (rad)</th><th>Relative to straight guide (rad)</th><th>Gate</th></tr></thead>
      <tbody>{check.ports.map(port=><tr key={port.port}><th>{port.port}</th>
        <td>{port.power_change_max.toPrecision(4)}</td><td>{largestAngle(port.physical_phase_change_rad)}</td>
        <td>{largestAngle(port.straight_relative_phase_change_rad)}</td>
        <td>{(port.physical_phase_within_0_10_rad ?? port.return_within_0_001_power_and_0_01_amplitude)?'Pass':'Outside limit'}</td>
      </tr>)}</tbody></table>
    <p>The forward phase limit is 0.10 rad. A return must change by at most 0.001 in power fraction and 0.01 in complex amplitude. Phase is reported where both amplitudes reach 0.02. <a target="_blank" rel="noreferrer" href={'https://github.com/srinjoyganguly/photonics-workbench/blob/main/docs/verification/'+check.receipt}>Native inputs and values</a>.</p>
  </details>;
}
