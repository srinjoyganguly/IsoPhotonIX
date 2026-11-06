import { useRef } from 'react';

type Port = {name:string; center_um:number[]; outward:string; width_um:number};
type Layout = {version:number; name:string; polygons:number[][][]; ports:Port[]};

export function parsePlanarLayout(text:string): Layout | null {
  try {
    const value = JSON.parse(text);
    return value && value.version===1 && Array.isArray(value.polygons) && Array.isArray(value.ports) &&
      value.ports.every((p:Port)=>p && Array.isArray(p.center_um) && p.center_um.length===2) ? value : null;
  } catch { return null; }
}

export function PlanarPortLayoutEditor({text, onChange}: {text:string; onChange:(text:string)=>void}) {
  const input = useRef<HTMLInputElement>(null);
  const layout = parsePlanarLayout(text);
  function setPort(id:number, change:Partial<Port>) {
    if (!layout) return;
    const ports = layout.ports.map((p,i)=> i===id ? {...p,...change} : p);
    onChange(JSON.stringify({...layout,ports},null,2));
  }
  return <div className="planar-layout-input">
    <h3>Planar layout</h3>
    {layout && <><p>{layout.name}</p>
      <table aria-label="Port placement"><thead><tr><th>Port</th><th>x (µm)</th><th>y (µm)</th><th>Outward normal</th></tr></thead>
        <tbody>{layout.ports.map((port,i)=><tr key={i}><th>{port.name}</th>
          {[0,1].map(axis=><td key={axis}><input type="number" step="any" aria-label={port.name+' '+(axis?'y':'x')}
            defaultValue={port.center_um[axis]} key={String(port.center_um[axis])} onBlur={event=>{
              if(!event.target.value.trim() || !Number.isFinite(event.target.valueAsNumber)) return;
              const center = [...port.center_um]; center[axis] = event.target.valueAsNumber;
              setPort(i,{center_um:center});
            }}/></td>)}
          <td><select aria-label={port.name+' outward normal'} value={port.outward}
            onChange={e=>setPort(i,{outward:e.target.value})}>
            {['-x','+x','-y','+y'].map(value=><option key={value}>{value}</option>)}</select></td>
        </tr>)}</tbody></table></>}
    <button type="button" className="button secondary compact" onClick={()=>input.current?.click()}>Open layout JSON</button>
    <input hidden type="file" accept=".json,application/json" ref={input} onChange={async e=>{
      const file=e.target.files?.[0]; e.target.value=''; if(file) onChange(await file.text());
    }}/>
    <details><summary>Edit polygons and ports</summary><textarea aria-label="Planar layout JSON" value={text}
      onChange={e=>onChange(e.target.value)} spellCheck={false}/></details>
    {!layout && text && <p role="alert">Enter a JSON layout with polygons and ports.</p>}
  </div>;
}
