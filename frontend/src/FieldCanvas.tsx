import { useEffect, useRef, useState } from 'react';

export type Field = { field: number[][]; epsilon: number[][]; x_um: number[]; y_um: number[]; metadata: Record<string, unknown> };

export function FieldCanvas({data, view, fixedPeak}: {data: Field; view: 'field'|'epsilon';fixedPeak?:number}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [point,setPoint] = useState<{x:number;y:number}|null>(null);
  const values = view === 'field' ? data.field : data.epsilon;
  const ny = values.length, nx = values[0]?.length || 0;
  let min = Infinity, max = -Infinity;
  values.forEach(row=>row.forEach(v=>{ min=Math.min(min,v); max=Math.max(max,v); }));
  const limit = fixedPeak && fixedPeak > 0 ? fixedPeak : Math.max(Math.abs(min),Math.abs(max)) || 1;
  useEffect(()=>{
    const el = canvas.current;
    if(!el || !nx || !ny) return;
    el.width=nx;el.height=ny;
    const ctx=el.getContext('2d')!; const image=ctx.createImageData(nx,ny);
    for(let y=0;y<ny;y++) for(let x=0;x<nx;x++) {
      const value=values[y][x], i=((ny-y-1)*nx+x)*4;
      let r,g,b;
      if(view==='field') {
        const v=Math.max(-1,Math.min(1,value/limit)); const strength=Math.abs(v);
        const base=[19,28,41], color=v<0?[73,141,249]:[255,157,104];
        [r,g,b]=base.map((c,j)=>Math.round(c+(color[j]-c)*strength));
      } else {
        const t=(value-min)/(max-min || 1); r=22+t*67;g=37+t*182;b=54+t*143;
      }
      image.data[i]=r;image.data[i+1]=g;image.data[i+2]=b;image.data[i+3]=255;
    }
    ctx.putImageData(image,0,0);
  },[values,view,nx,ny,min,max,limit]);
  const sample=(event:React.MouseEvent<HTMLCanvasElement>)=>{
    const rect=event.currentTarget.getBoundingClientRect();
    const x=Math.min(nx-1,Math.max(0,Math.floor((event.clientX-rect.left)/rect.width*nx)));
    const y=Math.min(ny-1,Math.max(0,ny-1-Math.floor((event.clientY-rect.top)/rect.height*ny)));
    setPoint({x,y});
  };
  const hover=point && values[point.y]?.[point.x]!==undefined ? `x ${data.x_um[point.x]?.toFixed(3)} µm · y ${data.y_um[point.y]?.toFixed(3)} µm · ${view==='field'?'Ez':'εr'} ${values[point.y][point.x].toExponential(4)}` : 'Move over the field to inspect a sample';
  return <div className="field-display">
    <div className="field-y-label">y (µm)</div>
    <div className="field-y-ticks"><span>{data.y_um.at(-1)?.toFixed(1)}</span><span>0</span><span>{data.y_um[0]?.toFixed(1)}</span></div>
    <canvas ref={canvas} className="field-canvas" onMouseMove={sample} aria-label={`${view==='field'?'Actual Ez electric field':'Actual discretized relative permittivity'}, ${nx} by ${ny} samples`} />
    <div className="field-x-ticks"><span>{data.x_um[0]?.toFixed(1)}</span><span>0</span><span>{data.x_um.at(-1)?.toFixed(1)}</span></div>
    <div className="field-x-label">x (µm)</div>
    <div className="colorbar-row"><span>{view==='field'?'Ez · solver units':'Relative permittivity εr'}</span><div className="colorbar-block"><div className={`colorbar ${view}`} /><div className="colorbar-labels"><span>{(view==='field'?-limit:min).toPrecision(3)}</span><span>{view==='field'?'0':''}</span><span>{(view==='field'?limit:max).toPrecision(3)}</span></div></div></div>
    <div className="sample-readout">{hover}</div>
  </div>;
}
