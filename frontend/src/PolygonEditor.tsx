import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { type GeometryObject } from './geometryModel';

type Point={x:string;y:string};
const strings=(points:{x:number;y:number}[])=>points.map(point=>({x:String(point.x),y:String(point.y)}));
export function PolygonEditor({object,onCommit}:{object:GeometryObject;onCommit:(patch:Partial<GeometryObject>)=>boolean}) {
  const source=JSON.stringify([object.vertices??[],object.holes??[]]);
  const [contours,setContours]=useState<Point[][]>(()=>[strings(object.vertices??[]),...(object.holes??[]).map(strings)]);
  const [active,setActive]=useState(0),[selected,setSelected]=useState(0),[error,setError]=useState(''),[snap,setSnap]=useState('.01');
  const dragging=useRef<number|null>(null),svg=useRef<SVGSVGElement>(null);
  useEffect(()=>{setContours([strings(object.vertices??[]),...(object.holes??[]).map(strings)]);setActive(0);setSelected(0);setError('');},[object.id,source]);
  const base=[...(object.vertices??[]),...(object.holes??[]).flat()];
  const xs=base.map(p=>p.x),ys=base.map(p=>p.y);
  const minX=Math.min(...xs,0),maxX=Math.max(...xs,0),minY=Math.min(...ys,0),maxY=Math.max(...ys,0),span=Math.max(maxX-minX,maxY-minY,.5)*1.35;
  const cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const px=(x:number)=>120+(x-cx)*220/span,py=(y:number)=>120-(y-cy)*220/span;
  const validPoint=(p:Point)=>p.x.trim()!==''&&p.y.trim()!==''&&Number.isFinite(Number(p.x))&&Number.isFinite(Number(p.y));
  const pointPath=(points:Point[])=>points.every(validPoint)?points.map((p,i)=>`${i?'L':'M'}${px(Number(p.x))},${py(Number(p.y))}`).join(' ')+' Z':'';
  const path=contours.map(pointPath).join(' ');
  function change(index:number,patch:Partial<Point>) {setContours(old=>old.map((points,i)=>i===active?points.map((p,j)=>j===index?{...p,...patch}:p):points));setError('');}
  function move(event:PointerEvent<SVGSVGElement>) {
    if(dragging.current===null||!svg.current)return;
    const box=svg.current.getBoundingClientRect(),x=((event.clientX-box.left)/box.width*240-120)*span/220+cx,y=(120-(event.clientY-box.top)/box.height*240)*span/220+cy;
    const grid=Number(snap),quantize=(v:number)=>Number((grid>0&&Number.isFinite(grid)?Math.round(v/grid)*grid:v).toFixed(6));
    change(dragging.current,{x:String(quantize(x)),y:String(quantize(y))});
  }
  function nudge(event:KeyboardEvent<SVGCircleElement>,contour:number,index:number) {
    const directions:Record<string,[number,number]>={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]};
    const direction=directions[event.key];if(!direction)return;
    event.preventDefault();setActive(contour);setSelected(index);
    const step=Number(snap)>0&&Number.isFinite(Number(snap))?Number(snap):.01;
    setContours(old=>old.map((ring,j)=>j===contour?ring.map((point,k)=>k===index?{x:String(Number(point.x)+direction[0]*step),y:String(Number(point.y)+direction[1]*step)}:point):ring));
  }
  function addVertex() {
    setContours(old=>old.map((points,i)=>{if(i!==active)return points;const a=points[selected]??points.at(-1),b=points[(selected+1)%points.length]??a;const at=Math.min(selected+1,points.length);const next=[...points];next.splice(at,0,{x:String((Number(a?.x??0)+Number(b?.x??1))/2),y:String((Number(a?.y??0)+Number(b?.y??0))/2)});return next;}));setSelected(selected+1);
  }
  function addHole() {
    const radius=Math.max(.01,Math.min(maxX-minX,maxY-minY)/8);
    setContours(old=>[...old,strings([{x:cx-radius,y:cy-radius},{x:cx+radius,y:cy-radius},{x:cx+radius,y:cy+radius},{x:cx-radius,y:cy+radius}])]);setActive(contours.length);setSelected(0);setError('');
  }
  function apply() {
    try {
      if(contours.some(points=>points.some(p=>!validPoint(p))))throw new Error('Each vertex needs finite local X and Y coordinates.');
      const numeric=contours.map(points=>points.map(p=>({x:Number(p.x),y:Number(p.y)})));
      if(onCommit({vertices:numeric[0],holes:numeric.slice(1)}))setError('');
    }catch(reason){setError((reason as Error).message);}
  }
  return <section className="polygon-editor" aria-label="Polygon sketch editor">
    <h3>POLYGON SKETCH · LOCAL XY</h3><p>Drag a vertex or edit its coordinates. Apply commits the whole sketch after topology checks. Z thickness and transforms remain in the Inspector.</p>
    <svg ref={svg} className="polygon-sketch" viewBox="0 0 240 240" role="img" aria-label="Editable polygon top view; points are local micrometre coordinates" onPointerMove={move} onPointerUp={()=>{dragging.current=null;}} onPointerCancel={()=>{dragging.current=null;}}>
      <path d={`M0 ${py(0)}H240M${px(0)} 0V240`} className="polygon-axes"/><path d={path} fillRule="evenodd" className="polygon-outline"/>
      {contours.map((points,c)=>points.map((point,i)=>validPoint(point)?<circle key={`${c}-${i}`} cx={px(Number(point.x))} cy={py(Number(point.y))} r={c===active&&i===selected?5:3.5} className={c===active?'active':'inactive'} tabIndex={c===active?0:-1} role="button" aria-label={`Sketch contour ${c} vertex ${i+1}`} onPointerDown={event=>{event.preventDefault();setActive(c);setSelected(i);dragging.current=i;svg.current?.setPointerCapture(event.pointerId);}} onKeyDown={event=>nudge(event,c,i)}/>:null))}
      <text x="10" y="230">X → · Y ↑ · µm</text>
    </svg>
    <label>Contour<select aria-label="Polygon contour" value={active} onChange={event=>{setActive(Number(event.target.value));setSelected(0);}}>{contours.map((_,i)=><option value={i} key={i}>{i===0?'Outer boundary':`Hole ${i}`}</option>)}</select></label>
    <label>Snap / keyboard step (µm)<input aria-label="Polygon snap step" type="number" min="0" step=".01" value={snap} onChange={event=>setSnap(event.target.value)}/></label>
    <div className="polygon-vertices">{(contours[active]??[]).map((point,i)=><div className={`polygon-vertex ${i===selected?'selected':''}`} key={i}><button onClick={()=>setSelected(i)} aria-label={`Select vertex ${i+1}`}>{i+1}</button><input aria-label={`Vertex ${i+1} X`} type="number" step=".01" value={point.x} onFocus={()=>setSelected(i)} onChange={event=>change(i,{x:event.target.value})}/><input aria-label={`Vertex ${i+1} Y`} type="number" step=".01" value={point.y} onFocus={()=>setSelected(i)} onChange={event=>change(i,{y:event.target.value})}/></div>)}</div>
    <div className="polygon-actions"><button onClick={addVertex}>Insert vertex</button><button disabled={(contours[active]?.length??0)<=3} onClick={()=>{setContours(old=>old.map((points,i)=>i===active?points.filter((_,j)=>j!==selected):points));setSelected(0);}}>Remove vertex</button><button onClick={addHole}>Add hole</button>{active>0&&<button onClick={()=>{setContours(old=>old.filter((_,i)=>i!==active));setActive(0);setSelected(0);}}>Remove hole</button>}</div>
    <button className="button primary compact" onClick={apply}>Apply polygon sketch</button>
    {error&&<p className="library-error" role="alert">{error} The design remains unchanged.</p>}
  </section>;
}
