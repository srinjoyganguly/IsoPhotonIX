import { useEffect, useId, useReducer, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { type GeometryDocument, type GeometryObject, type Vec2 } from './geometryModel';
import { addSketch, editContour, MAX_ROUTE_POINTS, moveSelection, routeOutline, snapPoint, worldPoint } from './sketchGeometry';
import { initialPointHistory, pointHistory } from './sketchHistory';
import './planSketch.css';

type Tool = 'move' | 'polygon' | 'route';
type Editing = { id: string; ring: number };
type Drag = { kind: 'point'; pointerId: number; index: number; start: Vec2; original: GeometryDocument }
  | { kind: 'move'; pointerId: number; id: string; start: Vec2; anchor: Vec2; original: GeometryDocument };
type Props = {
  scene: GeometryDocument; selectedId: string | null; onSelect: (id: string | null) => void;
  onApply: (document: GeometryDocument, message: string) => boolean;
};
const WIDTH = 800, HEIGHT = 438;
const centre = { x: WIDTH / 2, y: HEIGHT / 2 };
const tools: { id: Tool; name: string }[] = [
  { id: 'move', name: 'Select / move' }, { id: 'polygon', name: 'Polygon' }, { id: 'route', name: 'Route' },
];
const format = (n: number) => Number(n.toFixed(5)).toString();
function path(points: Vec2[], closed = false) {
  return points.length ? points.map((p, i) => `${i ? 'L' : 'M'} ${p.x} ${p.y}`).join(' ') + (closed ? ' Z' : '') : '';
}
function footprint(object: GeometryObject) {
  if (object.kind === 'polygon') return <path fillRule="evenodd" d={[object.vertices!, ...object.holes!].map(p => path(p, true)).join(' ')}/>;
  if (object.kind === 'box') return <rect x={-object.size.x / 2} y={-object.size.y / 2} width={object.size.x} height={object.size.y}/>;
  if (object.kind === 'cylinder') return <circle r={object.radius}/>;
  const circle = (r: number) => `M ${-r} 0 a ${r} ${r} 0 1 0 ${2*r} 0 a ${r} ${r} 0 1 0 ${-2*r} 0 Z`;
  return <path fillRule="evenodd" d={circle(object.radius) + ' ' + circle(object.inner_radius)}/>;
}

export function GeometryStage({children, ...props}: Props & {children: ReactNode}) {
  const [view, setView] = useState<'3d' | 'xy'>('3d');
  const [opened, setOpened] = useState(false);
  return <>
    <nav className="plan-view-tabs" aria-label="Geometry workspace view">
      <button aria-pressed={view === '3d'} onClick={() => setView('3d')}>3D view</button>
      <button aria-pressed={view === 'xy'} onClick={() => {setOpened(true); setView('xy');}}>Draw in XY</button>
    </nav>
    <div hidden={view !== '3d'}>{children}</div>
    {opened && <div hidden={view !== 'xy'}><PlanSketch {...props}/></div>}
  </>;
}

function PlanSketch({scene, selectedId, onSelect, onApply}: Props) {
  const [tool, setTool] = useState<Tool>('route');
  const [history, dispatch] = useReducer(pointHistory, undefined, () => initialPointHistory());
  const { points: placed, selected: selectedPoint } = history.present;
  const [previewPoint, setPreviewPoint] = useState<{ index: number; position: Vec2 } | null>(null);
  const points = previewPoint ? placed.map((p, i) => i === previewPoint.index ? previewPoint.position : p) : placed;
  const [entryX, setEntryX] = useState(''), [entryY, setEntryY] = useState('');
  const [snap, setSnap] = useState('0.25'), [width, setWidth] = useState('0.5'), [radius,setRadius] = useState('0');
  const [height, setHeight] = useState('0.22'), [z, setZ] = useState('0'), [name, setName] = useState('Drawn route');
  const [editing, setEditing] = useState<Editing | null>(null), [ring, setRing] = useState(0);
  const [hover, setHover] = useState<Vec2 | null>(null), [offset, setOffset] = useState<Vec2 | null>(null);
  const [error, setError] = useState(''), [notice, setNotice] = useState('');
  const drag = useRef<Drag | null>(null);
  const [base, setBase] = useState<GeometryDocument | null>(null);
  const svg = useRef<SVGSVGElement>(null), pattern = useId();
  const selected = scene.objects.find(o => o.id === selectedId);
  const member = scene.components?.find(c => c.id === selected?.component_id);
  const activeSketch = !!editing || !!placed.length || !!history.past.length || !!history.future.length;
  const stale = activeSketch && base !== scene;
  const scale = Math.min((WIDTH - 88) / scene.domain.x, (HEIGHT - 70) / scene.domain.y);
  const canvasPoint = (p: Vec2) => ({ x: centre.x + p.x * scale, y: centre.y - p.y * scale });
  const snapSize = snap === 'off' ? null : Number(snap);
  const guideStep = snapSize ?? Math.pow(10, Math.floor(Math.log10(Math.max(scene.domain.x, scene.domain.y))) - 1);
  const gridStep = guideStep * Math.max(1, Math.ceil(14 / (scale * guideStep)));
  const toWorld = (event: PointerEvent<SVGElement>) => {
    const matrix = svg.current?.getScreenCTM();
    if (!matrix) throw new Error('Drawing coordinates are unavailable.');
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return {x:(p.x - centre.x)/scale, y:(centre.y - p.y)/scale};
  };
  const currentPoint = points[selectedPoint];
  useEffect(() => {
    setEntryX(currentPoint ? String(currentPoint.x) : '');
    setEntryY(currentPoint ? String(currentPoint.y) : '');
  }, [selectedPoint, currentPoint?.x, currentPoint?.y]);
  useEffect(() => setRing(0), [selectedId]);
  function updatePoints(next: Vec2[], selection: number) {
    if (stale) return;
    if (base === null) setBase(scene);
    dispatch({type:'write', points:next, selected:selection}); setError(''); setNotice('');
  }
  function append(p: Vec2) {
    if (points.length >= (tool === 'route' ? MAX_ROUTE_POINTS : 256)) {setError('The point limit has been reached.'); return;}
    const last = points.at(-1);
    if (last && Math.hypot(last.x-p.x, last.y-p.y) <= 1e-9) {setError('Choose a different point.'); return;}
    updatePoints([...points, p], points.length);
  }
  function numericPoint(replace: boolean) {
    const p = {x:Number(entryX), y:Number(entryY)};
    if (!entryX.trim() || !entryY.trim() || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      setError('Enter finite X and Y coordinates in µm.'); return;
    }
    if (replace && selectedPoint >= 0) updatePoints(points.map((v, i) => i === selectedPoint ? p : v), selectedPoint);
    else append(p);
  }
  function reset() {
    dispatch({type:'reset', points:[], selected:-1}); setPreviewPoint(null); setEditing(null); setBase(null);
    setEntryX(''); setEntryY(''); setHover(null); setError(''); setNotice(''); drag.current = null; setOffset(null);
  }
  function choose(toolId: Tool) {
    reset(); setTool(toolId); setName(toolId === 'route' ? 'Drawn route' : 'Drawn polygon');
  }
  function startEditing() {
    if (selected?.kind !== 'polygon') return;
    const boundaries = [selected.vertices!, ...selected.holes!];
    const index = boundaries[ring] ? ring : 0;
    reset(); setTool('polygon'); setEditing({id:selected.id, ring:index}); setBase(scene);
    dispatch({type:'reset', points:boundaries[index].map(p => worldPoint(selected, p)), selected:0}); setName(selected.name);
    setHeight(String(selected.size.z)); setZ(String(selected.position.z));
  }
  function commit() {
    try {
      if (stale) throw new Error('The design changed during this sketch. Review the current design before applying.');
      if (!height.trim() || !z.trim() || !Number.isFinite(Number(height)) || !Number.isFinite(Number(z))) throw new Error('Enter a finite thickness and Z centre in µm.');
      let result: GeometryDocument, id: string;
      if (editing) {
        id = editing.id;
        result = editContour(scene, id, editing.ring, points, Number(height), Number(z));
      } else {
        const outline = tool === 'route' ? routeOutline(points, Number(width), radius.trim()?Number(radius):NaN) : points;
        ({ document: result, id } = addSketch(scene, outline, Number(height), Number(z), name));
      }
      if (onApply(result, editing ? 'Contour updated.' : `${name} added to the design.`)) {
        onSelect(id); reset(); setTool('move'); setNotice(editing ? 'Contour updated.' : `${name} placed.`);
      }
    } catch (reason) {setError((reason as Error).message);}
  }
  function moveDelta(action: Extract<Drag,{kind:'move'}>, end: Vec2): Vec2 {
    const next = snapPoint({ x: action.anchor.x + end.x - action.start.x, y: action.anchor.y + end.y - action.start.y }, snapSize);
    return {x: next.x - action.anchor.x, y: next.y - action.anchor.y};
  }
  function pointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0 || !event.isPrimary) return;
    svg.current?.focus({preventScroll:true});
    if (tool === 'move') {onSelect(null); return;}
    if (!stale) append(snapPoint(toWorld(event), snapSize));
    setHover(null);
  }
  function beginMove(event: PointerEvent<SVGGElement>, object: GeometryObject) {
    if (tool !== 'move' || event.button !== 0 || !event.isPrimary) return;
    event.stopPropagation(); svg.current?.focus({preventScroll:true}); onSelect(object.id); setError(''); setNotice('');
    const group = scene.components?.find(c => c.id === object.component_id);
    drag.current = { kind:'move', pointerId:event.pointerId, id:object.id, start:toWorld(event), anchor:group?.placement ?? object.position, original:scene };
    svg.current?.setPointerCapture(event.pointerId);
  }
  function beginPoint(event: PointerEvent<SVGCircleElement>, index: number) {
    if (!event.isPrimary || event.button !== 0) return;
    event.stopPropagation(); svg.current?.focus({preventScroll:true});
    if (stale) return;
    setHover(null); dispatch({type:'select', selected:index}); setError('');
    drag.current = {kind:'point', pointerId:event.pointerId, index, start:toWorld(event), original:scene};
    svg.current?.setPointerCapture(event.pointerId);
  }
  function draggedPosition(action: Extract<Drag,{kind:'point'}>, end: Vec2) {
    const dx = end.x - action.start.x, dy = end.y - action.start.y;
    const source = placed[action.index];
    if (!source || Math.hypot(dx,dy)*scale < 2) return null;
    return snapPoint({x:source.x+dx, y:source.y+dy}, snapSize);
  }
  function pointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!event.isPrimary) return;
    const p = toWorld(event), action = drag.current;
    if (action && action.pointerId !== event.pointerId) return;
    if (action?.kind === 'point') {
      const position = !stale && action.original === scene ? draggedPosition(action, p) : null;
      setPreviewPoint(position ? {index:action.index, position} : null);
    } else if (action?.kind === 'move') setOffset(moveDelta(action,p));
    else if (tool !== 'move') setHover(snapPoint(p,snapSize));
  }
  function pointerEnd(event: PointerEvent<SVGSVGElement>, cancelled: boolean) {
    const action = drag.current;
    if (!action || action.pointerId !== event.pointerId) return;
    drag.current = null; setOffset(null);
    if (action.kind === 'point') {
      setPreviewPoint(null);
      if (cancelled || stale || scene !== action.original) return;
      const position = draggedPosition(action,toWorld(event));
      if (position) updatePoints(placed.map((p,i) => i === action.index ? position : p), action.index);
      return;
    }
    if (cancelled) return;
    const end = toWorld(event);
    if (Math.hypot(end.x-action.start.x, end.y-action.start.y)*scale < 2) return;
    try {
      if (scene !== action.original) throw new Error('The design changed while moving. Select the solid again.');
      const delta = moveDelta(action,end);
      if (delta.x || delta.y) {
        const result = moveSelection(scene, action.id, delta);
        if (onApply(result,'Selection moved in XY.')) setNotice('Selection moved.');
      }
    } catch (reason) {setError((reason as Error).message);}
  }

  function removePoint() {
    if (selectedPoint < 0 || stale) return;
    updatePoints(placed.filter((_,i) => i !== selectedPoint), Math.max(0,selectedPoint-1));
  }
  function navigate(direction: 'undo' | 'redo') {
    drag.current = null; setPreviewPoint(null); setOffset(null); setHover(null); setError(''); setNotice('');
    svg.current?.focus({preventScroll:true});
    dispatch({type:direction});
  }
  function keyDown(event: KeyboardEvent<HTMLElement>) {
    if (tool === 'move' || event.defaultPrevented || event.altKey || event.repeat || event.nativeEvent.isComposing) return;
    const target = event.target;
    if (target instanceof Element && target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"]')) return;
    const key = event.key.toLowerCase();
    if (event.ctrlKey || event.metaKey) {
      if (!activeSketch) return;
      if (key === 'z') navigate(event.shiftKey ? 'redo' : 'undo');
      else if (key === 'y' && !event.shiftKey) navigate('redo');
      else return;
    } else if (!event.shiftKey && (key === 'delete' || key === 'backspace')) {
      drag.current = null; setPreviewPoint(null); svg.current?.focus({preventScroll:true}); removePoint();
    } else if (!event.shiftKey && key === 'escape' && activeSketch) {
      choose('move'); setNotice('Sketch cancelled.');
    } else return;
    event.preventDefault(); event.stopPropagation();
  }

  let preview: Vec2[] = [], previewProblem = '';
  // Preview a prospective segment at the pointer. Commit always uses only placed points.
  const tentative = tool !== 'move' && !editing && hover && (!points.length || Math.hypot(hover.x-points.at(-1)!.x,hover.y-points.at(-1)!.y)>1e-9) ? [...points,hover] : points;
  if (tool === 'route' && tentative.length >= 2) {
    try {preview = routeOutline(tentative,Number(width),radius.trim()?Number(radius):NaN);}
    catch (reason) {previewProblem = (reason as Error).message;}
  } else if (tool === 'polygon') preview = tentative;
  const visible = scene.objects.filter(o => o.visible).sort((a,b) => (a.position.z+a.size.z/2)-(b.position.z+b.size.z/2));
  const moved = new Set(member?.object_ids ?? (selectedId ? [selectedId] : []));
  const min = canvasPoint({x:-scene.domain.x/2,y:scene.domain.y/2});
  return <section className="plan-sketch" aria-label="XY layout editor"
    data-sketch-history={activeSketch ? 'active' : undefined} onKeyDown={keyDown}>
    <div className="plan-tools" role="group" aria-label="Drawing tool">
      {tools.map(item => <button key={item.id} aria-pressed={tool === item.id} disabled={activeSketch && tool !== item.id}
        onClick={() => {if (item.id !== tool) choose(item.id);}}>{item.name}</button>)}
      <label>Snap <select aria-label="XY snap spacing" value={snap} onChange={e => setSnap(e.target.value)}>
        <option value="off">Off</option>{['0.01','0.05','0.1','0.25','0.5','1'].map(s=><option key={s} value={s}>{s} µm</option>)}</select></label>
    </div>
    <svg ref={svg} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={tool === 'move' ? 'plan-board move' : 'plan-board'}
      tabIndex={0} role="img" aria-label="XY layout drawing surface" onPointerDown={pointerDown} onPointerMove={pointerMove}
      onPointerUp={e => pointerEnd(e,false)} onPointerCancel={e => pointerEnd(e,true)} onLostPointerCapture={e => pointerEnd(e,true)} onPointerLeave={() => setHover(null)}>
      <defs><pattern id={pattern} x={centre.x} y={centre.y} width={gridStep*scale} height={gridStep*scale} patternUnits="userSpaceOnUse">
        <path d={`M ${gridStep*scale} 0 H 0 V ${gridStep*scale}`} fill="none" stroke="#233749" strokeWidth="1.1"/></pattern></defs>
      <rect x={min.x} y={min.y} width={scene.domain.x*scale} height={scene.domain.y*scale} fill={`url(#${pattern})`} stroke="#456071"/>
      <path className="plan-axes" d={`M ${min.x} ${centre.y} h ${scene.domain.x*scale} M ${centre.x} ${min.y} v ${scene.domain.y*scale}`}/>
      <text x={WIDTH-14} y={HEIGHT-12} textAnchor="end">X → · Y ↑ · µm</text>
      <text x={14} y={22}>XY · {format(scene.domain.x)} × {format(scene.domain.y)} µm</text>
      <g transform={`translate(${centre.x} ${centre.y}) scale(${scale} ${-scale})`}>
        {visible.map(object=>{
          const d = offset && moved.has(object.id) ? offset : {x:0,y:0};
          return <g key={object.id} className={`plan-object ${object.id===selectedId?'current':member && object.component_id===member.id?'member':''} ${object.refractive_index<=1.6?'background':''}`}
            data-plan-object={object.id} transform={`translate(${object.position.x+d.x} ${object.position.y+d.y}) rotate(${object.rotation_deg})`}
            onPointerDown={e=>beginMove(e,object)}>{footprint(object)}</g>;
        })}
        {tool !== 'move' && <g pointerEvents="none">
          {preview.length >= 3 && <path className="plan-preview" d={path(preview,true)}/>}
          <path className="plan-centreline" d={path(tentative)}/>
        </g>}
      </g>
      {points.map((p,i)=>{const at=canvasPoint(p);return <circle key={i} cx={at.x} cy={at.y} r={i===selectedPoint?6:4.8} className={'plan-handle'+(i===selectedPoint?' selected':'')}
        data-plan-point={i} onPointerDown={e=>beginPoint(e,i)}/>;})}
    </svg>
    <div className="plan-readout"><span>{tool==='move' ? 'Drag a solid to move it in XY.' : editing ? 'Drag vertices or enter exact coordinates.' : tool==='route' ? (Number(radius)>0?'Click the centreline. Circular bends use polygon edges at ≤5° intervals.':'Click the centreline. Radius 0 uses miter joins.') : 'Click the outline vertices in sequence.'}</span>
      <span>{hover ? `X ${format(hover.x)} · Y ${format(hover.y)} µm` : `Grid ${format(gridStep)} µm`}</span></div>
    {tool === 'move' ? <div className="plan-selection"><strong>{selected?.name ?? 'Select a solid'}</strong>
      {member && <small>Dragging moves the full component and its layout ports.</small>}
      {selected?.kind === 'polygon' && <><label>Boundary <select aria-label="Plan contour" value={ring} onChange={e=>setRing(Number(e.target.value))}>
        <option value={0}>Outer outline</option>{selected.holes!.map((_,i)=><option value={i+1} key={i}>Hole {i+1}</option>)}</select></label>
        <button className="button secondary compact" onClick={startEditing}>Edit contour</button></>}
    </div> : <div className="plan-controls">
      <div className="plan-fields">
        {!editing && <label>Name<input aria-label="Sketch name" value={name} maxLength={80} onChange={e=>{setName(e.target.value);setError('');}}/></label>}
        {tool==='route' && <label>Width (µm)<input aria-label="Sketch route width" type="number" step=".05" value={width} onChange={e=>{setWidth(e.target.value);setError('');}}/></label>}
        {tool==='route' && <label>Bend radius (µm)<input aria-label="Sketch bend radius" type="number" min="0" step=".1" value={radius} onChange={e=>{setRadius(e.target.value);setError('');}}/></label>}
        <label>Height (µm)<input aria-label="Sketch extrusion height" type="number" step=".05" value={height} onChange={e=>{setHeight(e.target.value);setError('');}}/></label>
        <label>Z centre (µm)<input aria-label="Sketch Z centre" type="number" step=".05" value={z} onChange={e=>{setZ(e.target.value);setError('');}}/></label>
      </div>
      <div className="plan-point-entry"><label>Point<select aria-label="Selected sketch point" value={selectedPoint} disabled={!points.length}
        onChange={e=>dispatch({type:'select', selected:Number(e.target.value)})}>{!points.length&&<option value="-1">—</option>}{points.map((_,i)=><option key={i} value={i}>{i+1} of {points.length}</option>)}</select></label>
        <label>X (µm)<input aria-label="Sketch point X" type="number" step="any" value={entryX} onChange={e=>setEntryX(e.target.value)}/></label>
        <label>Y (µm)<input aria-label="Sketch point Y" type="number" step="any" value={entryY} onChange={e=>setEntryY(e.target.value)}/></label>
        <button type="button" disabled={stale} onClick={()=>numericPoint(false)}>Add point</button>
        <button type="button" disabled={!currentPoint || stale} onClick={()=>numericPoint(true)}>Update point</button>
        <button type="button" disabled={!currentPoint || stale} onClick={removePoint}>Remove point</button>
      </div>
      {editing && scene.objects.some(object => object.id === editing.id && object.component_id) && <p>Applying a manual contour detaches this component’s recipe.</p>}
      {stale && <p className="plan-warning">The design changed during this sketch. {editing ? 'Cancel and reopen the contour.' :
        <button onClick={()=>{setBase(scene);setNotice('Sketch now targets the current design.');}}>Use current design</button>}</p>}
      <div className="plan-actions"><button className="button primary compact" disabled={stale || points.length < (tool==='route'?2:3)} onClick={commit}>{editing?'Apply contour':tool==='route'?'Place route':'Place polygon'}</button>
        <button onClick={()=>navigate('undo')} disabled={!history.past.length} title="Undo a point edit (Ctrl/⌘ Z)" aria-keyshortcuts="Control+Z Meta+Z">Undo point edit</button>
        <button onClick={()=>navigate('redo')} disabled={!history.future.length} title="Redo a point edit (Ctrl/⌘ Shift Z)" aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y">Redo point edit</button>
        <button onClick={()=>choose('move')}>Cancel sketch</button>
      </div>
      {previewProblem && !error && <p className="plan-warning">{previewProblem}</p>}
    </div>}
    {error && <p className="plan-error" role="alert">{error}</p>}
    {notice && <p className="plan-notice" role="status">{notice}</p>}
  </section>;
}
