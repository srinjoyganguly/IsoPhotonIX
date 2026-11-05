import { useEffect, useRef, useState } from 'react';
import { request } from './api';
import { validateGeometryDocument, type GeometryDocument, type MaterialRecord, type ComponentInstance } from './geometryModel';
import './componentLibrary.css';
import { PortAlignment } from './PortAlignment';

type Parameter = { key:string; label:string; unit:string; min:number; max:number; step:number; default:number };
type Family = { id:string; name:string; category:string; summary:string; parameters:Parameter[]; defaults:Record<string,number>; limitations:string[]; simulation_support:'bounded-layout-fields'|'bounded-taper-fields' };
type Catalog = { components:Family[]; materials:MaterialRecord[] };
type Placement = { x:number; y:number; z:number; rotation_deg:number };
type Generated = { geometry:GeometryDocument; component:ComponentInstance };
type Props = { document:GeometryDocument; selectedId:string|null; editRequest?:number; onFamilies:(names:Record<string,string>)=>void; onStudy:()=>void; onApply:(geometry:GeometryDocument,message:string,selectedId:string|null,fit:boolean)=>boolean };
const origin:Placement = {x:0,y:0,z:0,rotation_deg:0};
const title=(text:string)=>text.replaceAll('_',' ').replaceAll('-',' ').replace(/\b\w/g,s=>s.toUpperCase());

/** Schematic family glyphs, never simulated fields or performance curves. */
function ComponentGlyph({family}:{family:string}) {
  const common={fill:'none',stroke:'currentColor',strokeWidth:6,strokeLinecap:'round' as const,strokeLinejoin:'round' as const};
  let drawing;
  if(family==='pulley_ring')drawing=<><circle cx="60" cy="29" r="20"/><path d="M7 36L33 52Q60 72 87 52L113 36"/></>;
  else if(family==='euler_bend')drawing=<path d="M8 66C54 66 83 48 83 7"/>;
  else if(family==='strip_slot_converter')drawing=<><path d="M9 40H42L111 24M42 40L111 56"/></>;
  else if(family==='mmi_1x4')drawing=<><path d="M8 40H41M41 10H78V70H41ZM78 16H112M78 32H112M78 48H112M78 64H112"/></>;
  else if(family==='fan_grating')drawing=<><path d="M8 40H30M30 40L95 10M30 40L95 70"/>{[55,68,81,94].map(x=><path key={x} d={`M${x} ${40-(x-30)*.43}Q${x+7} 40 ${x} ${40+(x-30)*.43}`}/>)}</>;
  else if(family==='tapered_coupler')drawing=<><path d="M8 28L110 12V28ZM8 52H110L8 68Z"/></>;
  else if(family==='phc_line_defect')drawing=<><path d="M8 40H112"/>{[17,63].map(y=>[25,42,59,76,93].map(x=><circle key={x+'-'+y} cx={x} cy={y} r="3"/>))}</>;
  else if(family.includes('ring')||family==='racetrack')drawing=<><path d="M10 63H110"/><rect x={family==='racetrack'?34:40} y="12" width={family==='racetrack'?52:40} height="40" rx="20"/>{family==='ring_add_drop'&&<path d="M10 5H110"/>}</>;
  else if(family==='mzi')drawing=<><path d="M8 40H25L42 18H78L95 40H112M25 40L42 62H78L95 40"/></>;
  else if(family==='taper')drawing=<path d="M10 31L110 17V63L10 49Z"/>;
  else if(family==='bend')drawing=<path d="M10 65H40Q80 65 80 25V8"/>;
  else if(family==='s_bend')drawing=<path d="M8 60H30C58 60 60 20 88 20H112"/>;
  else if(family==='serpentine')drawing=<path d="M8 65H96Q114 65 114 50Q114 35 96 35H24Q6 35 6 20Q6 5 24 5H112"/>;
  else if(family==='y_branch'||family.startsWith('mmi'))drawing=<><path d="M8 40H42M78 27H112M78 53H112"/><path d={family==='y_branch'?'M42 40Q68 40 78 27M42 40Q68 40 78 53':'M42 16H78V64H42Z'}/>{family==='mmi_2x2'&&<path d="M8 20H42"/>}</>;
  else if(family==='crossing')drawing=<path d="M8 40H112M60 8V72"/>;
  else if(['bragg','grating_coupler','dbr_cavity','nanobeam_cavity'].includes(family))drawing=<><path d="M8 40H112"/>{[25,38,51,69,82,95].map(x=><path key={x} d={`M${x} 21V59`}/>)}{family==='nanobeam_cavity'&&<circle cx="60" cy="40" r="7"/>}</>;
  else if(family==='slot'||family==='directional_coupler')drawing=<><path d="M10 27H110M10 53H110"/>{family==='directional_coupler'&&<path d="M10 10L30 27M90 53L110 70"/>}</>;
  else drawing=<><path d="M10 40H110"/>{family==='rib'&&<path d="M10 56H110" opacity=".35"/>}</>;
  return <svg className="component-glyph" viewBox="0 0 120 80" aria-hidden="true"><g {...common}>{drawing}</g></svg>;
}

export function ComponentLibrary({document,selectedId,onApply,onStudy,onFamilies,editRequest=0}:Props) {
  const [catalog,setCatalog]=useState<Catalog|null>(null),[loadError,setLoadError]=useState(''),[retry,setRetry]=useState(0);
  const [query,setQuery]=useState(''),[category,setCategory]=useState('all'),[familyId,setFamilyId]=useState('taper');
  const [parameters,setParameters]=useState<Record<string,string>>({}),[placement,setPlacement]=useState<Record<keyof Placement,string>>({x:'0',y:'0',z:'0',rotation_deg:'0'});
  const [materialId,setMaterialId]=useState('generic-core'),[fitDomain,setFitDomain]=useState(false),[editingId,setEditingId]=useState<string|null>(null);
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const sceneRef=useRef(document);sceneRef.current=document;
  const details=useRef<HTMLDetailsElement>(null);
  useEffect(()=>{let active=true;setLoadError('');request<Catalog>('/api/library').then(value=>{if(active){setCatalog(value);onFamilies(Object.fromEntries(value.components.map(c=>[c.id,c.name])));const first=value.components.find(c=>c.id==='taper')??value.components[0];if(first){setFamilyId(first.id);setParameters(Object.fromEntries(Object.entries(first.defaults).map(([key,v])=>[key,String(v)])));}}}).catch(e=>{if(active)setLoadError(e.message);});return()=>{active=false;};},[retry,onFamilies]);
  const family=catalog?.components.find(item=>item.id===familyId);
  const studyTaper = document.components?.length === 1 && document.components[0].family === 'taper'
    && document.objects.length === 1 && Object.values(document.components[0].placement).every(value => value === 0);
  const selectedComponent=document.components?.find(component=>component.object_ids.includes(selectedId??''));
  const currentInstance=document.components?.find(component=>component.id===editingId);
  const materialMap=new Map((catalog?.materials??[]).map(material=>[material.id,material]));
  for(const material of document.materials??[])materialMap.set(material.id,material);
  const materials=[...materialMap.values()];
  const families=(catalog?.components??[]).filter(item=>(category==='all'||item.category===category)&&`${item.name} ${item.summary} ${item.category}`.toLowerCase().includes(query.trim().toLowerCase()));
  function choose(item:Family) {setFamilyId(item.id);setParameters(Object.fromEntries(Object.entries(item.defaults).map(([key,value])=>[key,String(value)])));setEditingId(null);setError('');setNotice('');}
  function editInstance(instance:ComponentInstance) {
    if(!catalog?.components.some(item=>item.id===instance.family)){setError('This component family is unavailable in this library version. Its generated solids remain editable.');return;}
    setFamilyId(instance.family);setParameters(Object.fromEntries(Object.entries(instance.parameters).map(([key,value])=>[key,String(value)])));
    setPlacement(Object.fromEntries(Object.entries(instance.placement).map(([key,value])=>[key,String(value)])) as Record<keyof Placement,string>);
    setMaterialId(instance.material_id);setEditingId(instance.id);setError('');setNotice('');
    if(details.current)details.current.open=true;
  }
  useEffect(()=>{if(editRequest&&selectedComponent&&catalog){editInstance(selectedComponent);details.current?.scrollIntoView({behavior:'smooth',block:'start'});}},[editRequest,catalog]);
  async function generate(action:'new'|'add'|'update') {
    if(!family||busy)return;
    setBusy(true);setError('');setNotice('');
    const snapshot=JSON.stringify(document);
    try {
      const numbers=Object.fromEntries(Object.entries(parameters).map(([key,value])=>[key,value.trim()?Number(value):NaN]));
      const located=Object.fromEntries(Object.entries(placement).map(([key,value])=>[key,value.trim()?Number(value):NaN]));
      if([...Object.values(numbers),...Object.values(located)].some(value=>!Number.isFinite(value)))throw new Error('Enter a finite number in every component and placement field.');
      if(action==='update'&&!currentInstance)throw new Error('This component was detached or removed. Choose an existing component again.');
      const payload={component_id:family.id,parameters:numbers,placement:located,material_id:materialId,
        ...(action==='new'?{material:materialMap.get(materialId)}:{geometry:document,fit_domain:fitDomain,...(action==='update'?{replace_component_id:editingId}:{})})};
      const result=await request<Generated>(action==='new'?'/api/library/build':'/api/library/place',payload);
      if(JSON.stringify(sceneRef.current)!==snapshot)throw new Error('The design changed while this component was building. Your current work was retained; review the placement and try again.');
      const valid=validateGeometryDocument(result.geometry);
      const message=action==='new'?`${family.name} design opened. Undo restores your previous layout.`:action==='update'?`${family.name} parameters updated as one undoable change.`:`${family.name} added to the layout.`;
      if(onApply(valid,message,result.component.object_ids[0]??null,action==='new'||fitDomain)){
        setEditingId(result.component.id);setNotice(message);
      }
    }catch(reason){setError((reason as Error).message);}finally{setBusy(false);}
  }
  return <details className="component-library panel" ref={details}>
    <summary><span className="library-heading-icon" aria-hidden="true">▦</span><span><strong>Photonic component library</strong><small>{catalog?`${catalog.components.length} parametric families · reusable layouts · custom materials`:'Local component catalog'}</small></span><span className="library-preview-label">DESIGN LIBRARY</span><span className="library-open-label">Browse components ↓</span></summary>
    <div className="component-library-body">
      <div className="library-intro"><div><h2>Build the next circuit.</h2><p>Choose a component, tune its dimensions, and place it in your 3D layout. Generated geometry and recipes travel with your exports.</p></div><span className="library-design-badge">Geometry designs · solver coverage is checked separately</span></div>
      {loadError?<div className="frame-error" role="alert">Component library unavailable: {loadError}<button onClick={()=>setRetry(value=>value+1)}>Retry library</button></div>:!catalog?<p role="status">Reading local component definitions…</p>:<>
        <div className="library-controls"><label>Find a component<input type="search" aria-label="Search photonic components" placeholder="Try taper, cavity, grating…" value={query} onChange={event=>setQuery(event.target.value)}/></label><label>Category<select aria-label="Component category" value={category} onChange={event=>setCategory(event.target.value)}><option value="all">All categories</option>{[...new Set(catalog.components.map(item=>item.category))].map(value=><option key={value} value={value}>{title(value)}</option>)}</select></label><span>{families.length} of {catalog.components.length} designs</span></div>
        <PortAlignment scene={document} selected={selectedComponent} names={Object.fromEntries(catalog.components.map(c=>[c.id,c.name]))} onApply={onApply}/>
        <div className="library-layout"><div className="library-cards" role="list" aria-label="Photonic components">{families.map(item=><div role="listitem" key={item.id}><button className={`component-card ${item.id===familyId?'selected':''}`} aria-label={`Choose component ${item.name}`} aria-pressed={item.id===familyId} disabled={busy} onClick={()=>choose(item)}><ComponentGlyph family={item.id}/><span className="component-card-category">{title(item.category)}</span><strong>{item.name}</strong><span className="component-card-summary">{item.summary}</span><span className="component-card-status">{item.simulation_support === 'bounded-taper-fields' ? 'Taper fields and modal ports' : 'Finite-layout fields'}</span></button></div>)}{!families.length&&<p className="library-no-results">No components match this search. Try another term or category.</p>}</div>
          {family&&<aside className="library-config" aria-label="Component parameters"><div className="library-config-title"><span className="eyebrow">PARAMETRIC DESIGN</span><h3>{family.name}</h3><p>{family.summary}</p></div>
            {selectedComponent&&<button className="library-edit-selected" disabled={busy} onClick={()=>editInstance(selectedComponent)}>Edit selected component</button>}
            {editingId&&<p className="library-editing">Editing component {editingId.slice(0,18)}{!currentInstance?' · no longer in this design':''}. Values remain a draft until applied.</p>}
            <fieldset disabled={busy}><legend>Dimensions</legend><div className="library-parameters">{family.parameters.map(parameter=><label key={parameter.key}>{parameter.label}<span className="library-number"><input aria-label={`Component ${parameter.label}`} type="number" min={parameter.min} max={parameter.max} step={parameter.step} value={parameters[parameter.key]??String(parameter.default)} onChange={event=>setParameters(old=>({...old,[parameter.key]:event.target.value}))}/><small>{parameter.unit}</small></span></label>)}</div>
            <label className="library-material-label">Component material<select aria-label="Component material" value={materialId} onChange={event=>setMaterialId(event.target.value)}>{materials.map(material=><option key={material.id} value={material.id}>{material.name}</option>)}</select></label>
            <details className="library-placement"><summary>Placement & layout</summary><div className="library-parameters">{(['x','y','z','rotation_deg'] as const).map(key=><label key={key}>{key==='rotation_deg'?'Rotation Z':`Position ${key.toUpperCase()}`}<span className="library-number"><input aria-label={`Component placement ${key}`} type="number" step={key==='rotation_deg'?5:.25} value={placement[key]} onChange={event=>setPlacement(old=>({...old,[key]:event.target.value}))}/><small>{key==='rotation_deg'?'deg':'µm'}</small></span></label>)}</div><label className="library-fit"><input type="checkbox" checked={fitDomain} onChange={event=>setFitDomain(event.target.checked)}/>Enlarge domain to fit when adding or updating</label></details>
            <div className="library-actions"><button className="button primary compact" onClick={()=>void generate('add')}>Add to layout</button><button className="button secondary compact" onClick={()=>void generate('new')}>Open as new design</button>{editingId&&<button className="button primary compact" disabled={!currentInstance} onClick={()=>void generate('update')}>Update component</button>}</div></fieldset>
            {busy&&<p role="status">Building and validating exact geometry…</p>}
            {error&&<p className="library-error" role="alert">{error} The previous design is retained.</p>}{notice&&<div className="library-notice" role="status">{notice}<button onClick={()=>window.document.querySelector('.geo-stage-column')?.scrollIntoView({behavior:'smooth',block:'center'})}>Inspect in 3D ↓</button><button onClick={onStudy}>Field study ↓</button></div>}
            {document.schema_version===2 && <button className="button secondary compact" onClick={onStudy}>{studyTaper?'Set up taper simulation':'Set up layout simulation'}</button>}
            <details className="library-limits"><summary>Model and simulation scope</summary><p>These templates are original, editable geometry designs. A library entry does not certify electromagnetic performance or fabrication readiness.</p>{family.limitations.map((limit,index)=><p key={index}>{limit}</p>)}<p>{family.simulation_support === 'bounded-taper-fields' ? 'Open a single taper as a new design, then set up its 3D field study. The service validates geometry, material and grid before execution.' : 'The finite layout study uses the solids as drawn and a local Ey dipole. Its validation checks the mesh, materials, source and computation budget.'}</p></details>
          </aside>}
        </div>
      </>}
    </div>
  </details>;
}
