import { useEffect, useState } from 'react';
import { request } from './api';
import { detachComponent, evaluateMaterial, upgradeToV2, validateGeometryDocument, type GeometryDocument, type GeometryObject, type MaterialRecord } from './geometryModel';

type Props={document:GeometryDocument;object:GeometryObject;onApply:(document:GeometryDocument,message:string)=>boolean};
const newId=()=>`material-${crypto.randomUUID()}`;
function csvTable(text:string) {
  const lines=text.split(/\r?\n/).map(line=>line.trim()).filter(line=>line&&!line.startsWith('#'));
  if(lines[0]?.toLowerCase().replaceAll(' ','')==='wavelength_um,n,k')lines.shift();
  if(lines.length<2)throw new Error('Use at least two rows: wavelength_um,n,k.');
  const rows=lines.map((line,index)=>{const fields=line.split(',').map(item=>item.trim());if(fields.length!==3||fields.some(field=>!field||!Number.isFinite(Number(field))))throw new Error(`Material row ${index+1} needs three finite numbers: wavelength_um,n,k.`);return fields.map(Number);});
  return {wavelength_um:rows.map(row=>row[0]),n:rows.map(row=>row[1]),k:rows.map(row=>row[2])};
}
export function MaterialEditor({document,object,onApply}:Props) {
  const [catalog,setCatalog]=useState<MaterialRecord[]>([]),[catalogError,setCatalogError]=useState('');
  const [selected,setSelected]=useState(object.material_id??''),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const [model,setModel]=useState<'constant-nk'|'tabulated-nk'>('constant-nk');
  const [name,setName]=useState('Custom optical material'),[index,setIndex]=useState(String(object.refractive_index)),[loss,setLoss]=useState('0'),[reference,setReference]=useState('1.55');
  const [source,setSource]=useState('User-defined model'),[notes,setNotes]=useState(''),[table,setTable]=useState('wavelength_um,n,k\n1.45,2,0\n1.55,2,0\n1.65,2,0');
  const [editingMaterial,setEditingMaterial]=useState<string|null>(null);
  useEffect(()=>{let active=true;request<{materials:MaterialRecord[]}>('/api/library').then(value=>{if(active)setCatalog(value.materials);}).catch(e=>{if(active)setCatalogError(e.message);});return()=>{active=false;};},[]);
  useEffect(()=>{setSelected(object.material_id??'');},[object.material_id]);
  const records=new Map(catalog.map(material=>[material.id,material]));for(const material of document.materials??[])records.set(material.id,material);
  const assigned=document.materials?.find(material=>material.id===object.material_id);
  function useCandidate(candidate:GeometryDocument,message:string) {
    try {const valid=validateGeometryDocument(candidate);if(onApply(valid,message)){setError('');setNotice(message);return true;}}catch(reason){setError((reason as Error).message);}return false;
  }
  function applyMaterial(material:MaterialRecord,updateShared=false) {
    try {
      let candidate=upgradeToV2(document);
      const value=evaluateMaterial(material,material.reference_wavelength_um);
      if(!updateShared)candidate=detachComponent(candidate,object.id);
      // A shared record is edited atomically, with all references updated.
      // Detach affected component recipes: their generated material provenance
      // must not silently describe a different material definition.
      if(updateShared){for(const item of candidate.objects.filter(item=>item.material_id===material.id))candidate=detachComponent(candidate,item.id);}
      candidate={...candidate,materials:[...(candidate.materials??[]).filter(item=>item.id!==material.id),material],objects:candidate.objects.map(item=>(updateShared?item.material_id===material.id:item.id===object.id)?{...item,material_id:material.id,refractive_index:value.n}:item)};
      const count=candidate.objects.filter(item=>item.material_id===material.id).length;
      if(useCandidate(candidate,updateShared?`Material updated for ${count} linked solid${count===1?'':'s'}. Component recipes were detached.`:`${material.name} assigned. Material samples are retained in the design.`))setSelected(material.id);
    }catch(reason){setError((reason as Error).message);}
  }
  function loadForEditing(material:MaterialRecord) {
    setEditingMaterial(material.id);setName(material.name);setModel(material.model);setReference(String(material.reference_wavelength_um));setSource(material.source);setNotes(material.notes);setError('');setNotice('');
    if(material.model==='constant-nk'){setIndex(String(material.n));setLoss(String(material.k));}
    else setTable(['wavelength_um,n,k',...material.wavelength_um.map((w,i)=>`${w},${material.n[i]},${material.k[i]}`)].join('\n'));
  }
  function save(updateShared:boolean) {
    try {
      const common={id:updateShared&&editingMaterial?editingMaterial:newId(),name,model,reference_wavelength_um:Number(reference),source,notes};
      if(!reference.trim()||!Number.isFinite(common.reference_wavelength_um))throw new Error('Enter a finite reference wavelength.');
      let material:MaterialRecord;
      if(model==='constant-nk'){
        if(!index.trim()||!loss.trim()||![Number(index),Number(loss)].every(Number.isFinite))throw new Error('Enter finite n and k values.');
        material={...common,model:'constant-nk',n:Number(index),k:Number(loss)};
      }else material={...common,model:'tabulated-nk',...csvTable(table)};
      applyMaterial(material,updateShared);
    }catch(reason){setError((reason as Error).message);}
  }
  return <section className="material-editor" aria-label="Optical material editor">
    <h3>MATERIAL MODEL</h3>
    <label>Material<select aria-label="Object material" value={selected} onChange={event=>setSelected(event.target.value)}><option value="">Custom scalar index</option>{[...records.values()].map(material=><option key={material.id} value={material.id}>{material.name}</option>)}</select></label>
    <button className="button secondary compact" disabled={!records.has(selected)} onClick={()=>{const material=records.get(selected);if(material)applyMaterial(material);}}>Assign material to solid</button>
    {catalogError&&<p className="material-note">Material catalog unavailable. Custom material entry remains available.</p>}
    {assigned&&<div className="material-assigned"><strong>{assigned.name}</strong><span>{assigned.model==='constant-nk'?`n = ${assigned.n}, k = ${assigned.k}`:`${assigned.wavelength_um.length} n,k samples`} · reference {assigned.reference_wavelength_um} µm</span><p>{assigned.source}</p><button onClick={()=>loadForEditing(assigned)}>Edit assigned material</button></div>}
    <details className="material-custom"><summary>Custom material & CSV samples</summary>
      {editingMaterial&&<p className="material-note">Editing a design material. Save as new to leave its other users unchanged.</p>}
      <label>Name<input aria-label="Custom material name" value={name} maxLength={80} onChange={event=>setName(event.target.value)}/></label>
      <label>Optical model<select aria-label="Custom material model" value={model} onChange={event=>setModel(event.target.value as typeof model)}><option value="constant-nk">Constant n and k</option><option value="tabulated-nk">Tabulated n and k</option></select></label>
      <label>Reference wavelength (µm)<input aria-label="Material reference wavelength" type="number" step=".05" value={reference} onChange={event=>setReference(event.target.value)}/></label>
      {model==='constant-nk'?<div className="material-constant"><label>Index n<input aria-label="Material index n" type="number" step=".01" value={index} onChange={event=>setIndex(event.target.value)}/></label><label>Extinction k<input aria-label="Material extinction k" type="number" min="0" step=".001" value={loss} onChange={event=>setLoss(event.target.value)}/></label></div>:<><label>Samples: wavelength_um,n,k<textarea aria-label="Material CSV samples" rows={7} spellCheck={false} value={table} onChange={event=>setTable(event.target.value)}/></label><label className="material-file">Load local CSV<input aria-label="Import material CSV" type="file" accept=".csv,text/csv,text/plain" onChange={async event=>{const file=event.target.files?.[0];if(!file)return;try{if(file.size>100_000)throw new Error('Material CSV must be smaller than 100 KB.');setTable(await file.text());setError('');}catch(reason){setError((reason as Error).message);}event.target.value='';}}/></label></>}
      <label>Source / citation<input aria-label="Material source" value={source} maxLength={500} onChange={event=>setSource(event.target.value)}/></label>
      <label>Notes<textarea aria-label="Material notes" rows={3} value={notes} maxLength={2000} onChange={event=>setNotes(event.target.value)}/></label>
      <div className="material-actions"><button className="button primary compact" onClick={()=>save(false)}>Save as new & assign</button>{editingMaterial&&document.materials?.some(material=>material.id===editingMaterial)&&<button className="button secondary compact" onClick={()=>save(true)}>Update shared material</button>}</div>
    </details>
    <p className="material-note">Passive isotropic design records. Tabulated values use interpolation within their recorded band. Current electromagnetic studies reject loss, tabulation and active gain models rather than ignoring them.</p>
    {error&&<p className="library-error" role="alert">{error} Previous material and geometry retained.</p>}{notice&&<p className="library-notice" role="status">{notice}</p>}
  </section>;
}
