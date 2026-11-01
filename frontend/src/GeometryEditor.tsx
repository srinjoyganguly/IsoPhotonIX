import { useEffect, useRef, useState } from 'react';
import { GeometryViewport } from './GeometryViewport';
import { Study3DPanel } from './Study3DPanel';
import { PublicBenchmarks } from './PublicBenchmarks';
import { ModalPorts } from './ModalPorts';
import { CodeExport } from './CodeExport';
import { createGeometryObject, createGeometryTemplate, createSlabPreview, parseGeometryDocument,
  serializeGeometryDocument, validateGeometryDocument, type GeometryDocument, type GeometryObject, type Vec3 } from './geometryModel';
import './geometryEditor.css';

const STORAGE_KEY = 'photonics-workbench.geometry.v1';
const initialSelection = (scene: GeometryDocument) => scene.objects.find(object => object.id !== 'substrate')?.id ?? scene.objects[0]?.id ?? null;
type SlabConfig = { length_um: number; height_um: number; core_width_um: number; core_index: number };
type History = { past: GeometryDocument[]; present: GeometryDocument; future: GeometryDocument[] };
function initialDraft(): { document: GeometryDocument; warning: string; stored: string | null } {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(STORAGE_KEY);
    return { document: raw ? { ...parseGeometryDocument(raw), mode: 'draft-3d' } : createGeometryTemplate('ring'), warning: '', stored: raw };
  } catch {
    return { document: createGeometryTemplate('ring'), warning: 'The saved browser draft could not be loaded. It has not been overwritten. Edit or import a valid design to start a new draft.', stored: raw };
  }
}

function NumericInput({ label, value, onCommit, unit = 'µm', step = .1 }: {
  label: string; value: number; onCommit: (value: number) => void; unit?: string; step?: number;
}) {
  const [text, setText] = useState(String(value));
  const cancelled = useRef(false);
  useEffect(() => setText(String(value)), [value]);
  function commit() {
    if (!cancelled.current && text !== String(value)) onCommit(text.trim() ? Number(text) : NaN);
    cancelled.current = false;
    setText(String(value));
  }
  return <label className="parameter"><span>{label}</span><div className="input-wrap">
    <input aria-label={label} type="number" step={step} value={text} onChange={e => setText(e.target.value)}
      onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { cancelled.current = true; e.currentTarget.blur(); } }}/>
    <span>{unit}</span></div></label>;
}

function TextInput({ label, value, onCommit }: { label: string; value: string; onCommit: (value: string) => void }) {
  const [text, setText] = useState(value);
  const cancelled = useRef(false);
  useEffect(() => setText(value), [value]);
  return <input className="geo-text-input" aria-label={label} maxLength={80} value={text} onChange={e => setText(e.target.value)}
    onBlur={() => { if (!cancelled.current && text !== value) onCommit(text); cancelled.current = false; setText(value); }}
    onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { cancelled.current = true; e.currentTarget.blur(); } }}/>
}

export function GeometryEditor({ slabConfig, onSimulation }: { slabConfig: SlabConfig; onSimulation: () => void }) {
  const [initial] = useState(initialDraft);
  const [history, setHistory] = useState<History>({ past: [], present: initial.document, future: [] });
  const scene = history.present;
  const [selectedId, setSelectedId] = useState<string | null>(initialSelection(scene));
  const [message, setMessage] = useState(initial.warning);
  const [error, setError] = useState('');
  const [saveState, setSaveState] = useState('Browser draft · export JSON to keep a portable copy');
  const [autosavePaused, setAutosavePaused] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [studyMode,setStudyMode] = useState<'fields'|'modal'>('fields');
  const [addKind, setAddKind] = useState<GeometryObject['kind']>('box');
  const importRef = useRef<HTMLInputElement>(null);
  const changed = useRef(false);
  const paused = useRef(false);
  const lastStored = useRef(initial.stored);
  const selected = scene.objects.find(object => object.id === selectedId);

  function pauseAutosave() {
    // Pause synchronously as well as visually, before any pending scene effect can save.
    paused.current = true;
    setAutosavePaused(true);
    setSaveState('Autosave paused · another tab changed the saved draft');
  }

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY && event.key !== null) return;
      try {
        if (event.storageArea && event.storageArea !== localStorage) return;
        // Ignore stale events after an explicit save, and unrelated localStorage clears.
        if (localStorage.getItem(STORAGE_KEY) === lastStored.current) return;
      } catch { /* Conservatively preserve the open design if storage cannot be checked. */ }
      pauseAutosave();
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  useEffect(() => {
    if (!changed.current || paused.current) return;
    try {
      // Check before writing too: the other tab's storage event may still be queued.
      if (localStorage.getItem(STORAGE_KEY) !== lastStored.current) { pauseAutosave(); return; }
      const serialized = serializeGeometryDocument(scene);
      localStorage.setItem(STORAGE_KEY, serialized);
      lastStored.current = serialized;
      setSaveState('Draft saved in this browser');
    }
    catch { setSaveState('Browser storage unavailable · export JSON to save this design'); }
  }, [scene]);

  useEffect(() => {
    if (selectedId !== null && !scene.objects.some(object => object.id === selectedId)) setSelectedId(initialSelection(scene));
  }, [scene, selectedId]);

  function resumeSaving() {
    try {
      const serialized = serializeGeometryDocument(scene);
      localStorage.setItem(STORAGE_KEY, serialized);
      lastStored.current = serialized;
      changed.current = true;
      paused.current = false;
      setAutosavePaused(false);
      setSaveState('Draft saved in this browser');
      setMessage('This design is now the saved browser draft. Autosave resumed.');
    } catch {
      setSaveState('Browser storage unavailable · export JSON to save this design');
      setError('The browser draft could not be saved. Export geometry to keep a portable copy.');
    }
  }

  function apply(candidate: GeometryDocument, text = '') {
    try {
      const document = validateGeometryDocument(candidate);
      changed.current = true;
      setHistory(old => ({ past: [...old.past.slice(-49), old.present], present: document, future: [] }));
      setError(''); setMessage(text);
      return true;
    } catch (reason) { setError((reason as Error).message); return false; }
  }
  function editObject(patch: Partial<GeometryObject>) {
    if (!selected) return;
    apply({ ...scene, mode: 'draft-3d', objects: scene.objects.map(object => object.id === selectedId ? { ...object, ...patch } : object) });
  }
  function undo() {
    changed.current = true;
    setHistory(old => old.past.length ? { past: old.past.slice(0, -1), present: old.past[old.past.length - 1], future: [old.present, ...old.future] } : old);
    setError(''); setMessage('Previous design restored.');
  }
  function redo() {
    changed.current = true;
    setHistory(old => old.future.length ? { past: [...old.past, old.present], present: old.future[0], future: old.future.slice(1) } : old);
    setError(''); setMessage('Design change reapplied.');
  }
  function replace(document: GeometryDocument, text: string) {
    if (apply(document, text)) { setSelectedId(initialSelection(document)); setResetKey(key => key + 1); }
  }
  function addObject() {
    const object = createGeometryObject(addKind);
    if (apply({ ...scene, mode: 'draft-3d', objects: [...scene.objects, object] }, `${object.name} added at the origin.`)) setSelectedId(object.id);
  }
  function duplicate() {
    if (!selected) return;
    const object = { ...selected, id: createGeometryObject(selected.kind).id, name: `${selected.name.slice(0, 70)} copy` };
    if (apply({ ...scene, mode: 'draft-3d', objects: [...scene.objects, object] }, 'Duplicated in place. Change its position to separate the objects.')) setSelectedId(object.id);
  }
  function remove() {
    if (!selected) return;
    if (apply({ ...scene, mode: 'draft-3d', objects: scene.objects.filter(object => object.id !== selectedId) }, 'Object deleted. Undo restores it.')) setSelectedId(null);
  }
  function exportDesign() {
    const url = URL.createObjectURL(new Blob([serializeGeometryDocument(scene)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'photonics-geometry.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setMessage('Geometry exported. This file contains a design, not a simulated result.');
  }
  async function importDesign(file?: File) {
    if (!file) return;
    try {
      if (file.size > 250_000) throw new Error('Geometry files must be smaller than 250 KB.');
      replace({ ...parseGeometryDocument(await file.text()), mode: 'draft-3d' }, 'Geometry imported as a design draft. Undo restores your previous design.');
    } catch (reason) { setError(`Import failed: ${(reason as Error).message}`); }
    if (importRef.current) importRef.current.value = '';
  }
  function vectorInputs(label: string, vector: Vec3, onCommit: (vector: Vec3) => void) {
    return <div className="geo-vector">{(['x', 'y', 'z'] as const).map(axis => <NumericInput key={axis} label={`${label} ${axis.toUpperCase()}`} value={vector[axis]} onCommit={value => onCommit({ ...vector, [axis]: value })}/>)}</div>;
  }
  return <div className="geometry-editor">
    <div className="geo-intro"><div><span className="eyebrow">YOUR LOCAL PHOTONICS STUDIO</span><h1>Design in three dimensions.</h1><p>Build, inspect, and keep your geometry. Every dimension is in micrometres.</p></div><span className="geo-stage">GEOMETRY WORKSPACE · PREVIEW</span></div>
    <div className="geo-document-bar">
      <label className="geo-document-name"><span>DESIGN</span><TextInput label="Design name" value={scene.name} onCommit={name => apply({ ...scene, name })}/></label>
      <div className="geo-file-actions"><button className="button secondary compact" onClick={undo} disabled={!history.past.length}>Undo</button><button className="button secondary compact" onClick={redo} disabled={!history.future.length}>Redo</button><span className="geo-divider"/><button className="button secondary compact" onClick={() => importRef.current?.click()}>Import geometry</button><button className="button primary compact" onClick={exportDesign}>Export geometry</button><CodeExport value={scene} kind="geometry"/><button className="button secondary compact" onClick={()=>{setStudyMode('modal');setTimeout(()=>document.getElementById('geometry-studies')?.scrollIntoView({behavior:'smooth'}),0);}}>3D ports &amp; S-matrix</button><input ref={importRef} type="file" hidden aria-label="Import geometry file" accept=".json,application/json" onChange={e => importDesign(e.target.files?.[0])}/></div>
    </div>
    {autosavePaused && <div className="geo-message error-banner" role="alert"><span>Another tab changed the saved draft. Your current design is still open. Export it or choose Save this draft to resume saving.</span><button className="button secondary compact" onClick={resumeSaving}>Save this draft</button></div>}
    <div className="geo-workspace">
      <aside className="panel geo-tree-panel">
        <div className="panel-heading"><h2>Scene</h2><span className="tag">{scene.objects.length} objects</span></div>
        <div className="geo-template-list"><h3>START FROM A DESIGN</h3>
          <button onClick={() => replace(createGeometryTemplate('strip'), 'Strip template opened. Undo restores your previous design.')}><span aria-hidden="true">━</span><div>Strip waveguide<small>A finite dielectric guide</small></div></button>
          <button onClick={() => replace(createGeometryTemplate('coupler'), 'Coupler template opened. Undo restores your previous design.')}><span aria-hidden="true">═</span><div>Directional coupler<small>Two guides with an editable gap</small></div></button>
          <button onClick={() => replace(createGeometryTemplate('ring'), 'Ring template opened. Undo restores your previous design.')}><span aria-hidden="true">◎</span><div>Ring & bus<small>An annulus beside a straight guide</small></div></button>
        </div>
        <div className="geo-object-list" role="list" aria-label="Geometry objects">{scene.objects.map(object => <div role="listitem" className={`geo-object-row ${selectedId === object.id ? 'selected' : ''}`} key={object.id}>
          <button className="geo-object-select" aria-pressed={selectedId === object.id} onClick={() => setSelectedId(object.id)}><span className="geo-shape-symbol">{object.kind === 'box' ? '▱' : object.kind === 'ring' ? '◎' : '◉'}</span><span>{object.name}<small>{object.kind} · n = {object.refractive_index}</small></span></button>
          <button className="geo-visibility" aria-label={`${object.visible ? 'Hide' : 'Show'} ${object.name}`} title={`${object.visible ? 'Hide' : 'Show'} in viewport`} onClick={() => apply({ ...scene, objects: scene.objects.map(item => item.id === object.id ? { ...item, visible: !item.visible } : item) })}>{object.visible ? '◉' : '○'}</button>
        </div>)}{!scene.objects.length && <p className="geo-empty">Your scene is empty. Add a shape below.</p>}</div>
        <div className="geo-add"><select aria-label="New shape" value={addKind} onChange={e => setAddKind(e.target.value as GeometryObject['kind'])}><option value="box">Box</option><option value="cylinder">Cylinder</option><option value="ring">Ring</option></select><button className="button secondary compact" onClick={addObject} disabled={scene.objects.length >= 100}>+ Add</button></div>
        <div className="geo-tree-note">Select a shape in the scene or the viewport. Visibility changes only the preview. Material indices enter the solver only after the supported waveguide study validates.</div>
      </aside>
      <div className="geo-stage-column">
        <GeometryViewport document={scene} selectedId={selectedId} onSelect={setSelectedId} resetKey={resetKey}/>
        <div className="geo-science-boundary"><div><strong>{scene.mode === 'slab-preview' ? 'Extruded view of the 2D slab' : '3D geometry'}</strong><p>{scene.mode === 'slab-preview' ? 'The slab solver is invariant along z. The displayed thickness is illustrative and does not change its physics.' : 'The study below validates this exact geometry for the supported finite-height waveguide. Rings, couplers, and other shapes remain editable drafts.'}</p></div><button onClick={onSimulation}>Open tested 2D simulation →</button></div>
        <div className="geo-status" role="status"><span>{saveState}</span><span>X propagation · Y lateral · Z vertical</span></div>
      </div>
      <aside className="panel geo-inspector">
        <div className="panel-heading"><h2>Inspector</h2><span className="tag">{selected?.kind || 'DOMAIN'}</span></div>
        {selected ? <div className="geo-properties" key={selected.id}>
          <label className="parameter"><span>Object name</span><TextInput label="Object name" value={selected.name} onCommit={name => editObject({ name })}/></label>
          <h3>POSITION</h3>{vectorInputs('Position', selected.position, position => editObject({ position }))}
          <h3>DIMENSIONS</h3>{selected.kind === 'box' ? vectorInputs('Size', selected.size, size => editObject({ size })) : <><NumericInput label="Outer radius" value={selected.radius} onCommit={radius => editObject({ radius })}/>{selected.kind === 'ring' && <NumericInput label="Inner radius" value={selected.inner_radius} onCommit={inner_radius => editObject({ inner_radius })}/>}<NumericInput label="Thickness Z" value={selected.size.z} onCommit={z => editObject({ size: { ...selected.size, z } })}/></>}
          <NumericInput label="Rotation Z" unit="deg" step={5} value={selected.rotation_deg} onCommit={rotation_deg => editObject({ rotation_deg })}/>
          <NumericInput label="Refractive index" unit="n" step={.05} value={selected.refractive_index} onCommit={refractive_index => editObject({ refractive_index })}/>
          <div className="geo-object-actions"><button onClick={duplicate} disabled={scene.objects.length >= 100}>Duplicate</button><button className="geo-delete" onClick={remove}>Delete object</button></div>
        </div> : <p className="geo-empty">Select a shape to edit its dimensions, position, and material.</p>}
        <details className="geo-domain"><summary>Design domain</summary>{vectorInputs('Domain', scene.domain, domain => apply({ ...scene, mode: 'draft-3d', domain }))}<p>Objects must fit inside this centred box. Arbitrary overlaps remain drafts. A validated waveguide uses a centred core and optional substrate meeting its lower surface; its adapters define the material placement.</p></details>
        <button className="geo-slab-preview" onClick={() => replace(createSlabPreview(slabConfig), 'Current validated 2D slab opened as an illustrative extrusion.')}>Preview current 2D slab</button>
      </aside>
    </div>
    <nav className="geo-study-tabs" id="geometry-studies" aria-label="3D study type"><button aria-pressed={studyMode==='fields'} onClick={()=>setStudyMode('fields')}>3D field study</button><button aria-pressed={studyMode==='modal'} onClick={()=>setStudyMode('modal')}>3D modal S-parameters</button></nav>
    <div hidden={studyMode!=='fields'}><Study3DPanel geometry={scene} onOpenGeometry={document => replace(document, 'Runnable 3D waveguide opened. Choose a CPU or GPU target in the study below.')} onEditGeometry={document => apply(document, 'Substrate fitted to the core. Review the geometry and study validation before running.')}/></div>
    <div hidden={studyMode!=='modal'}><ModalPorts geometry={scene} active={studyMode==='modal'} onOpenGeometry={document=>replace(document,'Finite-height modal study opened. Inspect the geometry and explicitly select its execution target.')}/></div>
    <PublicBenchmarks/>
    {error && <div className="geo-message error-banner" role="alert"><span>{error} The last valid geometry is retained.</span><button aria-label="Dismiss geometry error" onClick={() => setError('')}>×</button></div>}
    {message && <div className="geo-message geo-notice" role="status"><span>{message}</span><button aria-label="Dismiss geometry notice" onClick={() => setMessage('')}>×</button></div>}
  </div>;
}
