import { useRef, useState } from 'react';
import { request } from './api';
import { validateGeometryDocument, type ComponentInstance, type GeometryDocument } from './geometryModel';

type Props = {
  scene: GeometryDocument;
  selected?: ComponentInstance;
  names: Record<string, string>;
  onApply: (geometry: GeometryDocument, message: string, selectedId: string | null, fit: boolean) => boolean;
};

export function PortAlignment({ scene, selected, names, onApply }: Props) {
  const [sourceId, setSourceId] = useState('');
  const [targetId, setTargetId] = useState('');
  const [sourcePort, setSourcePort] = useState('');
  const [targetPort, setTargetPort] = useState('');
  const [fit, setFit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState('');
  const latest = useRef(scene);
  latest.current = scene;
  const parts = scene.components?.filter(c => c.ports.length) ?? [];
  const source = parts.find(c => c.id === sourceId) ?? parts.find(c => c.id === selected?.id) ?? parts.at(-1);
  const targets = parts.filter(c => c.id !== source?.id);
  const target = targets.find(c => c.id === targetId) ?? targets[0];
  const from = source?.ports.find(p => p.name === sourcePort) ?? source?.ports[0];
  const to = target?.ports.find(p => p.name === targetPort) ?? target?.ports.find(p => p.name === 'out') ?? target?.ports[0];
  if (!source || !target || !from || !to) return null;
  const moving = source, fixed = target, start = from, end = to;
  const describe = (c: ComponentInstance) => `${names[c.family] || c.family} · ${c.id.slice(-6)}`;

  async function align() {
    setBusy(true);
    setFeedback('');
    try {
      const result = await request<{ geometry: GeometryDocument }>('/api/library/align', {
        geometry: scene, moving_component: moving.id, moving_port: start.name,
        fixed_component: fixed.id, fixed_port: end.name, fit_domain: fit,
      });
      if (latest.current !== scene) throw new Error('The layout changed during alignment. Select the ports again.');
      const document = validateGeometryDocument(result.geometry);
      const message = `Aligned ${describe(moving)} (${start.name}) to ${describe(fixed)} (${end.name}).`;
      if (onApply(document, message, moving.object_ids[0], fit)) setFeedback(message);
    } catch (reason) {
      setFeedback((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return <details className="port-alignment" onChange={() => setFeedback('')}>
    <summary>Align component ports</summary>
    <p>Move and rotate one component to meet an equal-width port. The fixed component stays in place.</p>
    <fieldset disabled={busy}>
      <div className="port-pair">
        <label>Moving component<select aria-label="Moving component" value={moving.id} onChange={e => { setSourceId(e.target.value); setSourcePort(''); }}>
          {parts.map(c => <option key={c.id} value={c.id}>{describe(c)}</option>)}
        </select></label>
        <label>Moving port<select aria-label="Moving port" value={start.name} onChange={e => setSourcePort(e.target.value)}>
          {moving.ports.map(p => <option key={p.name} value={p.name}>{p.name} · {p.width_um} µm</option>)}
        </select></label>
        <label>Fixed component<select aria-label="Fixed component" value={fixed.id} onChange={e => { setTargetId(e.target.value); setTargetPort(''); }}>
          {targets.map(c => <option key={c.id} value={c.id}>{describe(c)}</option>)}
        </select></label>
        <label>Fixed port<select aria-label="Fixed port" value={end.name} onChange={e => setTargetPort(e.target.value)}>
          {fixed.ports.map(p => <option key={p.name} value={p.name}>{p.name} · {p.width_um} µm</option>)}
        </select></label>
      </div>
      <label className="library-fit"><input type="checkbox" checked={fit} onChange={e => setFit(e.target.checked)}/>Enlarge domain to fit aligned components</label>
      <button className="button secondary compact" onClick={() => void align()}>{busy ? 'Aligning…' : 'Align ports'}</button>
    </fieldset>
    {feedback && <p role="status">{feedback}</p>}
  </details>;
}
