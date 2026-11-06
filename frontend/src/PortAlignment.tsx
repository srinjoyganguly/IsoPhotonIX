import { useRef, useState } from 'react';
import { request } from './api';
import {
  connectedPorts, portRelation, validateGeometryDocument,
  type CircuitConnection, type ComponentInstance, type GeometryDocument, type LayoutPort, type PortEndpoint,
} from './geometryModel';

type Props = {
  scene: GeometryDocument;
  selected?: ComponentInstance;
  selectedId: string | null;
  names: Record<string, string>;
  onApply: (geometry: GeometryDocument, message: string, selectedId: string | null, fit: boolean) => boolean;
};

function quantity(n: number) {
  if (n === 0) return '0';
  return Math.abs(n) < .001 ? n.toExponential(2) : Number(n.toPrecision(5)).toString();
}

function matches(end: PortEndpoint, component: ComponentInstance, port: LayoutPort) {
  return end.component_id === component.id && end.port === port.name;
}

export function PortAlignment({ scene, selected, selectedId, names, onApply }: Props) {
  const [sourceId, setSourceId] = useState('');
  const [targetId, setTargetId] = useState('');
  const [sourcePort, setSourcePort] = useState('');
  const [targetPort, setTargetPort] = useState('');
  const [fit, setFit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ text: string; error: boolean } | null>(null);
  const latest = useRef(scene);
  const pending = useRef(false);
  latest.current = scene;
  const parts = scene.components?.filter(c => c.ports.length) ?? [];
  const source = parts.find(c => c.id === sourceId) ?? parts.find(c => c.id === selected?.id) ?? parts.at(-1);
  const targets = parts.filter(c => c.id !== source?.id);
  const target = targets.find(c => c.id === targetId) ?? targets[0];
  const from = source?.ports.find(p => p.name === sourcePort) ?? source?.ports[0];
  const to = target?.ports.find(p => p.name === targetPort) ?? target?.ports.find(p => p.name === 'out') ?? target?.ports[0];
  const rows = connectedPorts(scene);
  const describe = (c: ComponentInstance) => `${names[c.family] || c.family} · ${c.id.length > 16 ? c.id.slice(-8) : c.id}`;
  const owner = (part: ComponentInstance, port: LayoutPort) =>
    scene.connections?.find(edge => matches(edge.a, part, port) || matches(edge.b, part, port));

  async function edit(url: string, payload: Record<string, unknown>, message: string, selection: string | null, refit = false) {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setFeedback(null);
    try {
      const { geometry } = await request<{ geometry: GeometryDocument }>(url, { geometry: scene, ...payload });
      if (latest.current !== scene) throw new Error('The layout changed during this edit. Select the ports and try again.');
      if (onApply(validateGeometryDocument(geometry), message, selection, refit)) setFeedback({ text: message, error: false });
    } catch (reason) {
      setFeedback({ text: (reason as Error).message, error: true });
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }

  function join(moving: ComponentInstance, start: LayoutPort, fixed: ComponentInstance, end: LayoutPort, action: 'snap' | 'plan' | 'align') {
    const moves = action !== 'plan';
    const payload = { moving_component: moving.id, moving_port: start.name, fixed_component: fixed.id,
      fixed_port: end.name, fit_domain: moves && fit };
    const pair = `${describe(moving)} (${start.name}) to ${describe(fixed)} (${end.name})`;
    if (action === 'align') return edit('/api/library/align', payload, `Aligned ${pair}.`, moving.object_ids[0], fit);
    return edit('/api/library/connect', { ...payload, align: moves },
      moves ? `Connected and aligned ${pair}.` : `Saved connection from ${pair}.`, moving.object_ids[0], moves && fit);
  }

  function disconnect(edge: CircuitConnection) {
    return edit('/api/library/disconnect', { connection_id: edge.id }, 'Connection removed.', selectedId);
  }

  if (scene.schema_version !== 2) return null;
  const pair = source && target && from && to ? { source, target, from, to } : null;
  const sourceOwner = pair && owner(pair.source, pair.from), targetOwner = pair && owner(pair.target, pair.to);
  const existing = sourceOwner && sourceOwner.id === targetOwner?.id;
  const occupied = !!(sourceOwner || targetOwner) && !existing;
  const sameWidth = pair && portRelation(pair.from, pair.to).widthsMatch;

  return <details className="port-alignment" onChange={() => setFeedback(null)}>
    <summary>Component connections{rows.length > 0 && <span className="connection-count">{rows.length}</span>}</summary>
    {pair ? <fieldset disabled={busy}>
      <div className="port-pair">
        <label>Moving component<select aria-label="Moving component" value={pair.source.id} onChange={e => { setSourceId(e.target.value); setSourcePort(''); }}>
          {parts.map(c => <option key={c.id} value={c.id}>{describe(c)}</option>)}
        </select></label>
        <label>Moving port<select aria-label="Moving port" value={pair.from.name} onChange={e => setSourcePort(e.target.value)}>
          {pair.source.ports.map(p => <option key={p.name} value={p.name}>{p.name} · {p.width_um} µm{owner(pair.source, p) ? ' · connected' : ''}</option>)}
        </select></label>
        <label>Fixed component<select aria-label="Fixed component" value={pair.target.id} onChange={e => { setTargetId(e.target.value); setTargetPort(''); }}>
          {targets.map(c => <option key={c.id} value={c.id}>{describe(c)}</option>)}
        </select></label>
        <label>Fixed port<select aria-label="Fixed port" value={pair.to.name} onChange={e => setTargetPort(e.target.value)}>
          {pair.target.ports.map(p => <option key={p.name} value={p.name}>{p.name} · {p.width_um} µm{owner(pair.target, p) ? ' · connected' : ''}</option>)}
        </select></label>
      </div>
      <label className="library-fit"><input type="checkbox" checked={fit} onChange={e => setFit(e.target.checked)}/>Enlarge domain to fit aligned components</label>
      <div className="connection-actions">
        <button type="button" className="button primary compact" disabled={occupied || !sameWidth}
          onClick={() => void join(pair.source, pair.from, pair.target, pair.to, 'snap')}>Connect and align</button>
        <button type="button" className="button secondary compact" disabled={occupied}
          onClick={() => void join(pair.source, pair.from, pair.target, pair.to, 'plan')}>Connect without moving</button>
        <button type="button" className="button secondary compact" title="Align the ports without saving a new connection" disabled={occupied || !sameWidth}
          onClick={() => void join(pair.source, pair.from, pair.target, pair.to, 'align')}>Align ports</button>
      </div>
      {occupied && <p className="connection-hint">One of these ports already has a different connection. Disconnect it first.</p>}
      {!sameWidth && <p className="connection-hint">These ports have different widths. You can save a planned connection and add a taper later.</p>}
    </fieldset> : <p>Place two library components with named ports to connect them.</p>}
    {rows.length > 0 && <ul className="connection-list" aria-label="Saved connections">
      {rows.map(({ edge, first, second, a, b, relation }) => <li key={edge.id} data-connection-id={edge.id}>
        <div className="connection-head"><span className={'connection-state ' + (relation.aligned ? 'aligned' : 'planned')}>{relation.aligned ? 'Aligned' : 'Needs route'}</span>
          <strong>{describe(first)} / {a.name} ↔ {describe(second)} / {b.name}</strong></div>
        <dl className="connection-metrics">
          <div><dt>Separation</dt><dd>{quantity(relation.distance)} µm</dd></div>
          <div><dt>Δx, Δy, Δz</dt><dd>{[relation.delta.x, relation.delta.y, relation.delta.z].map(quantity).join(', ')} µm</dd></div>
          <div><dt>Facing error</dt><dd>{quantity(relation.angleError)}°</dd></div>
          <div><dt>Width difference</dt><dd>{quantity(relation.widthDifference)} µm</dd></div>
        </dl>
        <div className="connection-actions">
          {!relation.aligned && relation.widthsMatch && <button type="button" className="button secondary compact" disabled={busy}
            onClick={() => void join(first, a, second, b, 'snap')}>Align connection</button>}
          <button type="button" className="button secondary compact" disabled={busy} onClick={() => void disconnect(edge)}>Disconnect</button>
        </div>
      </li>)}
    </ul>}
    {feedback && <p role={feedback.error ? 'alert' : 'status'}>{feedback.text}</p>}
  </details>;
}
