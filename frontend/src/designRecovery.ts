import { serializeGeometryDocument, validateGeometryDocument, type GeometryDocument } from './geometryModel.ts';
import { initialHistory, type DesignHistory } from './designHistory.ts';

export const HISTORY_KEY = 'photonics-workbench.edit-history.v1';
// Web Storage holds UTF-16 strings. Keep this tab's checkpoint below 2 MiB.
export const HISTORY_BYTES = 2 * 1024 * 1024;

export function packHistory(history: DesignHistory, byteLimit = HISTORY_BYTES) {
  const present = JSON.stringify(history.present);
  const head = '{"version":1,"present":' + present + ',"past":[';
  const middle = '],"future":[';
  const end = ']}';
  let available = Math.floor(byteLimit / 2) - head.length - middle.length - end.length;
  if (available < 0) return null;
  const past: string[] = [], future: string[] = [];
  let pastOpen = true, futureOpen = true;
  for (let i = 0; pastOpen || futureOpen; i++) {
    for (const side of ['past', 'future'] as const) {
      if (side === 'past' ? !pastOpen : !futureOpen) continue;
      const item = side === 'past' ? history.past.at(-1 - i) : history.future[i];
      const entries = side === 'past' ? past : future;
      const encoded = item && JSON.stringify(item);
      if (!encoded || encoded.length + Number(entries.length > 0) > available) {
        if (side === 'past') pastOpen = false;
        else futureOpen = false;
        continue;
      }
      available -= encoded.length + Number(entries.length > 0);
      entries.push(encoded);
    }
  }
  return {
    text: head + past.reverse().join(',') + middle + future.join(',') + end,
    trimmed: past.length !== history.past.length || future.length !== history.future.length,
  };
}

export function unpackHistory(text: string | null, savedDraft: string, draft: GeometryDocument): DesignHistory {
  const fresh = initialHistory(draft);
  if (!text || text.length * 2 > HISTORY_BYTES) return fresh;
  try {
    const stored = JSON.parse(text);
    if (stored.version !== 1 || !Array.isArray(stored.past) || !Array.isArray(stored.future) || stored.past.length + stored.future.length > 50) return fresh;
    function snapshot(item: { document: unknown; selectedId: unknown }) {
      const document = validateGeometryDocument(item.document);
      const id = item.selectedId;
      if (id !== null && (typeof id !== 'string' || !document.objects.some(object => object.id === id))) throw new Error('Invalid saved selection.');
      return { document, selectedId: id };
    }
    const present = snapshot(stored.present);
    if (serializeGeometryDocument(present.document) !== savedDraft) return fresh;
    return {
      past: stored.past.map(snapshot), present: { document: draft, selectedId: present.selectedId }, future: stored.future.map(snapshot),
    };
  } catch {
    return fresh;
  }
}
