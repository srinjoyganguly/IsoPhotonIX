import type { GeometryDocument } from './geometryModel';

type Snapshot = { document: GeometryDocument; selectedId: string | null };
export type DesignHistory = { past: Snapshot[]; present: Snapshot; future: Snapshot[] };

export function initialSelection(document: GeometryDocument): string | null {
  return document.objects.find(object => object.id !== 'substrate')?.id ?? document.objects[0]?.id ?? null;
}

export function initialHistory(document: GeometryDocument): DesignHistory {
  return { past: [], present: { document, selectedId: initialSelection(document) }, future: [] };
}

function snapshot(document: GeometryDocument, selectedId: string | null): Snapshot {
  const exists = selectedId !== null && document.objects.some(object => object.id === selectedId);
  return { document, selectedId: exists ? selectedId : null };
}

export function selectObject(history: DesignHistory, selectedId: string | null): DesignHistory {
  const present = snapshot(history.present.document, selectedId);
  return present.selectedId === history.present.selectedId ? history : { ...history, present };
}

export function commitDesign(history: DesignHistory, document: GeometryDocument, selectedId = history.present.selectedId): DesignHistory {
  // Canonical documents contain only validated JSON. Reapplying an unchanged edit must retain redo.
  if (JSON.stringify(document) === JSON.stringify(history.present.document)) return selectObject(history, selectedId);
  return { past: [...history.past.slice(-49), history.present], present: snapshot(document, selectedId), future: [] };
}

export function undoDesign(history: DesignHistory): DesignHistory {
  const previous = history.past.at(-1);
  return previous ? { past: history.past.slice(0, -1), present: previous, future: [history.present, ...history.future] } : history;
}

export function redoDesign(history: DesignHistory): DesignHistory {
  const next = history.future[0];
  return next ? { past: [...history.past, history.present], present: next, future: history.future.slice(1) } : history;
}

/** The document has already been validated, so each member belongs to exactly one instance. */
export function deleteSelection(document: GeometryDocument, selectedId: string | null): GeometryDocument {
  const selected = document.objects.find(object => object.id === selectedId);
  if (!selected) return document;
  const component = document.components?.find(instance => instance.id === selected.component_id);
  if (!component) {
    return { ...document, mode: 'draft-3d', objects: document.objects.filter(object => object.id !== selected.id) };
  }
  const removed = new Set(component.object_ids);
  return {
    ...document, mode: 'draft-3d',
    objects: document.objects.filter(object => !removed.has(object.id)),
    components: document.components!.filter(instance => instance.id !== component.id),
  };
}
