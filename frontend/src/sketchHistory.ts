import type { Vec2 } from './geometryModel';

type PointSelection = { points: Vec2[]; selected: number };
type PointHistory = { past: PointSelection[]; present: PointSelection; future: PointSelection[] };
type Action = { type: 'write' | 'reset'; points: Vec2[]; selected: number }
  | { type: 'select'; selected: number }
  | { type: 'undo' | 'redo' };

export function initialPointHistory(points: Vec2[] = [], selected = -1): PointHistory {
  return { past: [], present: { points, selected }, future: [] };
}

/** Only completed point edits enter history. Pointer previews and selection changes are transient. */
export function pointHistory(state: PointHistory, action: Action): PointHistory {
  const { past, present, future } = state;
  switch (action.type) {
    case 'reset':
      return initialPointHistory(action.points, action.selected);
    case 'select':
      return present.selected === action.selected ? state : { ...state, present: { ...present, selected: action.selected } };
    case 'write': {
      if (action.points.length === present.points.length && action.points.every((p, i) =>
        p.x === present.points[i].x && p.y === present.points[i].y)) return state;
      return {
        past: [...past.slice(-49), present],
        present: { points: action.points, selected: action.points.length ? action.selected : -1 },
        future: [],
      };
    }
    case 'undo':
      return past.length ? { past: past.slice(0, -1), present: past[past.length - 1], future: [present, ...future] } : state;
    case 'redo':
      return future.length ? { past: [...past, present], present: future[0], future: future.slice(1) } : state;
  }
}
