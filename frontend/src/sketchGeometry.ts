import {
  createGeometryObject, detachComponent, upgradeToV2, validateGeometryDocument, validatePolygon,
  type GeometryDocument, type GeometryObject, type Vec2,
} from './geometryModel.ts';

const MIN_SEGMENT = 1e-9;
export const MAX_ROUTE_POINTS = 128;

export function snapPoint(point: Vec2, spacing: number | null): Vec2 {
  if (spacing === null) return point;
  if (!Number.isFinite(spacing) || spacing <= 0) throw new Error('Snap spacing must be a positive number.');
  const snap = (value: number) => Number((Math.round(value / spacing) * spacing).toFixed(12));
  return { x: snap(point.x), y: snap(point.y) };
}

/** Constant full width, perpendicular end faces and intersecting offset lines at each bend. */
export function routeOutline(points: Vec2[], width: number, radius = 0): Vec2[] {
  if (points.length < 2 || points.length > MAX_ROUTE_POINTS) throw new Error('A route needs 2–128 points.');
  if (!Number.isFinite(width) || width < .001 || width > 1000) throw new Error('Route width must be 0.001–1000 µm.');
  const normals: Vec2[] = [];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x, dy = points[i].y - points[i - 1].y;
    const length = Math.hypot(dx, dy);
    if (!Number.isFinite(length) || length <= MIN_SEGMENT) throw new Error('Separate consecutive route points.');
    normals.push({ x: -dy / length, y: dx / length });
  }
  if (!Number.isFinite(radius) || radius < 0) throw new Error('Enter a nonnegative centreline bend radius.');
  if (radius > 0) return roundedOutline(points, normals, width, radius);
  const left: Vec2[] = [], right: Vec2[] = [];
  for (let i = 0; i < points.length; i++) {
    const before = normals[Math.max(0, i - 1)], after = normals[Math.min(i, normals.length - 1)];
    const denominator = 1 + before.x * after.x + before.y * after.y;
    // A miter longer than four half-widths makes a very acute spike.
    if (denominator < 1 / 8) throw new Error('This turn is too sharp. Move the corner or add a more gradual bend.');
    const offsetX = width / 2 * (before.x + after.x) / denominator;
    const offsetY = width / 2 * (before.y + after.y) / denominator;
    left.push({ x: points[i].x + offsetX, y: points[i].y + offsetY });
    right.push({ x: points[i].x - offsetX, y: points[i].y - offsetY });
  }
  for (let i = 0; i < normals.length; i++) {
    const n = normals[i];
    for (const side of [left, right]) {
      const length = (side[i + 1].x - side[i].x) * n.y - (side[i + 1].y - side[i].y) * n.x;
      if (length <= MIN_SEGMENT) throw new Error('The width consumes a segment at this bend. Narrow the route or move the corner.');
    }
  }
  return validatePolygon([...left, ...right.reverse()], []).vertices;
}

/** Tangent circular fillets. Both material boundaries are sampled directly at
 * their inner/outer radii. The maximum angular step is five degrees. */
function roundedOutline(points: Vec2[], normals: Vec2[], width: number, radius: number): Vec2[] {
  const half = width / 2;
  if (radius <= half) throw new Error('The bend radius must exceed half the route width.');
  const trims = Array<number>(points.length).fill(0);
  const turns = Array<number>(points.length).fill(0);
  for (let i = 1; i < points.length - 1; i++) {
    const a = normals[i-1], b = normals[i];
    const turn = Math.atan2(a.x*b.y-a.y*b.x, a.x*b.x+a.y*b.y);
    turns[i] = Math.abs(turn) < 1e-10 ? 0 : turn;
    trims[i] = radius * Math.tan(Math.abs(turns[i]) / 2);
  }
  for (let i = 0; i < normals.length; i++) {
    const length = Math.hypot(points[i+1].x - points[i].x, points[i+1].y - points[i].y);
    if (trims[i] + trims[i+1] >= length - MIN_SEGMENT) throw new Error('The circular bends consume a straight segment. Reduce the bend radius or move the corner.');
  }
  const left: Vec2[] = [], right: Vec2[] = [];
  const append = (p: Vec2, n: Vec2) => {
    left.push({x:p.x + half*n.x, y:p.y + half*n.y});
    right.push({x:p.x - half*n.x, y:p.y - half*n.y});
  };
  append(points[0], normals[0]);
  for (let i = 1; i < points.length - 1; i++) {
    const turn = turns[i], n = normals[i-1];
    if (!turn) {append(points[i], n); continue;}
    const signedRadius = Math.sign(turn) * radius;
    const centre = {x:points[i].x - trims[i]*n.y + signedRadius*n.x,
      y:points[i].y + trims[i]*n.x + signedRadius*n.y};
    const steps = Math.ceil(Math.abs(turn) / (Math.PI / 36));
    for (let j = 0; j <= steps; j++) {
      const angle = turn*j/steps;
      const normal = {x:n.x*Math.cos(angle)-n.y*Math.sin(angle), y:n.x*Math.sin(angle)+n.y*Math.cos(angle)};
      append({x:centre.x-signedRadius*normal.x, y:centre.y-signedRadius*normal.y}, normal);
    }
  }
  append(points.at(-1)!, normals.at(-1)!);
  if (left.length*2 > 256) throw new Error('This route exceeds 256 contour vertices. Reduce the number or total angle of its bends.');
  return validatePolygon([...left,...right.reverse()],[]).vertices;
}

export function worldPoint(object: GeometryObject, local: Vec2): Vec2 {
  const angle = object.rotation_deg * Math.PI / 180;
  return { x: object.position.x + Math.cos(angle) * local.x - Math.sin(angle) * local.y,
    y: object.position.y + Math.sin(angle) * local.x + Math.cos(angle) * local.y };
}

function localPoint(object: GeometryObject, world: Vec2): Vec2 {
  const angle = object.rotation_deg * Math.PI / 180;
  const x = world.x - object.position.x, y = world.y - object.position.y;
  return { x: Math.cos(angle) * x + Math.sin(angle) * y, y: -Math.sin(angle) * x + Math.cos(angle) * y };
}

export function addSketch(scene: GeometryDocument, outline: Vec2[], height: number, z: number, name: string) {
  const vertices = validatePolygon(outline, []).vertices;
  const xs = vertices.map(p => p.x), ys = vertices.map(p => p.y);
  const x = (Math.min(...xs) + Math.max(...xs)) / 2, y = (Math.min(...ys) + Math.max(...ys)) / 2;
  const solid = createGeometryObject('polygon');
  solid.name = name;
  solid.position = { x, y, z };
  solid.size = { x: Math.max(.001, Math.max(...xs) - Math.min(...xs)),
    y: Math.max(.001, Math.max(...ys) - Math.min(...ys)), z: height };
  solid.vertices = vertices.map(p => ({ x: p.x - x, y: p.y - y }));
  const base = upgradeToV2(scene);
  return { document: validateGeometryDocument({ ...base, mode: 'draft-3d', objects: [...base.objects, solid] }), id: solid.id };
}

/** Edit one boundary in world XY while preserving rotation, material and the other boundaries. */
export function editContour(scene: GeometryDocument, id: string, contour: number, points: Vec2[], height: number, z: number) {
  const base = detachComponent(scene, id);
  const object = base.objects.find(item => item.id === id);
  if (!object || object.kind !== 'polygon') throw new Error('Select a polygon in the scene.');
  const boundaries = [object.vertices!, ...object.holes!];
  if (!boundaries[contour]) throw new Error('The selected contour is missing from the design.');
  const oldPoints = boundaries[contour].map(local => ({ local, world: worldPoint(object, local) }));
  boundaries[contour] = points.map(point =>
    oldPoints.find(old => old.world.x === point.x && old.world.y === point.y)?.local ?? localPoint(object, point));
  const updated = { ...object, position: { ...object.position, z }, size: { ...object.size, z: height },
    vertices: boundaries[0], holes: boundaries.slice(1) };
  return validateGeometryDocument({ ...base, mode: 'draft-3d', objects: base.objects.map(o => o.id === id ? updated : o) });
}

/** Whole-instance translations must preserve ports and generator placement. */
export function moveSelection(scene: GeometryDocument, id: string, delta: Vec2): GeometryDocument {
  const selected = scene.objects.find(object => object.id === id);
  if (!selected) throw new Error('Select a solid to move.');
  const component = scene.components?.find(c => c.id === selected.component_id);
  const members = new Set(component?.object_ids ?? [id]);
  const shift = <T extends Vec2>(position: T): T => ({ ...position, x: position.x + delta.x, y: position.y + delta.y });
  const objects = scene.objects.map(o => members.has(o.id) ? { ...o, position: shift(o.position) } : o);
  const components = scene.components?.map(c => c === component ? { ...c, placement: shift(c.placement), ports: c.ports.map(shift) } : c);
  return validateGeometryDocument({ ...scene, mode: 'draft-3d', objects, ...(components ? { components } : {}) });
}
