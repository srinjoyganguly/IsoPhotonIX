/**
 * Local geometry documents, independent of the solver configuration.
 * Coordinates are physical micrometres: x propagation, y lateral, z vertical.
 * A draft is not a supported simulation; a slab preview is only a visual extrusion.
 */
export type Vec3 = { x: number; y: number; z: number };
export type GeometryKind = 'box' | 'cylinder' | 'ring' | 'polygon';
export type Vec2 = { x: number; y: number };
export type MaterialRecord = { id: string; name: string; reference_wavelength_um: number; source: string; notes: string } & (
  { model: 'constant-nk'; n: number; k: number } |
  { model: 'tabulated-nk'; wavelength_um: number[]; n: number[]; k: number[] }
);
export type LayoutPort = { name: string; x: number; y: number; z: number; angle_deg: number; width_um: number };
export type PortEndpoint = { component_id: string; port: string };
export type CircuitConnection = { id: string; a: PortEndpoint; b: PortEndpoint };
export type ComponentInstance = {
  id: string; family: string; generator_version: 1; parameters: Record<string, number>; object_ids: string[];
  placement: { x: number; y: number; z: number; rotation_deg: number }; material_id: string;
  ports: LayoutPort[];
};
export type GeometryTemplate = 'strip' | 'coupler' | 'ring';
export type GeometryObject = {
  id: string;
  name: string;
  visible: boolean;
  kind: GeometryKind;
  position: Vec3;
  /** Box dimensions. For cylinders and rings only z (height) is used. */
  size: Vec3;
  /** Outer radius for cylinders and rings; retained but unused for boxes. */
  radius: number;
  /** Used only by rings; retained when changing shape. */
  inner_radius: number;
  /** Rotation around physical z, normalized to [0, 360) on validation. */
  rotation_deg: number;
  refractive_index: number;
  /** Polygon contours are local XY; size.z is extrusion height, size.x/y are unused. */
  vertices?: Vec2[];
  holes?: Vec2[][];
  material_id?: string;
  component_id?: string;
};
export type GeometryDocument = {
  schema: 'photonics-workbench-geometry';
  schema_version: 1 | 2;
  units: 'um';
  name: string;
  mode: 'draft-3d' | 'slab-preview';
  domain: Vec3;
  objects: GeometryObject[];
  materials?: MaterialRecord[];
  components?: ComponentInstance[];
  connections?: CircuitConnection[];
};

export const GEOMETRY_LIMITS = {
  maxObjects: 100,
  maxV2Objects: 512,
  maxJsonBytes: 1_000_000,
  minDomain: 0.1,
  maxDomain: 1_000,
  minDimension: 0.001,
  maxDimension: 1_000,
  minIndex: 1,
  maxIndex: 5,
} as const;

const documentKeys = ['schema', 'schema_version', 'units', 'name', 'mode', 'domain', 'objects'];
const objectKeys = ['id', 'name', 'visible', 'kind', 'position', 'size', 'radius', 'inner_radius', 'rotation_deg', 'refractive_index'];

function record(value: unknown, label: string, keys: string[], optional: string[] = []): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  const object = value as Record<string, unknown>;
  const unexpected = Object.keys(object).find(key => !keys.includes(key) && !optional.includes(key));
  if (unexpected) throw new Error(`${label} contains an unsupported field: ${unexpected}.`);
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(object, key)) throw new Error(`${label}.${key} is required.`);
  }
  return object;
}

function number(value: unknown, label: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${label} must be a finite number from ${min} to ${max}.`);
  }
  return Object.is(value, -0) ? 0 : value;
}

function name(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.trim().length > 80 || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new Error(`${label} must contain 1–80 characters without control characters.`);
  }
  return value.trim();
}

function vector(value: unknown, label: string, min: number, max: number): Vec3 {
  const object = record(value, label, ['x', 'y', 'z']);
  return {
    x: number(object.x, `${label}.x`, min, max),
    y: number(object.y, `${label}.y`, min, max),
    z: number(object.z, `${label}.z`, min, max),
  };
}

const EPS = 1e-9;
const identifier = (value: unknown, label: string): string => {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(value)) throw new Error(`${label} must use the geometry identifier format.`);
  return value;
};
function text(value: unknown, max: number, label: string): string {
  if (typeof value !== 'string' || value.length > max || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)) throw new Error(`${label} must be text of at most ${max} characters.`);
  return value;
}
function positive(value: unknown, max: number, label: string): number {
  const result = number(value, label, 0, max);
  if (result <= 0) throw new Error(`${label} must be positive.`);
  return result;
}
const normalizedAngle = (value: unknown, label: string) => ((number(value, label, -360, 360) % 360) + 360) % 360;

export function validateMaterial(value: unknown): MaterialRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Material must be a record.');
  const model = (value as Record<string, unknown>).model;
  if (model !== 'constant-nk' && model !== 'tabulated-nk') throw new Error('Material model must be constant-nk or tabulated-nk. Gain is not implemented.');
  const item = record(value, 'Material', ['id', 'name', 'model', 'reference_wavelength_um', 'source', 'notes', 'n', 'k', ...(model === 'tabulated-nk' ? ['wavelength_um'] : [])]);
  const common = { id: identifier(item.id, 'Material id'), name: name(item.name, 'Material name'), reference_wavelength_um: positive(item.reference_wavelength_um, Number.MAX_VALUE, 'Reference wavelength'), source: text(item.source, 1024, 'Material source'), notes: text(item.notes, 2000, 'Material notes') };
  if (model === 'constant-nk') return { ...common, model, n: positive(item.n, 20, 'Refractive index n'), k: number(item.k, 'Extinction coefficient k', 0, 20) };
  if (!Array.isArray(item.wavelength_um) || !Array.isArray(item.n) || !Array.isArray(item.k) || item.wavelength_um.length < 2 || item.wavelength_um.length > 2048 || item.n.length !== item.wavelength_um.length || item.k.length !== item.wavelength_um.length) throw new Error('Tabulated n/k requires 2–2048 matching wavelength, n and k samples.');
  const wavelength_um = item.wavelength_um.map(w => positive(w, Number.MAX_VALUE, 'Material wavelength'));
  if (wavelength_um.some((w, i) => i > 0 && w <= wavelength_um[i - 1])) throw new Error('Material wavelengths must be strictly increasing.');
  if (common.reference_wavelength_um < wavelength_um[0] || common.reference_wavelength_um > wavelength_um.at(-1)!) throw new Error('Material reference wavelength is outside the table; extrapolation is not allowed.');
  return { ...common, model, wavelength_um, n: item.n.map(v => positive(v, 20, 'Refractive index n')), k: item.k.map(v => number(v, 'Extinction coefficient k', 0, 20)) };
}
export const validateMaterialRecord = validateMaterial;

export function evaluateMaterial(value: MaterialRecord, wavelengthUm: number): { n: number; k: number } {
  const material = validateMaterial(value);
  const wavelength = positive(wavelengthUm, Number.MAX_VALUE, 'Wavelength');
  if (material.model === 'constant-nk') return { n: material.n, k: material.k };
  const waves = material.wavelength_um;
  if (wavelength < waves[0] || wavelength > waves.at(-1)!) throw new Error('Wavelength is outside the material table; extrapolation is not allowed.');
  for (let i = 0; i < waves.length - 1; i++) if (wavelength <= waves[i + 1]) {
    const fraction = (wavelength - waves[i]) / (waves[i + 1] - waves[i]);
    return { n: material.n[i] + fraction * (material.n[i + 1] - material.n[i]), k: material.k[i] + fraction * (material.k[i + 1] - material.k[i]) };
  }
  throw new Error('Material interpolation failed.');
}

const cross = (a: Vec2, b: Vec2, c: Vec2) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
function onSegment(p: Vec2, a: Vec2, b: Vec2): boolean {
  const scale = Math.max(1, Math.hypot(b.x - a.x, b.y - a.y));
  return Math.abs(cross(a, b, p)) <= EPS * scale && p.x >= Math.min(a.x, b.x) - EPS && p.x <= Math.max(a.x, b.x) + EPS && p.y >= Math.min(a.y, b.y) - EPS && p.y <= Math.max(a.y, b.y) + EPS;
}
function intersects(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  if (Math.max(a.x, b.x) < Math.min(c.x, d.x) - EPS || Math.max(c.x, d.x) < Math.min(a.x, b.x) - EPS || Math.max(a.y, b.y) < Math.min(c.y, d.y) - EPS || Math.max(c.y, d.y) < Math.min(a.y, b.y) - EPS) return false;
  return onSegment(c, a, b) || onSegment(d, a, b) || onSegment(a, c, d) || onSegment(b, c, d) || ((cross(a, b, c) > 0) !== (cross(a, b, d) > 0) && (cross(c, d, a) > 0) !== (cross(c, d, b) > 0));
}
const edges = (points: Vec2[]) => points.map((a, i) => [a, points[(i + 1) % points.length]] as const);
function inside(point: Vec2, contour: Vec2[]): boolean {
  let result = false;
  for (const [a, b] of edges(contour)) {
    if (onSegment(point, a, b)) return false;
    if ((a.y > point.y) !== (b.y > point.y) && point.x < a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)) result = !result;
  }
  return result;
}
function contour(value: unknown, label: string, ccw: boolean): Vec2[] {
  if (!Array.isArray(value) || value.length < 3 || value.length > 256) throw new Error(`${label} must contain 3–256 XY vertices without a repeated closing point.`);
  const points = value.map(v => { const p = record(v, `${label} vertex`, ['x', 'y']); return { x: number(p.x, 'Vertex x', -1000, 1000), y: number(p.y, 'Vertex y', -1000, 1000) }; });
  if (edges(points).some(([a, b]) => Math.hypot(b.x - a.x, b.y - a.y) <= EPS)) throw new Error(`${label} has a zero-length or repeated edge.`);
  for (let i = 0; i < points.length; i++) {
    const a = points[(i + points.length - 1) % points.length], b = points[i], c = points[(i + 1) % points.length];
    if (Math.abs(cross(a, b, c)) <= EPS && (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) > EPS) throw new Error(`${label} doubles back along an adjacent edge.`);
    for (let j = i + 1; j < points.length; j++) {
      if (j === i + 1 || (i === 0 && j === points.length - 1)) continue;
      if (intersects(points[i], points[(i + 1) % points.length], points[j], points[(j + 1) % points.length])) throw new Error(`${label} self-intersects or touches itself.`);
    }
  }
  const area2 = edges(points).reduce((sum, [a, b]) => sum + a.x * b.y - b.x * a.y, 0);
  if (Math.abs(area2) <= EPS) throw new Error(`${label} must have nonzero area.`);
  if ((area2 > 0) !== ccw) points.reverse();
  return points;
}
export function validatePolygon(vertices: unknown, holes: unknown): { vertices: Vec2[]; holes: Vec2[][] } {
  const outer = contour(vertices, 'Polygon outline', true);
  if (!Array.isArray(holes) || holes.length > 32) throw new Error('Polygon holes must be an array of at most 32 contours.');
  if (outer.length + holes.reduce((sum, h) => sum + (Array.isArray(h) ? h.length : 1025), 0) > 1024) throw new Error('A polygon may contain at most 1024 total vertices.');
  const rings = holes.map((h, i) => contour(h, `Polygon hole ${i + 1}`, false));
  for (let i = 0; i < rings.length; i++) {
    const ring = rings[i];
    if (!ring.every(p => inside(p, outer)) || edges(ring).some(([a, b]) => edges(outer).some(([c, d]) => intersects(a, b, c, d)))) throw new Error('Polygon holes must lie strictly inside the outline without touching it.');
    for (const other of rings.slice(0, i)) if (inside(ring[0], other) || inside(other[0], ring) || edges(ring).some(([a, b]) => edges(other).some(([c, d]) => intersects(a, b, c, d)))) throw new Error('Polygon holes may not overlap, nest or touch.');
  }
  return { vertices: outer, holes: rings };
}

/** Actual world-space bounds, including off-centre polygon vertices. */
export function geometryBounds(object: GeometryObject): { min: Vec3; max: Vec3 } {
  let min: Vec3, max: Vec3;
  if (object.kind === 'polygon') {
    const angle = object.rotation_deg * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
    const points = object.vertices!.map(p => ({ x: p.x * c - p.y * s, y: p.x * s + p.y * c }));
    min = { x: Math.min(...points.map(p => p.x)), y: Math.min(...points.map(p => p.y)), z: -object.size.z / 2 };
    max = { x: Math.max(...points.map(p => p.x)), y: Math.max(...points.map(p => p.y)), z: object.size.z / 2 };
  } else {
    let half = { x: object.radius, y: object.radius, z: object.size.z / 2 };
    if (object.kind === 'box') {
      const angle = object.rotation_deg * Math.PI / 180, c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle));
      half = { x: (object.size.x * c + object.size.y * s) / 2, y: (object.size.x * s + object.size.y * c) / 2, z: object.size.z / 2 };
    }
    min = { x: -half.x, y: -half.y, z: -half.z }; max = half;
  }
  return { min: { x: object.position.x + min.x, y: object.position.y + min.y, z: object.position.z + min.z }, max: { x: object.position.x + max.x, y: object.position.y + max.y, z: object.position.z + max.z } };
}
export function geometryHalfExtents(object: GeometryObject): Vec3 {
  const bounds = geometryBounds(object);
  return { x: (bounds.max.x - bounds.min.x) / 2, y: (bounds.max.y - bounds.min.y) / 2, z: (bounds.max.z - bounds.min.z) / 2 };
}

function validateComponents(value: unknown, objects: GeometryObject[], materials: Map<string, MaterialRecord>, domain: Vec3): ComponentInstance[] {
  if (!Array.isArray(value) || value.length > 512) throw new Error('A geometry draft may contain at most 512 component instances.');
  const byId = new Map(objects.map(o => [o.id, o])), ids = new Set<string>(), owned = new Set<string>();
  const result = value.map(raw => {
    const item = record(raw, 'Component', ['id', 'family', 'generator_version', 'parameters', 'object_ids', 'placement', 'material_id', 'ports']);
    const id = identifier(item.id, 'Component id'); if (ids.has(id)) throw new Error('Component ids must be unique.'); ids.add(id);
    if (item.generator_version !== 1) throw new Error('Unsupported component generator version.');
    if (!item.parameters || typeof item.parameters !== 'object' || Array.isArray(item.parameters) || Object.keys(item.parameters).length > 64) throw new Error('Component parameters must contain at most 64 named finite numbers.');
    const parameters = Object.fromEntries(Object.entries(item.parameters).map(([k, v]) => [identifier(k, 'Parameter key'), number(v, 'Component parameter', -Number.MAX_VALUE, Number.MAX_VALUE)]));
    if (!Array.isArray(item.object_ids) || item.object_ids.length < 1 || item.object_ids.length > 512 || !item.object_ids.every(m => typeof m === 'string') || new Set(item.object_ids).size !== item.object_ids.length) throw new Error('Component object ids must be a nonempty unique array.');
    const object_ids = item.object_ids as string[];
    if (object_ids.some(m => !byId.has(m) || owned.has(m) || byId.get(m)!.component_id !== id)) throw new Error('Component membership does not match the actual scene objects.');
    object_ids.forEach(m => owned.add(m));
    const material_id = identifier(item.material_id, 'Component material'); if (!materials.has(material_id)) throw new Error('Component material is missing.');
    const rawPlacement = record(item.placement, 'Component placement', ['x', 'y', 'z', 'rotation_deg']);
    const placement = { x: number(rawPlacement.x, 'Placement x', -500, 500), y: number(rawPlacement.y, 'Placement y', -500, 500), z: number(rawPlacement.z, 'Placement z', -500, 500), rotation_deg: normalizedAngle(rawPlacement.rotation_deg, 'Placement rotation') };
    if (!Array.isArray(item.ports) || item.ports.length > 64) throw new Error('A component may contain at most 64 named layout ports.');
    const portNames = new Set<string>();
    const ports = item.ports.map(rawPort => {
      const port = record(rawPort, 'Layout port', ['name', 'x', 'y', 'z', 'angle_deg', 'width_um']);
      const portName = name(port.name, 'Port name'); if (portNames.has(portName)) throw new Error('Layout port names must be unique within a component.'); portNames.add(portName);
      const normalized = { name: portName, x: number(port.x, 'Port x', -500, 500), y: number(port.y, 'Port y', -500, 500), z: number(port.z, 'Port z', -500, 500), angle_deg: normalizedAngle(port.angle_deg, 'Port direction'), width_um: positive(port.width_um, 1000, 'Port width') };
      if ((['x', 'y', 'z'] as const).some(a => Math.abs(normalized[a]) > domain[a] / 2 + EPS)) throw new Error('A layout port extends outside the domain.');
      return normalized;
    });
    return { id, family: identifier(item.family, 'Component family'), generator_version: 1 as const, parameters, object_ids: [...object_ids], placement, material_id, ports };
  });
  if (objects.some(o => o.component_id !== undefined && (!ids.has(o.component_id) || !owned.has(o.id))) || owned.size !== objects.filter(o => o.component_id !== undefined).length) throw new Error('An object has a missing or inconsistent component instance.');
  return result;
}

function validateConnections(value: unknown, components: ComponentInstance[]): CircuitConnection[] {
  if (!Array.isArray(value) || value.length > 16384) throw new Error('A design may contain at most 16,384 named-port connections.');
  const ports = new Map(components.map(c => [c.id, new Set(c.ports.map(p => p.name))]));
  const ids = new Set<string>(), occupied = new Set<string>();
  function endpoint(raw: unknown): PortEndpoint {
    const recordValue = record(raw, 'Connection endpoint', ['component_id', 'port']);
    const component_id = identifier(recordValue.component_id, 'Component id');
    const port = name(recordValue.port, 'Port name');
    if (!ports.get(component_id)?.has(port)) throw new Error('A connection references a missing component or port.');
    const key = JSON.stringify([component_id, port]);
    if (occupied.has(key)) throw new Error('Each named port may belong to only one saved connection.');
    occupied.add(key);
    return { component_id, port };
  }
  return value.map(raw => {
    const item = record(raw, 'Connection', ['id', 'a', 'b']);
    const id = identifier(item.id, 'Connection id');
    if (ids.has(id)) throw new Error('Connection ids must be unique.');
    ids.add(id);
    const a = endpoint(item.a), b = endpoint(item.b);
    if (a.component_id === b.component_id) throw new Error('Choose ports on two different components.');
    return { id, a, b };
  });
}

export function portRelation(a: LayoutPort, b: LayoutPort) {
  const delta = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const distance = Math.hypot(delta.x, delta.y, delta.z);
  const angleError = Math.abs(((a.angle_deg - b.angle_deg) % 360 + 360) % 360 - 180);
  const widthDifference = Math.abs(a.width_um - b.width_um);
  const widthsMatch = widthDifference <= Math.max(1e-9, 1e-9 * Math.max(a.width_um, b.width_um));
  return { delta, distance, angleError, widthDifference, widthsMatch, aligned: distance <= 1e-8 && angleError <= 1e-7 && widthsMatch };
}

export function connectedPorts(document: GeometryDocument) {
  const parts = new Map((document.components ?? []).map(part => [part.id, part]));
  return (document.connections ?? []).map(edge => {
    // The document validator enforces both named endpoints before rendering.
    const first = parts.get(edge.a.component_id)!, second = parts.get(edge.b.component_id)!;
    const a = first.ports.find(port => port.name === edge.a.port)!, b = second.ports.find(port => port.name === edge.b.port)!;
    return { edge, first, second, a, b, relation: portRelation(a, b) };
  });
}

export function removeComponentLinks(document: GeometryDocument, id: string): Pick<GeometryDocument, 'connections'> {
  return document.connections === undefined ? {} : {
    connections: document.connections.filter(edge => edge.a.component_id !== id && edge.b.component_id !== id),
  };
}

/** Validate a fresh normalized document; v1 stays the original primitive format. */
export function validateGeometryDocument(value: unknown): GeometryDocument {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Geometry document must be an object.');
  const version = (value as Record<string, unknown>).schema_version;
  if (version !== 1 && version !== 2) throw new Error('Expected a photonics-workbench-geometry document with schema_version 1 or 2. Solver configuration JSON is a different format.');
  const input = record(value, 'Geometry document', [...documentKeys, ...(version === 2 ? ['materials', 'components'] : [])], version === 2 ? ['connections'] : []);
  if (input.schema !== 'photonics-workbench-geometry') throw new Error('Expected a photonics-workbench-geometry document. Solver configuration JSON is a different format.');
  if (input.units !== 'um') throw new Error('Geometry units must be um (micrometres).');
  if (input.mode !== 'draft-3d' && input.mode !== 'slab-preview') throw new Error('Geometry mode must be draft-3d or slab-preview.');
  const domain = vector(input.domain, 'Domain', GEOMETRY_LIMITS.minDomain, GEOMETRY_LIMITS.maxDomain);
  let materials: MaterialRecord[] = [];
  if (version === 2) {
    if (!Array.isArray(input.materials) || input.materials.length > 512) throw new Error('A geometry draft may contain at most 512 materials.');
    materials = input.materials.map(validateMaterial);
    if (new Set(materials.map(m => m.id)).size !== materials.length) throw new Error('Material ids must be unique.');
  }
  const materialById = new Map(materials.map(m => [m.id, m]));
  const limit = version === 1 ? GEOMETRY_LIMITS.maxObjects : GEOMETRY_LIMITS.maxV2Objects;
  if (!Array.isArray(input.objects) || input.objects.length > limit) throw new Error(`Geometry must contain an objects array with at most ${limit} objects.`);
  const ids = new Set<string>();
  const objects = input.objects.map((raw, index): GeometryObject => {
    const label = `Object ${index + 1}`, polygon = raw && typeof raw === 'object' && raw.kind === 'polygon';
    const item = record(raw, label, [...objectKeys, ...(polygon && version === 2 ? ['vertices', 'holes'] : [])], version === 2 ? ['material_id', 'component_id'] : []);
    const id = identifier(item.id, `${label}.id`); if (ids.has(id)) throw new Error(`${label} has a duplicate id: ${id}.`); ids.add(id);
    if (![...['box', 'cylinder', 'ring'], ...(version === 2 ? ['polygon'] : [])].includes(String(item.kind))) throw new Error(`${label}.kind is unsupported for this geometry version.`);
    if (typeof item.visible !== 'boolean') throw new Error(`${label}.visible must be true or false.`);
    const object: GeometryObject = { id, name: name(item.name, `${label}.name`), visible: item.visible, kind: item.kind as GeometryKind,
      position: vector(item.position, `${label}.position`, -500, 500), size: vector(item.size, `${label}.size`, .001, 1000), radius: number(item.radius, `${label}.radius`, .001, 500), inner_radius: number(item.inner_radius, `${label}.inner_radius`, 0, 500), rotation_deg: normalizedAngle(item.rotation_deg, `${label}.rotation_deg`), refractive_index: number(item.refractive_index, `${label}.refractive_index`, version === 1 ? 1 : item.material_id !== undefined ? Number.MIN_VALUE : .01, version === 1 ? 5 : 20) };
    if (object.kind === 'ring' && (object.inner_radius < .001 || object.radius - object.inner_radius < .001 - 1e-12)) throw new Error(`${object.name}: ring inner radius and wall thickness must each be at least .001 µm.`);
    if (polygon) Object.assign(object, validatePolygon(item.vertices, item.holes));
    if (item.material_id !== undefined) {
      const id = identifier(item.material_id, 'Material reference'), material = materialById.get(id);
      if (!material) throw new Error('An object refers to a missing material.');
      const n = evaluateMaterial(material, material.reference_wavelength_um).n;
      if (Math.abs(object.refractive_index - n) > Math.max(1e-12, 1e-9 * Math.max(Math.abs(object.refractive_index), Math.abs(n)))) throw new Error('Object preview index differs from its material at the reference wavelength.');
      object.material_id = id;
    }
    if (item.component_id !== undefined) object.component_id = identifier(item.component_id, 'Component reference');
    const bounds = geometryBounds(object);
    for (const axis of ['x', 'y', 'z'] as const) if (bounds.min[axis] < -domain[axis] / 2 - EPS || bounds.max[axis] > domain[axis] / 2 + EPS) throw new Error(`${object.name} extends outside the domain along ${axis}. Move or resize the object, or enlarge the domain.`);
    return object;
  });
  const components = version === 2 ? validateComponents(input.components, objects, materialById, domain) : [];
  return { schema: 'photonics-workbench-geometry', schema_version: version, units: 'um', name: name(input.name, 'Document name'), mode: input.mode, domain, objects,
    ...(version === 2 ? { materials, components } : {}),
    ...(Object.hasOwn(input, 'connections') ? { connections: validateConnections(input.connections, components) } : {}) };
}

export function upgradeToV2(document: GeometryDocument): GeometryDocument {
  const value = validateGeometryDocument(document);
  return value.schema_version === 2 ? value : { ...value, schema_version: 2, materials: [], components: [] };
}

/** Manual object edits detach the whole recipe so component metadata never lies. */
export function detachComponent(document: GeometryDocument, objectId: string): GeometryDocument {
  const value = validateGeometryDocument(document);
  const componentId = value.objects.find(o => o.id === objectId)?.component_id;
  if (!componentId) return value;
  return { ...value, ...removeComponentLinks(value, componentId), components: value.components!.filter(c => c.id !== componentId), objects: value.objects.map(o => {
    if (o.component_id !== componentId) return o;
    const { component_id: _removed, ...detached } = o; return detached;
  }) };
}

export function parseGeometryDocument(text: string): GeometryDocument {
  if (new TextEncoder().encode(text).byteLength > GEOMETRY_LIMITS.maxJsonBytes) throw new Error('Geometry JSON must be smaller than 1 MB.');
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error('Geometry file is not valid JSON.'); }
  return validateGeometryDocument(value);
}

export function serializeGeometryDocument(document: GeometryDocument): string {
  return `${JSON.stringify(validateGeometryDocument(document), null, 2)}\n`;
}

let nextId = 0;
function uniqueId(): string {
  return typeof globalThis.crypto?.randomUUID === 'function'
    ? `object-${globalThis.crypto.randomUUID()}`
    : `object-${Date.now().toString(36)}-${(++nextId).toString(36)}`;
}

/** Creates an object centered at the origin. Validate against the target domain after adding. */
export function createGeometryObject(kind: GeometryKind, id = uniqueId()): GeometryObject {
  return {
    id, name: kind === 'box' ? 'Dielectric box' : kind === 'cylinder' ? 'Dielectric cylinder' : kind === 'ring' ? 'Dielectric ring' : 'Extruded polygon',
    visible: true, kind, position: { x: 0, y: 0, z: 0 }, size: { x: 1, y: 0.5, z: 0.22 },
    radius: 1, inner_radius: 0.6, rotation_deg: 0, refractive_index: 3.45,
    ...(kind === 'polygon' ? { vertices: [{x:-.5,y:-.25},{x:.5,y:-.25},{x:.5,y:.25},{x:-.5,y:.25}], holes: [] } : {}),
  };
}

function draft(name: string, objects: GeometryObject[]): GeometryDocument {
  return validateGeometryDocument({
    schema: 'photonics-workbench-geometry', schema_version: 1, units: 'um', mode: 'draft-3d',
    name, domain: { x: 12, y: 8, z: 2.5 }, objects,
  });
}

export function createGeometryTemplate(template: GeometryTemplate): GeometryDocument {
  const substrate: GeometryObject = {
    ...createGeometryObject('box', 'substrate'), name: 'Substrate',
    position: { x: 0, y: 0, z: -0.56 }, size: { x: 12, y: 8, z: 0.9 }, refractive_index: 1.44,
  };
  const strip: GeometryObject = {
    ...createGeometryObject('box', 'core'), name: 'Strip core', size: { x: 10, y: 0.5, z: 0.22 },
  };
  if (template === 'strip') return draft('Strip waveguide', [substrate, strip]);
  if (template === 'coupler') return draft('Directional coupler', [
    substrate,
    { ...strip, id: 'core-a', name: 'Waveguide A', position: { x: 0, y: -0.4, z: 0 } },
    { ...strip, id: 'core-b', name: 'Waveguide B', position: { x: 0, y: 0.4, z: 0 } },
  ]);
  if (template === 'ring') return draft('Ring and bus waveguide', [
    substrate,
    { ...createGeometryObject('ring', 'ring'), name: 'Ring core', radius: 1.6, inner_radius: 1.1 },
    { ...strip, id: 'bus', name: 'Bus waveguide', position: { x: 0, y: -2.05, z: 0 } },
  ]);
  throw new Error('Unknown geometry template.');
}

/** A visual extrusion only: the existing solver model is uniform along z. */
export function createSlabPreview(config: { length_um: number; height_um: number; core_width_um: number; core_index: number }): GeometryDocument {
  return validateGeometryDocument({
    schema: 'photonics-workbench-geometry', schema_version: 1, units: 'um', mode: 'slab-preview',
    name: '2D slab · illustrative extrusion',
    domain: { x: config.length_um, y: config.height_um, z: 1.2 },
    objects: [{
      ...createGeometryObject('box', 'slab-core'), name: 'Slab core · uniform along z',
      size: { x: config.length_um, y: config.core_width_um, z: 0.22 }, refractive_index: config.core_index,
    }],
  });
}
