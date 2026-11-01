/**
 * Local geometry documents, independent of the solver configuration.
 * Coordinates are physical micrometres: x propagation, y lateral, z vertical.
 * A draft is not a supported simulation; a slab preview is only a visual extrusion.
 */
export type Vec3 = { x: number; y: number; z: number };
export type GeometryKind = 'box' | 'cylinder' | 'ring';
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
};
export type GeometryDocument = {
  schema: 'photonics-workbench-geometry';
  schema_version: 1;
  units: 'um';
  name: string;
  mode: 'draft-3d' | 'slab-preview';
  domain: Vec3;
  objects: GeometryObject[];
};

export const GEOMETRY_LIMITS = {
  maxObjects: 100,
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

function record(value: unknown, label: string, keys: string[]): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  const object = value as Record<string, unknown>;
  const unexpected = Object.keys(object).find(key => !keys.includes(key));
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

/** Axis-aligned half extents after rotation; useful for containment and framing. */
export function geometryHalfExtents(object: GeometryObject): Vec3 {
  if (object.kind !== 'box') return { x: object.radius, y: object.radius, z: object.size.z / 2 };
  const angle = object.rotation_deg * Math.PI / 180;
  const c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle));
  return {
    x: (object.size.x * c + object.size.y * s) / 2,
    y: (object.size.x * s + object.size.y * c) / 2,
    z: object.size.z / 2,
  };
}

/** Validate and return a fresh normalized document. No input values are coerced. */
export function validateGeometryDocument(value: unknown): GeometryDocument {
  const input = record(value, 'Geometry document', documentKeys);
  if (input.schema !== 'photonics-workbench-geometry' || input.schema_version !== 1) {
    throw new Error('Expected a photonics-workbench-geometry document with schema_version 1. Solver configuration JSON is a different format.');
  }
  if (input.units !== 'um') throw new Error('Geometry units must be um (micrometres).');
  if (input.mode !== 'draft-3d' && input.mode !== 'slab-preview') throw new Error('Geometry mode must be draft-3d or slab-preview.');
  const domain = vector(input.domain, 'Domain', GEOMETRY_LIMITS.minDomain, GEOMETRY_LIMITS.maxDomain);
  if (!Array.isArray(input.objects) || input.objects.length > GEOMETRY_LIMITS.maxObjects) {
    throw new Error(`Geometry must contain an objects array with at most ${GEOMETRY_LIMITS.maxObjects} objects.`);
  }
  const ids = new Set<string>();
  const objects = input.objects.map((value, index): GeometryObject => {
    const label = `Object ${index + 1}`;
    const item = record(value, label, objectKeys);
    if (typeof item.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(item.id)) {
      throw new Error(`${label}.id must contain 1–80 letters, digits, underscores or hyphens, starting with a letter or digit.`);
    }
    if (ids.has(item.id)) throw new Error(`${label} has a duplicate id: ${item.id}.`);
    ids.add(item.id);
    if (!['box', 'cylinder', 'ring'].includes(String(item.kind))) throw new Error(`${label}.kind must be box, cylinder or ring.`);
    if (typeof item.visible !== 'boolean') throw new Error(`${label}.visible must be true or false.`);
    const rawRotation = number(item.rotation_deg, `${label}.rotation_deg`, -360, 360);
    const object: GeometryObject = {
      id: item.id,
      name: name(item.name, `${label}.name`),
      visible: item.visible,
      kind: item.kind as GeometryKind,
      position: vector(item.position, `${label}.position`, -GEOMETRY_LIMITS.maxDomain / 2, GEOMETRY_LIMITS.maxDomain / 2),
      size: vector(item.size, `${label}.size`, GEOMETRY_LIMITS.minDimension, GEOMETRY_LIMITS.maxDimension),
      radius: number(item.radius, `${label}.radius`, GEOMETRY_LIMITS.minDimension, GEOMETRY_LIMITS.maxDimension / 2),
      inner_radius: number(item.inner_radius, `${label}.inner_radius`, 0, GEOMETRY_LIMITS.maxDimension / 2),
      rotation_deg: ((rawRotation % 360) + 360) % 360,
      refractive_index: number(item.refractive_index, `${label}.refractive_index`, GEOMETRY_LIMITS.minIndex, GEOMETRY_LIMITS.maxIndex),
    };
    if (object.kind === 'ring' && (object.inner_radius < GEOMETRY_LIMITS.minDimension || object.radius - object.inner_radius < GEOMETRY_LIMITS.minDimension - 1e-12)) {
      throw new Error(`${object.name}: ring inner radius and wall thickness must each be at least ${GEOMETRY_LIMITS.minDimension} µm.`);
    }
    const half = geometryHalfExtents(object);
    for (const axis of ['x', 'y', 'z'] as const) {
      if (Math.abs(object.position[axis]) + half[axis] > domain[axis] / 2 + 1e-9) {
        throw new Error(`${object.name} extends outside the domain along ${axis}. Move or resize the object, or enlarge the domain.`);
      }
    }
    return object;
  });
  return {
    schema: 'photonics-workbench-geometry', schema_version: 1, units: 'um',
    name: name(input.name, 'Document name'), mode: input.mode, domain, objects,
  };
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
    id, name: kind === 'box' ? 'Dielectric box' : kind === 'cylinder' ? 'Dielectric cylinder' : 'Dielectric ring',
    visible: true, kind, position: { x: 0, y: 0, z: 0 }, size: { x: 1, y: 0.5, z: 0.22 },
    radius: 1, inner_radius: 0.6, rotation_deg: 0, refractive_index: 3.45,
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
