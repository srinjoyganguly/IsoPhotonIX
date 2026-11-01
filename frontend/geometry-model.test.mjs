// Node 24 executes TypeScript via native type stripping; no browser or solver required.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createGeometryObject, createGeometryTemplate, createSlabPreview,
  parseGeometryDocument, serializeGeometryDocument, validateGeometryDocument,
} from './src/geometryModel.ts';

const clone = value => JSON.parse(JSON.stringify(value));

test('every bundled 3D draft round-trips as a distinct format from solver configuration', () => {
  for (const template of ['strip', 'coupler', 'ring']) {
    const document = createGeometryTemplate(template);
    assert.equal(document.mode, 'draft-3d');
    assert.equal(document.units, 'um');
    assert.deepEqual(parseGeometryDocument(serializeGeometryDocument(document)), document);
  }
  assert.throws(() => parseGeometryDocument('{"schema_version":1,"core_width_um":0.6}'), /unsupported field|required|different format/);
});

test('slab preview follows validated XY dimensions and identifies its illustrative extrusion', () => {
  const preview = createSlabPreview({ length_um: 10, height_um: 5, core_width_um: 0.8, core_index: 2 });
  assert.equal(preview.mode, 'slab-preview');
  assert.equal(preview.domain.x, 10);
  assert.equal(preview.domain.y, 5);
  assert.equal(preview.objects[0].size.y, 0.8);
  assert.equal(preview.objects[0].refractive_index, 2);
  assert.match(preview.name, /illustrative/);
});

test('validator returns an independent normalized document without mutating imported data', () => {
  const input = createGeometryTemplate('ring');
  input.name = '  My chip  ';
  input.objects[1].rotation_deg = -90;
  const validated = validateGeometryDocument(input);
  assert.equal(validated.name, 'My chip');
  assert.equal(validated.objects[1].rotation_deg, 270);
  assert.equal(input.objects[1].rotation_deg, -90);
  validated.objects[0].position.z = 0;
  assert.notEqual(input.objects[0].position.z, 0);
});

test('empty scene is a valid editable document and IDs survive export', () => {
  const input = createGeometryTemplate('strip');
  input.objects = [];
  assert.deepEqual(validateGeometryDocument(input).objects, []);
  const a = createGeometryObject('box'), b = createGeometryObject('box');
  assert.notEqual(a.id, b.id);
  input.objects = [a, b];
  assert.equal(parseGeometryDocument(serializeGeometryDocument(input)).objects[0].id, a.id);
});

test('duplicate object IDs cannot alias selection or edits', () => {
  const input = createGeometryTemplate('coupler');
  input.objects[2].id = input.objects[1].id;
  assert.throws(() => validateGeometryDocument(input), /duplicate id/);
});

test('numeric strings and nonfinite values are rejected rather than coerced', () => {
  for (const bad of ['0.5', NaN, Infinity, -Infinity, null]) {
    const input = createGeometryTemplate('strip');
    input.objects[1].size.y = bad;
    assert.throws(() => validateGeometryDocument(input), /finite number/);
  }
});

test('rotation accounts for the full box footprint at domain edges', () => {
  const input = createGeometryTemplate('strip');
  input.domain = { x: 4, y: 4, z: 3 };
  input.objects = [{ ...createGeometryObject('box', 'diagonal'), size: { x: 4, y: 4, z: 1 }, rotation_deg: 45 }];
  assert.throws(() => validateGeometryDocument(input), /extends outside/);
  input.objects[0].size = { x: 2, y: 2, z: 1 };
  assert.doesNotThrow(() => validateGeometryDocument(input));
});

test('cylinder containment uses radius and ring containment uses outer radius', () => {
  for (const kind of ['cylinder', 'ring']) {
    const input = createGeometryTemplate('strip');
    input.objects = [{ ...createGeometryObject(kind), position: { x: 5.1, y: 0, z: 0 } }];
    assert.throws(() => validateGeometryDocument(input), /outside the domain along x/);
    input.objects[0].position.x = 5;
    assert.doesNotThrow(() => validateGeometryDocument(input));
    input.objects[0].position.z = 1.2;
    assert.throws(() => validateGeometryDocument(input), /outside the domain along z/);
  }
});

test('hidden objects still have physically bounded geometry', () => {
  const input = createGeometryTemplate('strip');
  input.objects[1].visible = false;
  input.objects[1].position.x = 20;
  assert.throws(() => validateGeometryDocument(input), /extends outside/);
});

test('rings must have a positive hole and resolvable positive wall', () => {
  for (const inner of [0, 1.6, 2, 1.5999]) {
    const input = createGeometryTemplate('ring');
    input.objects[1].inner_radius = inner;
    assert.throws(() => validateGeometryDocument(input), /inner radius and wall thickness/);
  }
});

test('malformed and unrelated JSON cannot become a geometry document', () => {
  for (const text of ['', '{', 'null', '[]', '42']) assert.throws(() => parseGeometryDocument(text));
  const valid = createGeometryTemplate('strip');
  for (const patch of [{ units: 'm' }, { schema_version: 2 }, { mode: 'solved' }, { name: '' }, { name: 'bad\nname' }, { secret: 'unsupported' }]) {
    assert.throws(() => validateGeometryDocument({ ...valid, ...patch }));
  }
  for (const patch of [{ visible: 'true' }, { kind: 'sphere' }, { id: '../bad' }, { refractive_index: -1 }, { size: { x: 1, y: 1 } }]) {
    const input = clone(valid);
    Object.assign(input.objects[1], patch);
    assert.throws(() => validateGeometryDocument(input));
  }
});

test('document size and object count are bounded', () => {
  assert.throws(() => parseGeometryDocument(' '.repeat(1_000_001)), /smaller than 1 MB/);
  const input = createGeometryTemplate('strip');
  input.objects = Array.from({ length: 101 }, (_, index) => createGeometryObject('box', `box-${index}`));
  assert.throws(() => validateGeometryDocument(input), /at most 100/);
  input.objects.pop();
  assert.equal(validateGeometryDocument(input).objects.length, 100);
});
