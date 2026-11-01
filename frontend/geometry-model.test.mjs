// Node 24 executes TypeScript via native type stripping; no browser or solver required.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  createGeometryObject, createGeometryTemplate, createSlabPreview,
  parseGeometryDocument, serializeGeometryDocument, validateGeometryDocument,
  evaluateMaterial, geometryBounds, upgradeToV2, detachComponent,
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

const shared = JSON.parse(readFileSync(new URL('./geometry-fixtures.json', import.meta.url), 'utf8'));

test('shared v1/v2 fixtures preserve exactly normalized data and independent copies', () => {
  for (const fixture of shared.valid) {
    const input = clone(fixture.input);
    const result = validateGeometryDocument(input);
    assert.deepEqual(result, fixture.expected, fixture.name);
    assert.deepEqual(input, fixture.input, 'Validation must not mutate the caller.');
    assert.deepEqual(parseGeometryDocument(serializeGeometryDocument(result)), result);
    result.objects[0].position.x = 99;
    assert.deepEqual(input, fixture.input);
  }
});

test('shared invalid topology, material and component records are rejected', () => {
  for (const fixture of shared.invalid) assert.throws(() => validateGeometryDocument(fixture.input), undefined, fixture.name);
});

test('tabulated material interpolation is explicit and never extrapolates', () => {
  const material = shared.valid[1].input.materials[0];
  assert.deepEqual(evaluateMaterial(material, 1), { n: 2, k: 0 });
  assert.deepEqual(evaluateMaterial(material, 1.5), { n: 2.5, k: .1 });
  assert.deepEqual(evaluateMaterial(material, 2), { n: 3, k: .2 });
  for (const wavelength of [.5, 2.1, 0, NaN, Infinity, true, '1.5']) assert.throws(() => evaluateMaterial(material, wavelength));
  const constant = shared.valid[5].input.materials[0];
  assert.deepEqual(evaluateMaterial(constant, 10), { n: .005, k: 0 });
});

test('off-centre polygon bounds use actual vertices, position and z rotation', () => {
  const item = clone(shared.valid[1].input.objects[0]);
  Object.assign(item, { vertices: [{x:1,y:1},{x:3,y:1},{x:3,y:2},{x:1,y:2}], holes: [], position:{x:.5,y:-1,z:.3}, rotation_deg:90, size:{x:999,y:999,z:.4} });
  const actual = geometryBounds(item);
  for (const [bound, expected] of [['min',{x:-1.5,y:0,z:.1}],['max',{x:-.5,y:2,z:.5}]]) for (const axis of ['x','y','z']) assert.ok(Math.abs(actual[bound][axis]-expected[axis])<1e-12);
  const document = clone(shared.valid[1].input); document.objects=[item];
  assert.equal(validateGeometryDocument(document).objects[0].size.x,999,'Polygon XY comes from vertices, not legacy box dimensions.');
});

test('v1 upgrading is explicit and manual edits detach all component members', () => {
  const legacy = createGeometryTemplate('strip'), before = clone(legacy);
  const upgraded = upgradeToV2(legacy);
  assert.equal(upgraded.schema_version,2);assert.deepEqual(upgraded.objects,legacy.objects);
  assert.deepEqual(upgraded.materials,[]);assert.deepEqual(upgraded.components,[]);assert.deepEqual(legacy,before);
  const component = clone(shared.valid[4].input);
  const member = clone(component.objects[0]);member.id='second';member.position.x=.2;component.objects.push(member);component.components[0].object_ids.push('second');
  const detached = detachComponent(component,'core');
  assert.deepEqual(detached.components,[]);assert.ok(detached.objects.every(o=>o.component_id===undefined));
  assert.ok(detached.objects.every(o=>o.material_id==='core-material'));
  assert.equal(component.components.length,1);assert.ok(component.objects.every(o=>o.component_id==='instance-1'));
  assert.deepEqual(validateGeometryDocument(detached),detached);
});

test('v2 accepts bounded larger layouts while v1 limits remain unchanged', () => {
  const document=upgradeToV2(createGeometryTemplate('strip'));
  document.objects=Array.from({length:512},(_,i)=>createGeometryObject('box',`box-${i}`));
  assert.equal(validateGeometryDocument(document).objects.length,512);
  document.objects.push(createGeometryObject('box','too-many'));
  assert.throws(()=>validateGeometryDocument(document),/512/);
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
