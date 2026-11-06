// Node 24 executes TypeScript via native type stripping; no browser or solver required.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  createGeometryObject, createGeometryTemplate, createSlabPreview,
  parseGeometryDocument, serializeGeometryDocument, validateGeometryDocument,
  evaluateMaterial, geometryBounds, upgradeToV2, detachComponent, connectedPorts, portRelation,
} from './src/geometryModel.ts';
import { commitDesign, deleteSelection, initialHistory, redoDesign, selectObject, undoDesign } from './src/designHistory.ts';
import { packHistory, unpackHistory } from './src/designRecovery.ts';

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

test('delete one entire instance without changing shared materials, other ports or overlap order', () => {
  const raw = clone(shared.valid[4].input);
  const core = raw.objects[0];
  raw.objects.push({ ...core, id: 'other', component_id: 'instance-2' }, { ...core, id: 'second' });
  raw.components[0].object_ids.push('second');
  raw.components.push({ ...raw.components[0], id: 'instance-2', object_ids: ['other'] });
  const original = validateGeometryDocument(raw);
  let history = selectObject(initialHistory(original), 'second');
  assert.equal(history.past.length, 0);
  const deleted = validateGeometryDocument(deleteSelection(original, 'second'));
  assert.deepEqual(deleted.objects, [original.objects[1]]);
  assert.deepEqual(deleted.components, [original.components[1]]);
  assert.deepEqual(deleted.materials, original.materials);
  history = commitDesign(history, deleted, null);
  assert.equal(history.present.selectedId, null);
  assert.deepEqual(undoDesign(history).present, { document: original, selectedId: 'second' });
  assert.deepEqual(redoDesign(undoDesign(history)).present, { document: deleted, selectedId: null });
});

test('empty designs, old primitives and missing selections are safe; redo follows only actual edits', () => {
  const legacy = createGeometryTemplate('strip');
  let history = initialHistory({ ...legacy, objects: [legacy.objects[1]] });
  const original = history.present;
  assert.equal(deleteSelection(original.document, null), original.document);
  assert.equal(deleteSelection(original.document, 'missing'), original.document);
  const empty = validateGeometryDocument(deleteSelection(original.document, original.selectedId));
  assert.deepEqual(empty.objects, []);
  assert.equal(empty.schema_version, 1);
  assert.equal('components' in empty, false);
  history = undoDesign(commitDesign(history, empty, null));
  assert.equal(history.future.length, 1);
  history = commitDesign(history, clone(history.present.document));
  assert.equal(history.future.length, 1);
  history = commitDesign(history, { ...original.document, name: 'Changed' });
  assert.equal(history.future.length, 0);
  assert.equal(redoDesign(history), history);
  assert.deepEqual(undoDesign(history).present, original);
});

function connectedCircuit() {
  const scene = clone(shared.valid[4].input);
  const part = scene.components[0], solid = scene.objects[0];
  const port = part.ports[0];
  part.ports.push({ ...port, name: 'out', x: 2, angle_deg: 0 });
  for (const i of [2, 3]) {
    const next = clone(part);
    next.id = 'instance-' + i;
    next.object_ids = ['core-' + i];
    scene.components.push(next);
    scene.objects.push({ ...clone(solid), id: next.object_ids[0], component_id: next.id });
  }
  scene.connections = [1, 2].map(i => ({
    id: 'link-' + i, a: { component_id: 'instance-' + i, port: 'out' },
    b: { component_id: 'instance-' + (i+1), port: 'in' },
  }));
  return validateGeometryDocument(scene);
}

test('persistent edges round-trip independently; absent connections stay absent on older documents', () => {
  const old = shared.valid[4].input;
  assert.equal(Object.hasOwn(parseGeometryDocument(serializeGeometryDocument(old)), 'connections'), false);
  assert.deepEqual(parseGeometryDocument(serializeGeometryDocument({ ...old, connections: [] })).connections, []);
  assert.throws(() => validateGeometryDocument({ ...createGeometryTemplate('strip'), connections: [] }));
  const linked = connectedCircuit();
  const parsed = parseGeometryDocument(serializeGeometryDocument(linked));
  assert.deepEqual(parsed, linked);
  parsed.connections[0].a.port = 'changed';
  assert.equal(linked.connections[0].a.port, 'out');
});

test('edge validation rejects duplicate endpoints in either order, missing references and malformed records', () => {
  const linked = connectedCircuit();
  const [one, two] = linked.connections;
  const invalid = [
    null, {}, [null], [{ ...one, unknown: 'field' }], [{ ...one, id: '../bad' }],
    [{ ...one, b: { ...one.b, port: 'absent' } }], [{ ...one, b: { ...one.b, component_id: 'absent' } }],
    [{ ...one, a: { ...one.a, x: 1 } }], [{ ...one, b: { ...one.a, port: 'in' } }],
    [one, { ...two, id: one.id }], [one, { id: 'other', a: one.b, b: one.a }],
    [one, { ...two, a: one.a }], new Array(16385),
  ];
  for (const connections of invalid) assert.throws(() => validateGeometryDocument({ ...linked, connections }));
});

test('connection status measures world XYZ, antiparallel directions and width tolerance', () => {
  const scene = connectedCircuit();
  const pair = connectedPorts(scene)[0];
  assert.deepEqual(pair.relation.delta, { x: -4, y: 0, z: 0 });
  assert.equal(pair.relation.distance, 4);
  assert.equal(pair.relation.aligned, false);
  const a = { ...pair.a, x: 0, y: 0, z: 0, angle_deg: 350, width_um: 2 };
  const b = { ...a, angle_deg: 170 };
  assert.equal(portRelation(a, b).aligned, true);
  assert.equal(portRelation(a, { ...b, z: 1e-8 }).aligned, true);
  assert.equal(portRelation(a, { ...b, z: 1.1e-8 }).aligned, false);
  assert.equal(portRelation(a, { ...b, angle_deg: 170.0000002 }).aligned, false);
  assert.equal(portRelation(a, { ...b, width_um: 2 + 1.9e-9 }).aligned, true);
  assert.equal(portRelation(a, { ...b, width_um: 2 + 2.1e-9 }).aligned, false);
  assert.equal(portRelation(a, { ...b, x: 3, y: 4, z: 12 }).distance, 13);
  Object.assign(scene.components[1].ports[0], { x: pair.a.x });
  assert.equal(connectedPorts(scene)[0].relation.aligned, true);
  assert.deepEqual(scene.connections, connectedCircuit().connections);
});

test('deleting or detaching a component removes only incident edges and recovers through saved history', () => {
  const scene = connectedCircuit(), selectedId = scene.objects[0].id;
  for (const operation of [deleteSelection, detachComponent]) {
    const after = validateGeometryDocument(operation(scene, selectedId));
    assert.deepEqual(after.connections, [scene.connections[1]]);
    assert.deepEqual(after.materials, scene.materials);
    assert.deepEqual(after.components, scene.components.slice(1));
    assert.equal(scene.connections.length, 2);
    const history = commitDesign(selectObject(initialHistory(scene), selectedId), after, null);
    assert.deepEqual(undoDesign(history).present, { document: scene, selectedId });
    const undo = undoDesign(history);
    const restored = unpackHistory(packHistory(undo).text, serializeGeometryDocument(scene), scene);
    assert.deepEqual(redoDesign(restored), history);
  }
});

test('design history retains the latest fifty changes and selections without mutating prior snapshots', () => {
  const document = createGeometryTemplate('ring');
  let history = initialHistory(document);
  for (let i = 1; i <= 55; i++) {
    history = commitDesign(history, { ...document, name: String(i) });
  }
  assert.equal(history.past.length, 50);
  for (let i = 0; i < 50; i++) history = undoDesign(history);
  assert.equal(history.present.document.name, '5');
  assert.equal(history.future.length, 50);
  assert.equal(undoDesign(history), history);
  for (let i = 0; i < 50; i++) history = redoDesign(history);
  assert.equal(history.present.document.name, '55');
  assert.equal(redoDesign(history), history);
});

test('tab checkpoints require the exact saved draft and validate every restored document and selection', () => {
  const doc = createGeometryTemplate('ring');
  const changed = { ...doc, name: 'Edited' };
  const history = undoDesign(commitDesign(initialHistory(doc), changed));
  const checkpoint = packHistory(history);
  assert.ok(checkpoint && !checkpoint.trimmed);
  const saved = serializeGeometryDocument(doc);
  assert.deepEqual(unpackHistory(checkpoint.text, saved, doc), history);
  assert.deepEqual(unpackHistory(checkpoint.text, serializeGeometryDocument(changed), changed), initialHistory(changed));
  for (const corrupt of [
    '{', { ...JSON.parse(checkpoint.text), version: 2 },
    { ...JSON.parse(checkpoint.text), future: [{ document: changed, selectedId: 'missing' }] },
    { ...JSON.parse(checkpoint.text), past: [{ document: { ...doc, schema_version: 99 }, selectedId: null }] },
  ]) assert.deepEqual(unpackHistory(typeof corrupt === 'string' ? corrupt : JSON.stringify(corrupt), saved, doc), initialHistory(doc));
});

test('history storage budget preserves adjacent undo/redo steps and the in-memory history remains complete', () => {
  const first = createGeometryTemplate('strip');
  let history = initialHistory(first);
  for (let i = 1; i <= 12; i++) history = commitDesign(history, { ...first, name: String(i) });
  for (let i = 0; i < 6; i++) history = undoDesign(history);
  const budget = JSON.stringify(history.present).length * 2 * 6;
  const packet = packHistory(history, budget);
  assert.ok(packet.trimmed);
  assert.ok(packet.text.length * 2 <= budget);
  const cut = JSON.parse(packet.text);
  assert.ok(cut.past.length > 0 && cut.future.length > 0);
  assert.deepEqual(cut.past, history.past.slice(-cut.past.length));
  assert.deepEqual(cut.future, history.future.slice(0, cut.future.length));
  assert.equal(packHistory(history, 32), null);
  assert.equal(history.past.length, 6);
  assert.equal(history.future.length, 6);
});
