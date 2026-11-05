import test from 'node:test';
import assert from 'node:assert/strict';
import { addSketch, editContour, routeOutline, worldPoint, moveSelection, snapPoint } from './src/sketchGeometry.ts';
import { initialPointHistory, pointHistory } from './src/sketchHistory.ts';
import { createGeometryTemplate, geometryBounds } from './src/geometryModel.ts';
const scene = createGeometryTemplate('ring');
const v = (x,y) => ({x,y});
test('right-angle route has perpendicular flat caps and constant full width on both straight legs', () => {
  const outline = routeOutline([v(-3,0),v(0,0),v(0,2)], .5);
  assert.deepEqual(new Set(outline.map(p=>JSON.stringify(p))),new Set([v(-3,.25),v(-.25,.25),v(-.25,2),v(.25,2),v(.25,-.25),v(-3,-.25)].map(p=>JSON.stringify(p))));
  const area = Math.abs(outline.reduce((sum,a,i)=>{const b=outline[(i+1)%outline.length];return sum+a.x*b.y-a.y*b.x;},0))/2;
  assert.equal(area, 2.5);
  for (const chain of [[v(0,0),v(1,0),v(0,0)],[v(-2,-2),v(2,2),v(-2,2),v(2,-2)],[v(0,0),v(.1,0),v(.1,.05),v(.2,.05)]]) {
    assert.throws(()=>routeOutline(chain,.5));
  }
});
test('polygon edit keeps existing local transforms, untouched coordinates and holes; invalid placement is atomic', () => {
  const {document,id}=addSketch(scene,[v(1,1),v(2,1),v(2,2),v(1,2)],.3,.2,'Shape');
  const solid=document.objects.find(o=>o.id===id);
  solid.rotation_deg=30;
  solid.holes=[[v(-.1,-.1),v(-.1,.1),v(.1,.1),v(.1,-.1)]];
  const outline=solid.vertices.map(p=>worldPoint(solid,p));
  outline[0].x-=.12;
  const updated=editContour(document,id,0,outline,.4,.23).objects.find(o=>o.id===id);
  assert.deepEqual(updated.vertices.slice(1),solid.vertices.slice(1));
  assert.deepEqual(updated.holes,solid.holes);
  assert.equal(updated.rotation_deg,30);
  assert.ok(Math.abs(worldPoint(updated,updated.vertices[0]).x-outline[0].x)<1e-12);
  assert.equal(solid.size.z,.3);
  assert.throws(()=>moveSelection(document,id,v(11,0)),/outside the domain/);
  assert.throws(()=>addSketch(scene,[v(-2,0),v(0,2),v(-2,2),v(0,0)],.3,0,'Crossed'),/self-intersect/);
  assert.equal(scene.schema_version,1);
  assert.deepEqual(snapPoint(v(-.63, .36), .25),v(-.75,.25));
  const moved=moveSelection(document,id,v(1,.25)).objects.find(o=>o.id===id);
  assert.equal(moved.position.x,solid.position.x+1);
  assert.ok(geometryBounds(moved).max.x>geometryBounds(solid).max.x);
});

test('circular fillets preserve inner and outer radii with valid tangencies on both turn signs', () => {
  for (const sign of [-1,1]) {
    const outline=routeOutline([v(0,0),v(2,0),v(2,2*sign)],.5,.75);
    const interior=outline.filter(p=>Math.abs(p.x)>1e-12 && Math.abs(p.y-2*sign)>1e-12);
    assert.equal(interior.length,38);
    const radii=interior.map(p=>Math.hypot(p.x-1.25,p.y-.75*sign));
    assert.equal(radii.filter(r=>Math.abs(r-.5)<1e-12).length,19);
    assert.equal(radii.filter(r=>Math.abs(r-1)<1e-12).length,19);
  }
  assert.throws(()=>routeOutline([v(0,0),v(.4,0),v(.4,1)],.5,.75),/consume/);
  assert.throws(()=>routeOutline([v(0,0),v(2,0),v(2,2)],.5,.25),/half/);
});

test('point undo and redo restore the exact vertices and selection across update, deletion and empty drafts', () => {
  const original = [v(-.025,0),v(1.125,-.2),v(0,1)];
  let state = initialPointHistory(original,1);
  const edited = [original[0],v(1.125,.75),original[2]];
  state = pointHistory(state,{type:'write',points:edited,selected:1});
  state = pointHistory(state,{type:'write',points:[edited[0], edited[2]],selected:0});
  state = pointHistory(state,{type:'undo'});
  assert.deepEqual(state.present,{points:edited,selected:1});
  state = pointHistory(state,{type:'undo'});
  assert.deepEqual(state.present,{points:original,selected:1});
  state = pointHistory(state,{type:'redo'});
  state = pointHistory(state,{type:'redo'});
  assert.deepEqual(state.present,{points:[original[0],original[2]],selected:0});
  const cleared = pointHistory(state,{type:'write',points:[],selected:0});
  assert.deepEqual(cleared.present,{points:[],selected:-1});
  assert.deepEqual(pointHistory(cleared,{type:'undo'}).present,state.present);
});

test('selecting and zero-distance edits preserve redo; branching or cancelling clears it', () => {
  let state = initialPointHistory();
  for (let i = 0; i < 57; i++) state = pointHistory(state,{type:'write',points:[v(i,0)],selected:0});
  assert.equal(state.past.length,50);
  state = pointHistory(state,{type:'undo'});
  state = pointHistory(state,{type:'select', selected:0});
  assert.strictEqual(pointHistory(state,{type:'write',points:[v(55,0)],selected:0}),state);
  assert.equal(state.future.length,1);
  state = pointHistory(state,{type:'write',points:[v(.1,-.25)],selected:0});
  assert.equal(state.future.length,0);
  state = pointHistory(state,{type:'reset',points:[],selected:-1});
  assert.deepEqual(state,initialPointHistory());
});
