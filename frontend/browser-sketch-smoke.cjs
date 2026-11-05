/** Draw through the production GUI, inspect the exact saved outlines and download both design formats. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/xy-sketch');
  fs.mkdirSync(output,{recursive:true});
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const browser = await chromium.launch({headless:true,channel:process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome'});
  const page = await browser.newPage({viewport:{width:1500,height:1040},acceptDownloads:true});
  page.setDefaultTimeout(30000);
  const plan = page.getByRole('region',{name:'XY layout editor'});
  const board=plan.getByRole('img',{name:'XY layout drawing surface'});
  const button = (name, root=page) => root.getByRole('button',{name,exact:true});
  const input = name => page.getByLabel(name,{exact:true});
  const storageKey = 'photonics-workbench.geometry.v1';
  const saved = () => page.evaluate(key=>JSON.parse(localStorage.getItem(key)),storageKey);
  const errors=[],checks=[];
  page.on('pageerror',e=>errors.push(e.message));
  async function download(locator,name) {
    const pending=page.waitForEvent('download'); await locator.click();
    const file=path.join(output,name); await (await pending).saveAs(file);return file;
  }
  async function xy(x,y) {
    await board.scrollIntoViewIfNeeded();
    // Use the displayed physical domain border to target points a person can click.
    return board.evaluate((svg,[x,y,dx,dy])=>{
      const domain=svg.querySelector(':scope > rect');
      const pos=new DOMPoint(domain.x.baseVal.value+(x/dx+.5)*domain.width.baseVal.value,
        domain.y.baseVal.value+(.5-y/dy)*domain.height.baseVal.value).matrixTransform(svg.getScreenCTM());
      return {x:pos.x,y:pos.y};
    },[x,y,domain.x,domain.y]);
  }
  async function clickWorld(x,y) {const p=await xy(x,y);await page.mouse.click(p.x,p.y);}
  async function enter(x,y,action='Add point') {
    await input('Sketch point X').fill(String(x));await input('Sketch point Y').fill(String(y));await button(action,plan).click();
  }
  async function expectSaved(predicate,args) {
    await page.waitForFunction(predicate,args);return saved();
  }
  const point=(x,y)=>({x,y});
  const world=(solid,p)=>{const a=solid.rotation_deg*Math.PI/180;return{x:solid.position.x+Math.cos(a)*p.x-Math.sin(a)*p.y,
    y:solid.position.y+Math.sin(a)*p.x+Math.cos(a)*p.y};};
  let domain;
  try {
    await page.goto(base);await page.locator('.geometry-3d-host canvas').waitFor();
    const initialPath=await download(button('Export geometry'),'initial.json');
    const initial=JSON.parse(fs.readFileSync(initialPath,'utf8'));domain=initial.domain;
    await button('Draw in XY').click();
    // Start with an unpublished route. These shortcuts must only change the draft.
    const baseline=await saved();
    const handles=board.locator('[data-plan-point]');
    const draftShortcut=async keys=>{await board.focus();await page.keyboard.press(keys);};
    const waitPointX=value=>page.waitForFunction(x=>document.querySelector('[aria-label="Sketch point X"]').value===x,value);
    await clickWorld(-2,0);await clickWorld(0,0);await clickWorld(0,2);
    assert.equal(await handles.count(),3);
    await page.keyboard.press('Control+z');assert.equal(await handles.count(),2);
    await page.keyboard.press('Control+Shift+z');assert.equal(await handles.count(),3);
    await page.keyboard.press('Meta+z');assert.equal(await handles.count(),2);
    await page.keyboard.press('Control+y');assert.equal(await handles.count(),3);
    await input('Selected sketch point').selectOption('1');
    await enter(.125,.75,'Update point');
    await draftShortcut('Delete');assert.equal(await handles.count(),2);
    await page.keyboard.press('Control+z');
    assert.equal(await input('Selected sketch point').inputValue(),'1');
    await waitPointX('0.125');
    await page.keyboard.press('Control+z');await waitPointX('0');
    await clickWorld(1,2);
    assert.equal(await button('Redo point edit',plan).isDisabled(),true);
    // A pointer gesture is one edit. A cancelled gesture leaves the original points and history intact.
    const p0=await xy(1,2),p1=await xy(1.5,2.25);
    await page.mouse.move(p0.x,p0.y);await page.mouse.down();await page.mouse.move(p1.x,p1.y,{steps:5});await page.mouse.up();
    await waitPointX('1.5');
    await draftShortcut('Control+z');await waitPointX('1');
    assert.equal(await handles.count(),4);
    await page.mouse.move(p0.x,p0.y);await page.mouse.down();await page.mouse.move(p1.x,p1.y,{steps:4});
    await board.dispatchEvent('pointercancel',{pointerId:1,isPrimary:true});await page.mouse.up();
    await waitPointX('1');
    assert.equal(await button('Redo point edit',plan).isEnabled(),true);
    await button('Redo point edit',plan).click();await waitPointX('1.5');
    await input('Sketch point X').press('Control+z');await input('Sketch point X').press('Delete');
    assert.equal(await handles.count(),4);
    await draftShortcut('Escape');assert.equal(await handles.count(),0);
    assert.deepEqual(await saved(),baseline);
    await button('Route',plan).click();
    checks.push('Scoped undo/redo and point deletion preserve the unpublished scene, restore the exact selected vertex, protect numeric input, and cancel safely. A drag is one edit, a cancelled drag preserves redo, and a new branch clears redo.');
    await input('Sketch name').fill('Routed guide');
    await input('Sketch extrusion height').fill('.3');await input('Sketch Z centre').fill('.1');
    await input('Sketch route width').fill('.5');
    for (const p of [[-4.02,-.96],[-2.04,-.98],[-2.02,1.03],[1.04,1.01]]) await clickWorld(...p);
    assert.equal(await board.locator('[data-plan-point]').count(),4);
    await plan.screenshot({path:path.join(output,'xy-route-preview.png')});
    await button('Place route',plan).click();
    let doc=await expectSaved(({key,count})=>JSON.parse(localStorage.getItem(key))?.objects.length===count,{key:storageKey,count:initial.objects.length+1});
    let solid=doc.objects.at(-1);assert.equal(solid.name,'Routed guide');assert.equal(solid.size.z,.3);assert.equal(solid.position.z,.1);
    const expect=[[-4,-.75],[-2.25,-.75],[-2.25,1.25],[1,1.25],[1,.75],[-1.75,.75],[-1.75,-1.25],[-4,-1.25]];
    const actual=solid.vertices.map(p=>world(solid,p)).map(p=>[p.x,p.y]);
    assert.deepEqual(new Set(actual.map(String)),new Set(expect.map(String)));
    checks.push('Pointer centreline snapped to the selected physical grid, with exact perpendicular caps, mitered corners and extrusion.');
    const routeId=solid.id, routeDoc=doc;
    await button('Undo').click();
    await expectSaved(({key,count})=>JSON.parse(localStorage.getItem(key))?.objects.length===count,{key:storageKey,count:initial.objects.length});
    assert.deepEqual(await saved(),initial);
    await button('Redo').click();await page.waitForFunction(id=>JSON.parse(localStorage.getItem('photonics-workbench.geometry.v1')).objects.some(o=>o.id===id),routeId);
    assert.deepEqual(await saved(),routeDoc);
    const from=await xy(-3,-1),to=await xy(-2.5,-.75);
    await page.mouse.move(from.x,from.y);await page.mouse.down();await page.mouse.move(to.x,to.y,{steps:6});await page.mouse.up();
    // Snap moves the solid's anchor to the nearest grid location.
    await page.waitForFunction(([id,x])=>JSON.parse(localStorage.getItem('photonics-workbench.geometry.v1')).objects.find(o=>o.id===id)?.position.x!==x,[routeId,solid.position.x]);
    doc=await saved(); solid=doc.objects.find(o=>o.id===routeId);
    assert.equal(solid.position.x,Math.round((routeDoc.objects.at(-1).position.x+.5)*4)/4);
    checks.push('Whole-document undo/redo restored exact scenes. A pointer drag committed a validated move.');
    await button('Polygon',plan).click();await clickWorld(1,-1);await clickWorld(3,-1);await clickWorld(2,1);
    await button('Undo').click();await button('Use current design',plan).waitFor();
    assert.equal(await button('Place polygon',plan).isDisabled(),true);assert.deepEqual(await saved(),routeDoc);
    await button('Use current design',plan).click();assert.equal(await button('Place polygon',plan).isEnabled(),true);
    await button('Redo').click();await button('Use current design',plan).waitFor();
    assert.equal(await button('Place polygon',plan).isDisabled(),true);
    await button('Cancel sketch',plan).click();assert.deepEqual(await saved(),doc);
    checks.push('Document undo/redo invalidates stale sketch placement. Explicit retargeting permits a new drawing only on the reviewed document.');
    await button('Polygon',plan).click();
    for (const p of [[1,-1],[3,1],[1,1],[3,-1]]) await clickWorld(...p);
    await button('Place polygon',plan).click();await plan.getByRole('alert').waitFor();
    assert.match(await plan.getByRole('alert').innerText(),/self-intersect/);
    assert.deepEqual(await saved(),doc);await button('Cancel sketch',plan).click();assert.deepEqual(await saved(),doc);
    await button('Polygon',plan).click();await input('Sketch name').fill('Tapered island');
    for(const p of [[1.125,-.75],[3,-1],[2.75,.75],[1.5,.5]]) await enter(...p);
    await button('Undo point edit',plan).click();assert.equal(await board.locator('[data-plan-point]').count(),3);
    await enter(1.5,.5);
    await button('Place polygon',plan).click();
    doc=await expectSaved(({key,count})=>JSON.parse(localStorage.getItem(key))?.objects.length===count,{key:storageKey,count:initial.objects.length+2});
    let island=doc.objects.at(-1);assert.equal(island.name,'Tapered island');
    assert.ok(island.vertices.map(p=>world(island,p)).some(p=>p.x===1.125));
    await input('Rotation Z').fill('30');await input('Rotation Z').press('Enter');
    await page.waitForFunction(id=>JSON.parse(localStorage.getItem('photonics-workbench.geometry.v1')).objects.find(o=>o.id===id).rotation_deg===30,island.id);
    island=(await saved()).objects.at(-1);
    await button('Edit contour',plan).click();
    await button('Undo').click();await plan.getByText(/Cancel and reopen the contour/).waitFor();
    assert.equal(await button('Apply contour',plan).isDisabled(),true);
    assert.equal(await button('Use current design',plan).count(),0);
    await button('Cancel sketch',plan).click();await button('Redo').click();
    await button('Edit contour',plan).click();
    const before=world(island,island.vertices[0]);
    await enter(before.x-.12,before.y,'Update point');await button('Apply contour',plan).click();
    await plan.getByText('Contour updated.',{exact:true}).waitFor();
    doc=await saved();const edited=doc.objects.at(-1);
    assert.deepEqual(edited.vertices.slice(1),island.vertices.slice(1));
    assert.ok(Math.abs(world(edited,edited.vertices[0]).x-before.x+.12)<1e-12);
    assert.equal(edited.rotation_deg,30);
    checks.push('Invalid crossed boundary and cancellation retained the scene. Numeric points preserved off-grid values, undo-point edited the draft, and editing in world coordinates retained a rotated solid’s untouched local vertices.');
    await button('Route',plan).click();
    await input('Sketch name').fill('Circular bend');await input('Sketch route width').fill('.4');
    await input('Sketch bend radius').fill('3');
    for (const p of [[-1.5,2],[.5,2],[.5,3.5]]) await enter(...p);
    await button('Place route',plan).click();await plan.getByRole('alert').waitFor();
    assert.deepEqual(await saved(),doc);
    await input('Sketch bend radius').fill('.6');
    await plan.screenshot({path:path.join(output,'xy-rounded-route.png')});
    await button('Place route',plan).click();
    doc=await expectSaved(({key,count})=>JSON.parse(localStorage.getItem(key))?.objects.length===count,{key:storageKey,count:initial.objects.length+3});
    assert.equal(doc.objects.at(-1).vertices.length,42);
    checks.push('Circular corners use tangent inner/outer radii. An oversized radius is rejected before placement.');
    const geomFile=await download(button('Export geometry'),'sketched-layout.json');
    assert.deepEqual(JSON.parse(fs.readFileSync(geomFile,'utf8')),doc);
    await download(button('Export geometry Python + notebook'),'sketched-layout-python-notebook.zip');
    await input('Import geometry file').setInputFiles(geomFile);
    await page.reload();await page.locator('.geometry-3d-host canvas').waitFor();assert.deepEqual(await saved(),doc);
    await button('Draw in XY').click();
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await plan.screenshot({path:path.join(output,'xy-sketch-mobile.png')});
    await page.setViewportSize({width:1500,height:1040});
    const library=page.locator('.component-library');
    await library.locator(':scope > summary').click();
    await library.getByRole('button',{name:'Choose component Directional coupler',exact:true}).click();
    const opening=page.waitForResponse(r=>r.url().endsWith('/api/library/build')&&r.request().method()==='POST');
    await library.getByRole('button',{name:'Open as new design',exact:true}).click();
    const instance=(await (await opening).json()).geometry;
    await page.waitForFunction(id=>JSON.parse(localStorage.getItem('photonics-workbench.geometry.v1'))?.components?.[0]?.id===id,instance.components[0].id);
    domain=instance.domain;
    await button('Select / move',plan).click();
    const selectedSolid=instance.objects.at(-1);
    const start=await xy(selectedSolid.position.x,selectedSolid.position.y);
    const end=await xy(selectedSolid.position.x+.25,selectedSolid.position.y+.25);
    await page.mouse.move(start.x,start.y);await page.mouse.down();await page.mouse.move(end.x,end.y,{steps:5});await page.mouse.up();
    const movedInstance=await expectSaved(({key,old})=>JSON.parse(localStorage.getItem(key)).components[0].placement.x!==old,{key:storageKey,old:instance.components[0].placement.x});
    const movedRecipe=movedInstance.components[0], oldRecipe=instance.components[0];
    assert.equal(movedRecipe.id,oldRecipe.id);assert.deepEqual(movedRecipe.object_ids,oldRecipe.object_ids);
    for(const [i,p] of movedRecipe.ports.entries()){
      assert.equal(p.x,oldRecipe.ports[i].x+.25);assert.equal(p.y,oldRecipe.ports[i].y+.25);assert.equal(p.angle_deg,oldRecipe.ports[i].angle_deg);
    }
    for(const [i,o] of movedInstance.objects.entries()){assert.equal(o.position.x,instance.objects[i].position.x+.25);assert.equal(o.position.y,instance.objects[i].position.y+.25);}
    checks.push('Dragging a coupler translates all members, port coordinates and recipe placement together, preserving membership and directions.');
    assert.deepEqual(errors,[]);
    checks.push('Exact JSON export/import/reload and Python/notebook ZIP download, 390 px layout, no browser exceptions.');
    const report={passed:true,checked_at:new Date().toISOString(),browser:await browser.version(),checks,new_objects:3,browser_errors:errors};
    fs.writeFileSync(path.join(output,'sketch-browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  } catch(e) {await page.screenshot({path:path.join(output,'sketch-failure.png'),fullPage:true}).catch(()=>{});throw e;}
  finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
