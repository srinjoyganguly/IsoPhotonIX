/** Real local component/material/geometry UI checks. No electromagnetic data is fabricated. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR||'test-results/library');fs.mkdirSync(output,{recursive:true});
  const base=process.env.PHOTONICS_URL||'http://127.0.0.1:8000';
  const browser=await chromium.launch({headless:true,channel:process.env.PHOTONICS_BROWSER_CHANNEL||'chrome'});
  const page=await browser.newPage({viewport:{width:1600,height:1100},acceptDownloads:true});page.setDefaultTimeout(60000);
  const errors=[],checks=[];let jobs=0;page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST'&&/\/api\/(?:jobs|jobs3d|ports\/jobs|ports3d\/jobs|taper\/jobs)$/.test(r.url()))jobs++;});
  const library=page.locator('.component-library'),saved=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('isophotonix.geometry.v1')));
  const input=name=>page.getByLabel(name,{exact:true}),button=name=>page.getByRole('button',{name,exact:true});
  async function commit(label,value){await input(label).fill(String(value));await input(label).press('Enter');}
  async function action(name,endpoint){const ready=page.waitForResponse(r=>r.url().endsWith(endpoint)&&r.request().method()==='POST');await library.getByRole('button',{name,exact:true}).click();const response=await ready;return{response,data:await response.json()};}
  async function waitSaved(geometry){await page.waitForFunction(expected=>{const canonical=v=>JSON.stringify(v,(_k,item)=>item&&typeof item==='object'&&!Array.isArray(item)?Object.fromEntries(Object.keys(item).sort().map(k=>[k,item[k]])):item);return canonical(JSON.parse(localStorage.getItem('isophotonix.geometry.v1')))===canonical(expected);},geometry);}
  async function download(locator,name){const waiting=page.waitForEvent('download');await locator.click();const d=await waiting;const file=path.join(output,name);await d.saveAs(file);assert.ok(fs.statSync(file).size>100);return file;}
  try{
    await page.goto(base);await page.locator('.geometry-3d-host canvas').first().waitFor();
    await library.locator(':scope>summary').click();
    const response=await page.request.get(base+'/api/library');assert.equal(response.status(),200);const catalog=await response.json();assert.equal(catalog.components.length,27);
    await library.getByRole('button',{name:`Choose component ${catalog.components[0].name}`,exact:true}).waitFor();
    assert.equal(await library.locator('.component-card').count(),catalog.components.length);
    await input('Search photonic components').fill('there-is-no-such-part');await library.getByText('No components match this search.',{exact:false}).waitFor();assert.equal(await library.locator('.component-card').count(),0);
    await input('Search photonic components').fill('');await input('Component category').selectOption('laser-foundations');assert.equal(await library.locator('.component-card').count(),2);await input('Component category').selectOption('all');
    checks.push(`Search and category filtering cover all ${catalog.components.length} current component families.`);
    const opened=[];
    for(const family of catalog.components){
      const card=library.getByRole('button',{name:`Choose component ${family.name}`,exact:true});
      const taper=family.id==='taper';
      assert.equal(family.simulation_support,taper?'bounded-taper-fields':'bounded-layout-fields');
      assert.deepEqual(family.eligible_studies,taper?['layout-3d','taper-3d','taper-ports-3d']:['layout-3d']);
      assert.equal(await card.locator('.component-card-status').innerText(),taper?'Taper fields and modal ports':'Finite-layout fields');
      await card.click();
      const {response,data}=await action('Open as new design','/api/library/build');assert.equal(response.status(),200,JSON.stringify(data));
      await waitSaved(data.geometry);const design=await saved();assert.equal(design.schema_version,2);assert.equal(design.components[0].family,family.id);assert.ok(design.objects.length>0);
      const canvas=page.locator('.geo-stage-column .geometry-3d-host canvas');await page.waitForFunction(count=>document.querySelector('.geo-stage-column canvas')?.dataset.visibleObjects===String(count),design.objects.filter(o=>o.visible).length);
      assert.equal(await canvas.getAttribute('data-layout-ports'),String(design.components[0].ports.length));opened.push({family:family.id,objects:design.objects.length,ports:design.components[0].ports.length,eligible_studies:family.eligible_studies});
    }
    checks.push('Every catalog family displayed its actual API study capabilities, generated through the API, persisted exact v2 geometry and rendered actual solids and layout ports.');
    const fan=catalog.components.find(f=>f.id==='fan_grating');
    await library.getByRole('button',{name:`Choose component ${fan.name}`,exact:true}).click();
    const openedFan=await action('Open as new design','/api/library/build');assert.equal(openedFan.response.status(),200);
    await waitSaved(openedFan.data.geometry);
    const fields=page.getByRole('region',{name:'Finite-height 3D simulation',exact:true});
    await fields.locator('input[name=study3d-target][value="meep:cpu"]').check();
    const presetRequest=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/layout/example'&&new URL(r.url()).searchParams.get('family')==='fan_grating');
    await fields.getByRole('button',{name:`Open tested ${fan.name} field setup`,exact:true}).click();
    const presetResponse=await presetRequest;assert.equal(presetResponse.status(),200);const preset=await presetResponse.json();
    await waitSaved(preset.geometry);assert.notDeepEqual(preset.geometry,openedFan.data.geometry);
    assert.equal(await fields.getByLabel('3D excitation model').inputValue(),'layout-3d');
    await fields.getByText('3D study validated',{exact:true}).waitFor();
    assert.equal(await fields.locator('input[name=study3d-target]:checked').count(),0);
    for(const axis of ['x','y','z'])assert.equal(Number(await fields.getByLabel(`Dipole ${axis.toUpperCase()} (µm)`).inputValue()),preset.source_um[axis]);
    assert.equal(Number(await fields.getByLabel('3D Resolution',{exact:true}).inputValue()),preset.resolution);
    assert.equal(Number(await fields.getByLabel('3D Duration',{exact:true}).inputValue()),preset.duration_fs);
    assert.ok(await fields.getByRole('button',{name:'Run 3D simulation',exact:true}).isDisabled());
    await button('Undo').click();await waitSaved(openedFan.data.geometry);
    checks.push('Fan-grating tested preset returned the exact measured finite scene, mesh, duration and source; target cleared. Undo restored the original library geometry.');
    await library.getByRole('button',{name:'Choose component Linear taper',exact:true}).click();
    let built=await action('Open as new design','/api/library/build');assert.equal(built.response.status(),200);await waitSaved(built.data.geometry);
    const initial=await saved();await input('Component Length').fill('8');
    built=await action('Update component','/api/library/place');assert.equal(built.response.status(),200,JSON.stringify(built.data));await waitSaved(built.data.geometry);
    assert.equal((await saved()).components[0].parameters.length_um,8);assert.equal((await saved()).components[0].id,initial.components[0].id);
    const beforeInvalid=await saved();await input('Component Output width').fill('0');
    const rejected=await action('Update component','/api/library/place');assert.equal(rejected.response.status(),422);await library.getByRole('alert').waitFor();assert.deepEqual(await saved(),beforeInvalid);await input('Component Output width').fill(String(beforeInvalid.components[0].parameters.output_width_um));
    await library.locator('.library-placement>summary').click();await input('Component placement y').fill('8');
    const outside=await action('Add to layout','/api/library/place');assert.equal(outside.response.status(),422);assert.deepEqual(await saved(),beforeInvalid);
    await library.getByLabel('Enlarge domain to fit when adding or updating',{exact:true}).check();built=await action('Add to layout','/api/library/place');assert.equal(built.response.status(),200,JSON.stringify(built.data));await waitSaved(built.data.geometry);
    const layout=await saved();assert.equal(layout.components.length,2);assert.ok(layout.domain.y>beforeInvalid.domain.y);
    const selectedId=built.data.component.object_ids[0];await commit('Position X',.1);await page.waitForFunction(()=>JSON.parse(localStorage.getItem('isophotonix.geometry.v1')).components.length===1);assert.ok(!(await saved()).objects.find(o=>o.id===selectedId).component_id);
    await button('Undo').click();await waitSaved(layout);await button('Convert component to editable solids').click();
    checks.push('Component regeneration preserves instance identity; invalid parameters/domain placement retain the scene; explicit domain fit and atomic manual detach/undo work.');

    const polygon=page.getByRole('region',{name:'Polygon sketch editor',exact:true});await polygon.waitFor();
    const beforeHole=await saved();await polygon.getByRole('button',{name:'Add hole',exact:true}).click();await polygon.getByRole('button',{name:'Apply polygon sketch',exact:true}).click();
    await page.waitForFunction(id=>(JSON.parse(localStorage.getItem('isophotonix.geometry.v1')).objects.find(o=>o.id===id).holes||[]).length===1,selectedId);
    let current=await saved();assert.equal(current.objects.find(o=>o.id===selectedId).holes.length,1);
    await input('Polygon contour').selectOption('1');const x=await input('Vertex 1 X').inputValue();await input('Vertex 1 X').fill('999');await polygon.getByRole('button',{name:'Apply polygon sketch',exact:true}).click();
    await page.locator('.geometry-editor>.error-banner').waitFor();assert.deepEqual(await saved(),current);await button('Dismiss geometry error').click();await input('Vertex 1 X').fill(x);
    await input('Polygon contour').selectOption('0');await polygon.getByRole('button',{name:'Sketch contour 0 vertex 1',exact:true}).focus();await polygon.getByRole('button',{name:'Sketch contour 0 vertex 1',exact:true}).press('ArrowLeft');await polygon.getByRole('button',{name:'Apply polygon sketch',exact:true}).click();
    await page.waitForFunction(({id,x})=>JSON.parse(localStorage.getItem('isophotonix.geometry.v1')).objects.find(o=>o.id===id).vertices[0].x<x,{id:selectedId,x:current.objects.find(o=>o.id===selectedId).vertices[0].x});
    const afterKeyboard=await saved();const sketch=polygon.locator('svg');const point=polygon.getByRole('button',{name:'Sketch contour 0 vertex 2',exact:true});const bounds=await point.boundingBox();await point.scrollIntoViewIfNeeded();const visible=await point.boundingBox();await page.mouse.move(visible.x+visible.width/2,visible.y+visible.height/2);await page.mouse.down();await page.mouse.move(visible.x+visible.width/2-3,visible.y+visible.height/2+3,{steps:4});await page.mouse.up();await polygon.getByRole('button',{name:'Apply polygon sketch',exact:true}).click();
    await page.waitForFunction(previous=>JSON.stringify(JSON.parse(localStorage.getItem('isophotonix.geometry.v1')))!==previous,JSON.stringify(afterKeyboard));
    checks.push('Real polygon hole editing, invalid-hole retention, keyboard vertex movement and pointer dragging update validated geometry.');

    const material=page.getByRole('region',{name:'Optical material editor',exact:true});await material.locator('.material-custom>summary').click();await input('Custom material model').selectOption('tabulated-nk');await input('Custom material name').fill('Browser tabulated material');
    await input('Import material CSV').setInputFiles({name:'input-material.csv',mimeType:'text/csv',buffer:Buffer.from('wavelength_um,n,k\n1.45,2,0.01\n1.55,2.1,0.02\n1.65,2.2,0.03\n')});
    await input('Material source').fill('User-supplied browser test samples; not measured material evidence');await material.getByRole('button',{name:'Save as new & assign',exact:true}).click();
    await page.waitForFunction(id=>JSON.parse(localStorage.getItem('isophotonix.geometry.v1')).objects.find(o=>o.id===id).refractive_index===2.1,selectedId);
    current=await saved();const materialId=current.objects.find(o=>o.id===selectedId).material_id;assert.deepEqual(current.materials.find(m=>m.id===materialId).k,[.01,.02,.03]);
    await input('Material reference wavelength').fill('2');await material.getByRole('button',{name:'Save as new & assign',exact:true}).click();await material.getByRole('alert').waitFor();assert.deepEqual(await saved(),current);await input('Material reference wavelength').fill('1.55');
    const example=await (await page.request.get(base+'/api/example3d')).json();const unsupported=await page.request.post(base+'/api/validate3d',{data:{...example,geometry:current}});assert.equal(unsupported.status(),422);
    const taperValidation = await page.request.post(base+'/api/taper/validate',{data:{...example,schema_version:5,study_type:'taper-3d',geometry:current}});
    assert.equal(taperValidation.status(),422);
    assert.ok(await page.getByRole('button',{name:'Run 3D simulation',exact:true}).isDisabled());
    checks.push('Local CSV n/k samples, reference index and provenance preserved; no extrapolation; current solvers reject unqualified geometry/materials before execution.');
    const jsonPath=await download(button('Export geometry'),'library-layout.json');assert.deepEqual(JSON.parse(fs.readFileSync(jsonPath,'utf8')),current);
    await download(button('Export geometry Python + notebook'),'library-layout-python-notebook.zip');
    await input('Import geometry file').setInputFiles(jsonPath);await waitSaved(current);await page.reload();await page.locator('.geo-stage-column canvas').waitFor();assert.deepEqual(await saved(),current);
    checks.push('Geometry JSON import/export/reload is exact; actual Python/notebook design bundle downloaded.');
    await page.locator('.component-library>summary').click();await page.locator('.component-library').screenshot({path:path.join(output,'component-library.png')});
    for(const [width,height] of [[1280,900],[800,1000],[390,844]]){await page.setViewportSize({width,height});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No horizontal overflow at ${width}px`);}
    await page.screenshot({path:path.join(output,'library-mobile.png'),fullPage:true});
    assert.equal(jobs,0);assert.deepEqual(errors,[]);
    const report={passed:true,checked_at:new Date().toISOString(),browser:await browser.version(),opened,checks,new_solver_jobs:jobs,browser_errors:errors,material_sample_scope:'Synthetic user input tests material storage/interpolation only; no optical material or electromagnetic accuracy claim.',geometry:current};
    fs.writeFileSync(path.join(output,'library-browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,families:opened.length,checks:checks.length,new_solver_jobs:jobs}));
  }catch(error){await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
