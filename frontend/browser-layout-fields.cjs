/** Real finite-layout study, native fields and replay ZIP using the normal UI. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');

(async()=>{
  const target=process.env.PHOTONICS_TEST_TARGET || 'meep:cpu';
  const base=process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/layout');
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({acceptDownloads:true,viewport:{width:1480,height:960}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  const panel=page.getByRole('region',{name:'Finite-height 3D simulation',exact:true});
  async function read(url){const r=await page.request.get(base+url);assert.equal(r.status(),200);return r.json();}
  async function save(locator,name){
    const pending=page.waitForEvent('download');await locator.click();
    const download=await pending;await download.saveAs(path.join(output,name));
  }
  try {
    await page.goto(base);
    const loading=page.waitForResponse(r=>r.url().endsWith('/api/layout/example'));
    await panel.getByRole('button',{name:'Open finite coupler example'}).click();
    const example=await (await loading).json();
    await panel.getByText('3D study validated',{exact:true}).waitFor();
    const run=panel.getByRole('button',{name:'Run 3D simulation',exact:true});
    assert.ok(await run.isDisabled());
    await panel.getByLabel('Dipole x (µm)',{exact:true}).fill('-0.85');
    await panel.locator('input[value="'+target+'"]').check();
    const submit=page.waitForResponse(r=>r.url().endsWith('/api/layout/jobs') && r.request().method()==='POST');
    await run.click();
    const response=await submit;assert.equal(response.status(),202,await response.text());
    const job=await response.json();assert.equal(job.config.source_um.x,-.85);
    assert.deepEqual(job.config.geometry.objects.map(o=>[o.kind,o.position,o.size]),example.geometry.objects.map(o=>[o.kind,o.position,o.size]));
    await panel.locator('.status-tag.completed').waitFor({timeout:300000});
    const saved=await read('/api/jobs/'+job.id);
    const volume=await read('/api/jobs/'+job.id+'/volume');
    assert.equal(saved.result.requested_solver+':'+saved.result.requested_device,target);
    if(target.endsWith(':gpu'))assert.match(saved.result.actual_device,/cuda|gpu/i);
    for(const plane of ['xy','xz','yz']){
      await panel.getByLabel('3D slice plane',{exact:true}).selectOption(plane);
      for(const component of ['Ex','Ey','Ez']){
        await panel.getByLabel('3D field component',{exact:true}).selectOption(component);
        const canvas=panel.locator('canvas[aria-label^="Actual 3D '+component+' '+plane.toUpperCase()+' slice"]');
        await canvas.waitFor();
        const index=Number(await canvas.getAttribute('data-slice-index'));
        const slice=await read('/api/jobs/'+job.id+'/slice?plane='+plane+'&index='+index+'&component='+component);
        const sample=await canvas.evaluate(c=>{
          const x=Math.floor(c.width*.57),y=Math.floor(c.height*.43);
          return {x,y,h:c.height,pixel:[...c.getContext('2d').getImageData(x,y,1,1).data]};
        });
        const val=slice.field[sample.h-1-sample.y][sample.x];
        const factor=Math.min(1,Math.abs(val)/(volume.field_peaks_by_component[component] || 1));
        const color=val<0?[73,141,249]:[255,157,104];
        assert.deepEqual(sample.pixel,[...[19,28,41].map((v,i)=>Math.round(v+(color[i]-v)*factor)),255]);
      }
    }
    await panel.getByLabel('3D slice plane',{exact:true}).selectOption('xy');
    await panel.getByLabel('3D field component',{exact:true}).selectOption('Ey');
    await panel.getByRole('button',{name:'Material grid',exact:true}).click();
    await panel.locator('canvas[aria-label^="Actual 3D relative permittivity XY slice"]').waitFor();
    await panel.screenshot({path:path.join(output,'layout-numerical-study.png')});
    await save(panel.getByRole('link',{name:'Export layout run Python + notebook'}),'layout-run-python-notebook.zip');
    await save(panel.getByRole('button',{name:'Save run configuration'}),'layout-run-config.json');
    const passport=await read('/api/jobs/'+job.id+'/passport');assert.ok(passport.checks.every(c=>c.passed));
    assert.deepEqual(errors,[]);
    const evidence={passed:true,preflight:false,new_solver_jobs:1,checked_at:new Date().toISOString(),run_id:job.id,target,
      actual_device:saved.result.actual_device,device_name:saved.result.device_name,shape:volume.shape,
      material_grid_check:saved.result.material_grid_check,compiled_material_sha256:saved.result.geometry_translation.epsilon_xyz_sha256,
      checks:['Edited dipole and explicit execution target','Actual nine plane/component canvas samples match stored solver slices',
        'Native material grid, passport and immutable config','Downloaded runnable Python and notebook'],browser_errors:errors};
    fs.writeFileSync(path.join(output,'layout-browser.json'),JSON.stringify(evidence,null,2)+'\n');console.log(JSON.stringify(evidence));
  } finally {await browser.close();}
})().catch(err=>{console.error(err);process.exitCode=1});
