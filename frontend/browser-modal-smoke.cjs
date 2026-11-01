/** Real 3D modal journey. PHOTONICS_TEST_TARGET=meep:cpu|fdtdx:cpu|fdtdx:gpu.
 * PHOTONICS_MODAL_RUN_ID reopens an actual completed run without another solve.
 * Optional PHOTONICS_URL, PHOTONICS_BROWSER_CHANNEL, PHOTONICS_ARTIFACT_DIR.
 * Numerical data are never mocked; no external solver/cloud service is used.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
  const base=process.env.PHOTONICS_URL||'http://127.0.0.1:8000';
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR||'test-results/modal');fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:process.env.PHOTONICS_BROWSER_CHANNEL||'chrome'});
  const page=await browser.newPage({viewport:{width:1580,height:1100},acceptDownloads:true});
  page.setDefaultTimeout(90000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const panel=page.getByRole('region',{name:'Finite-height 3D modal ports',exact:true});
  let id=process.env.PHOTONICS_MODAL_RUN_ID||null;
  const readOnly=!!id,target=process.env.PHOTONICS_TEST_TARGET||'meep:cpu';
  let submissions=0;
  page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/ports3d/jobs'))submissions++;});
  async function json(url){const response=await page.request.get(base+url);assert.equal(response.status(),200,url);return response.json();}
  async function download(locator){const waiting=page.waitForEvent('download');await locator.click();const d=await waiting;const dest=path.join(output,d.suggestedFilename());await d.saveAs(dest);assert.ok(fs.statSync(dest).size>100);return path.basename(dest);}
  try{
    await page.goto(base);await page.getByRole('button',{name:'3D ports & S-matrix',exact:true}).click();
    await panel.getByRole('button',{name:'Open 3D gap example',exact:true}).click();
    await panel.getByText('3D modal study validated',{exact:true}).waitFor();
    const run=panel.getByRole('button',{name:'Run 3D S-matrix',exact:true});assert.ok(await run.isDisabled(),'Target must be explicitly selected.');
    await panel.getByLabel('Modal Gap length (µm)',{exact:true}).fill('.01');
    await panel.locator('.modal-validation.invalid').waitFor();assert.ok(await run.isDisabled());
    await panel.getByLabel('Modal Gap length (µm)',{exact:true}).fill('.375');
    await panel.getByText('3D modal study validated',{exact:true}).waitFor();
    if(readOnly){await panel.getByLabel('Saved 3D modal run',{exact:true}).selectOption(id);}
    else{
      await panel.locator(`input[name="modal-target"][value="${target}"]`).check();
      const submitted=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/api/ports3d/jobs'));
      await run.click();const response=await submitted;assert.equal(response.status(),202);id=(await response.json()).id;
      console.log(JSON.stringify({event:'submitted',run_id:id,target}));
      const deadline=Date.now()+1220000;
      while(Date.now()<deadline){const job=await json(`/api/jobs/${id}`);if(job.status==='completed')break;if(['failed','cancelled'].includes(job.status))throw new Error(JSON.stringify({status:job.status,error:job.error,logs:job.logs}));await page.waitForTimeout(1500);}
    }
    await panel.locator('.modal-run-status[data-status="completed"]').waitFor({timeout:30000});
    await panel.locator(`.modal-results[data-run-id="${id}"]`).waitFor();
    const job=await json(`/api/jobs/${id}`),data=await json(`/api/jobs/${id}/sparameters`);
    assert.equal(job.config.study_type,'waveguide-ports-3d');assert.equal(job.status,'completed');
    if(!readOnly){assert.equal(`${job.config.solver}:${job.config.device}`,target);assert.equal(job.config.gap_um,.375);}
    assert.ok(data.valid_bin.every(Boolean),'The real example must settle with guided modes at every bin.');
    for(const metric of ['power','magnitude','phase_unwrapped_rad','real','imag']){
      await panel.getByLabel('S-parameter plot quantity',{exact:true}).selectOption(metric);
      await panel.locator(`.modal-chart[data-metric="${metric}"]`).waitFor();
      for(const entry of ['S11','S21','S12','S22']){
        const dots=panel.locator(`g[data-entry="${entry}"] circle`);
        const expected=data.parameters[entry][metric];assert.equal(await dots.count(),expected.filter(v=>v!==null).length);
        for(let i=0;i<expected.length;i++)if(expected[i]!==null)assert.equal(Number(await panel.locator(`g[data-entry="${entry}"] circle[data-bin="${i}"]`).getAttribute('data-value')),expected[i]);
      }
    }
    await panel.getByLabel('Modal spectral sample',{exact:true}).focus();await panel.getByLabel('Modal spectral sample',{exact:true}).press('Home');
    for(const entry of ['S11','S21','S12','S22']){
      assert.equal(await panel.locator(`[data-matrix-entry="${entry}"] [data-complex="real"]`).innerText(),`Re ${data.parameters[entry].real[0].toPrecision(6)}`);
      assert.equal(await panel.locator(`[data-matrix-entry="${entry}"] [data-complex="imag"]`).innerText(),`Im ${data.parameters[entry].imag[0].toPrecision(6)}`);
    }
    const csv=await download(panel.getByRole('link',{name:'Export complex CSV',exact:true}));
    const apiCsv=await page.request.get(`${base}/api/jobs/${id}/sparameters.csv`);assert.equal(fs.readFileSync(path.join(output,csv),'utf8'),await apiCsv.text());
    const spectrum=await download(panel.getByRole('button',{name:'Export complex JSON',exact:true}));assert.deepEqual(JSON.parse(fs.readFileSync(path.join(output,spectrum),'utf8')),data);
    const results=await download(panel.getByRole('link',{name:'Export 3D modal results .zip',exact:true}));
    const code=await download(panel.locator('.modal-output>.modal-actions').getByRole('link',{name:'Export run Python + notebook',exact:true}));
    await panel.locator('.modal-fields>summary').click();await panel.locator('.slice3d-canvas').waitFor();
    await panel.getByLabel('3D slice plane',{exact:true}).selectOption('xz');await panel.getByLabel('3D field component',{exact:true}).selectOption('Ez');
    await panel.locator('.slice3d-canvas[data-plane="xz"][data-component="Ez"]').waitFor();
    const canvas=panel.locator('.slice3d-canvas'),index=Number(await canvas.getAttribute('data-slice-index'));
    const slice=await json(`/api/jobs/${id}/slice?plane=xz&index=${index}&component=Ez`),volume=await json(`/api/jobs/${id}/volume`);
    const pixel=await canvas.evaluate(c=>{const x=Math.floor(c.width*.57),y=Math.floor(c.height*.42);return{x,y,height:c.height,rgba:[...c.getContext('2d').getImageData(x,y,1,1).data]};});
    const v=slice.field[pixel.height-pixel.y-1][pixel.x],norm=Math.max(-1,Math.min(1,v/(volume.field_peaks_by_component.Ez||1))),color=norm<0?[73,141,249]:[255,157,104];
    assert.deepEqual(pixel.rgba,[...[19,28,41].map((c,i)=>Math.round(c+(color[i]-c)*Math.abs(norm))),255]);
    await panel.locator('.run-passport>summary').click();await panel.locator('.passport-file code').first().waitFor();
    assert.ok((await panel.locator('.passport-file').allTextContents()).some(t=>t.includes('sparameters.npz')));
    if(job.result?.implementation_snapshot){
      assert.ok((await panel.locator('.passport-file').allTextContents()).some(t=>t.includes('implementation.zip')&&t.includes(job.result.implementation_snapshot.sha256)));
      await panel.getByText('Original numerical source is preserved from execution.',{exact:false}).waitFor();
    }
    const displayedSolver=await panel.locator('.passport-facts>div').first().innerText();assert.ok(displayedSolver.includes(job.config.solver==='meep'?'Meep':'FDTDX'));
    await panel.locator('.run-passport>summary').click();await panel.locator('.modal-fields>summary').click();
    await panel.getByLabel('S-parameter plot quantity',{exact:true}).selectOption('phase_unwrapped_rad');
    await panel.locator('.modal-output').screenshot({path:path.join(output,'modal-complex-phase.png')});
    await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Page must fit a phone viewport.');
    await panel.locator('.modal-results').screenshot({path:path.join(output,'modal-mobile.png')});
    await page.setViewportSize({width:1580,height:1100});
    await panel.getByLabel('Modal Gap length (µm)',{exact:true}).fill('.5');await panel.getByText('The editor differs from this saved run.',{exact:false}).waitFor();
    assert.deepEqual((await json(`/api/jobs/${id}`)).config,job.config);
    assert.deepEqual(errors,[]);
    const report={passed:true,checked_at:new Date().toISOString(),run_id:id,config:job.config,actual_device:data.metadata.actual_device,device_name:data.metadata.device_name,new_solver_jobs:submissions,read_only:readOnly,valid_bins:data.valid_bin.filter(Boolean).length,downloads:{csv,spectrum,results,code},checks:['Explicit target required; invalid physical input rejected','Real four-direction S-matrix and measured settling','All plotted power/amplitude/phase/real/imaginary samples equal actual API values','Complex matrix real/imaginary values inspectable by keyboard','CSV and JSON downloads exactly equal saved API data','Result and standalone Python/notebook ZIP downloads','Actual finite-height field slice pixel matches saved Ez values','Passport solver and complex array hashes visible','Immutable saved configuration after editing','390px layout without page overflow','No browser exceptions'],browser_errors:errors};
    fs.writeFileSync(path.join(output,'modal-browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
