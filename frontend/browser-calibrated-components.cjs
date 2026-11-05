/** Exercise a real local calibrated study, its complex samples and offline export. */
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
  const origin=process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/calibrated-components');
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
  page.setDefaultTimeout(60000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  try {
    await page.goto(origin);
    await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const lab=page.getByRole('region',{name:'Calibrated components',exact:true});
    await lab.getByRole('button',{name:'Multimode slab joint',exact:true}).click();
    const run=lab.getByRole('button',{name:'Run calibrated study'});
    assert.equal(await run.isDisabled(),true);
    await lab.getByLabel('Mesh resolution').selectOption('48');
    await lab.getByLabel('Lateral offset (µm)',{exact:true}).fill('0.125');
    await lab.locator('input[name="cal-target"][value="meep:cpu"]').check();
    const receipt=page.waitForResponse(r=>r.url().endsWith('/api/calibrations/jobs') && r.request().method()==='POST');
    await run.click();
    const response=await receipt;
    assert.equal(response.status(),202,await response.text());const job=await response.json();
    assert.equal(job.config.device,'cpu');assert.equal(job.config.case,'multimode');assert.equal(job.config.resolution,48);
    const deadline=Date.now()+3*60*60*1000;
    let record;
    do {
      record=await (await page.request.get(origin+'/api/jobs/'+job.id)).json();
      if(!['running','queued'].includes(record.status))break;
      await new Promise(resolve=>setTimeout(resolve,3500));
    }while(Date.now()<deadline);
    assert.equal(record.status,'completed',record.error || record.logs);
    const report=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/report')).json();
    const measured=lab.getByRole('region',{name:'Measured scattering'}).last();
    await measured.getByLabel('Input channel').selectOption('0');
    await measured.getByLabel('Sample wavelength').selectOption('1');
    const rows=measured.locator('tbody tr');
    assert.equal(await rows.count(),4);
    for(let i=0;i<4;i++){
      const cells=await rows.nth(i).locator('td').allTextContents();
      assert.deepEqual(cells,[report.s_real[1][i][0],report.s_imag[1][i][0],report.power[1][i][0]].map(x=>x.toPrecision(5)));
    }
    const pending=page.waitForEvent('download');
    await lab.getByRole('link',{name:'Export Python, notebook and measured arrays'}).click();
    await (await pending).saveAs(path.join(output,'export.zip'));
    await lab.screenshot({path:path.join(output,'calibration.png')});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    assert.deepEqual(errors,[]);
    const result={passed:true,checked_at:new Date().toISOString(),job_id:job.id,config:job.config,
      report_checks:report.checks,all_table_samples_equal:true,offline_bundle_downloaded:true,mobile_width:390,browser_errors:errors};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
