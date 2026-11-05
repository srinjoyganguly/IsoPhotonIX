/** Check target-specific exports and run a native compact grating through the UI. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const origin = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/calibrated-grating');
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true, channel:'chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:960},acceptDownloads:true});
    page.setDefaultTimeout(180000);
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin);
    await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const panel=page.getByRole('region',{name:'Calibrated components',exact:true});
    await panel.getByRole('button',{name:'Y splitter',exact:true}).click();
    await panel.locator('input[value="fdtdx:gpu"]').check();
    await panel.getByLabel('Material sampling').selectOption('yee-aligned');
    await panel.getByLabel('Mesh resolution').selectOption('16');
    const program=page.waitForEvent('download');
    await panel.getByRole('button',{name:'Export Python + notebook',exact:true}).click();
    await (await program).saveAs(path.join(output,'splitter-gpu.zip'));
    await panel.getByRole('button',{name:'Grating radiation',exact:true}).click();
    await panel.getByLabel('Measurement').selectOption('tidy3d-compact');
    await panel.getByLabel('Mesh resolution').selectOption('40');
    await panel.locator('input[value="meep:cpu"]').check();
    const kickoff=page.waitForResponse(r=>r.url().endsWith('/api/calibrations/jobs') && r.request().method()==='POST');
    await panel.getByRole('button',{name:'Run calibrated study'}).click();
    const response=await kickoff;
    assert.equal(response.status(),202,await response.text());
    const job=await response.json();
    const deadline=Date.now()+90*60*1000;
    let saved;
    do {
      saved=await (await page.request.get(origin+'/api/jobs/'+job.id)).json();
      if(!['running','queued'].includes(saved.status))break;
      await page.waitForTimeout(4000);
    } while(Date.now()<deadline);
    assert.equal(saved.status,'completed',saved.error);
    const report=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/report')).json();
    assert.equal(report.passed,true);
    assert(Math.abs(report.result.center_beam_to_guide-.582341311)<1e-6);
    await panel.getByRole('region',{name:'Measured spectrum'}).last().getByRole('img').waitFor();
    const exported=page.waitForEvent('download');
    await panel.getByRole('link',{name:'Export Python, notebook and measured arrays'}).click();
    await (await exported).saveAs(path.join(output,'grating-result.zip'));
    await panel.screenshot({path:path.join(output,'grating.png')});
    assert.deepEqual(errors,[]);
    const audit={passed:true,checked_at:new Date().toISOString(),job_id:job.id,config:job.config,
      coupling_fraction:report.result.center_beam_to_guide,plotted_samples:report.result.wavelength_um.length,
      gpu_design_export:true,result_export:true,browser_errors:errors};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(audit,null,2)+'\n');
    console.log(JSON.stringify(audit));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
