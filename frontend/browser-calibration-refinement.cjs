/** Exercise the calibrated wall, domain, timing and polygon controls in a running local workbench. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const origin=process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const out=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/calibration-refinement');
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:960}});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin);
    await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const lab=page.getByRole('region',{name:'Calibrated components',exact:true});
    await lab.getByRole('button',{name:'Y splitter',exact:true}).click();
    await lab.locator('input[value="fdtdx:gpu"]').check();
    await lab.getByLabel('Material sampling').selectOption('polygon-area');
    await lab.getByLabel('Mesh resolution').selectOption('16');
    let job;
    if (process.env.PHOTONICS_RUN_SPLITTER==='1') {
      const posted=page.waitForResponse(r=>r.url().endsWith('/api/calibrations/jobs') && r.request().method()==='POST');
      await lab.getByRole('button',{name:'Run calibrated study'}).click();
      const response=await posted;
      assert.equal(response.status(),202,await response.text());
      job=await response.json();
      assert.equal(job.config.material_model,'polygon-area');
      fs.writeFileSync(path.join(out,'job.json'),JSON.stringify({id:job.id,config:job.config}));
    }
    const catalog=await (await page.request.get(origin+'/api/calibrations')).json();
    for (const preset of ['ring-3d','grating-3d']) {
      await lab.getByRole('button',{name:catalog.find(c=>c.case===preset).name,exact:true}).click();
      await lab.getByText('Phase and weak reflections',{exact:true}).click();
      assert((await lab.getByLabel('Comparison').locator('option').count())>0);
      await lab.getByText('GPU timing records',{exact:true}).click();
      const timings=await (await page.request.get(origin+'/api/calibrations/performance?case='+preset)).json();
      const row=timings.find(r=>r.synchronized_step_seconds!==null);
      assert(row);
      await lab.getByLabel('Timing record').selectOption(row.receipt);
      assert((await lab.locator('.cal-time-scroll').allTextContents()).some(t=>t.includes(row.cells.toLocaleString('en-US'))));
      const csv=await (await page.request.get(origin+'/api/calibrations/performance.csv?case='+preset)).text();
      assert(csv.includes(row.acquisition_id) && csv.includes('cuda_event_elapsed_seconds'));
      await lab.getByLabel('3D geometry').selectOption('lumerical');
      if (preset==='ring-3d') {
        await lab.getByLabel('Ring boundary sampling').selectOption('wall-tensor');
        assert(!(await lab.getByLabel('Mesh resolution').locator('option').allTextContents()).includes('12 cells/µm'));
      } else {
        await lab.getByLabel('Grating material sampling').selectOption('direct-yee');
        await lab.getByLabel('Simulation domain').selectOption('mirror-y');
      }
      await lab.locator('input[value="fdtdx:gpu"]').check();
      const download=page.waitForEvent('download');
      await lab.getByRole('button',{name:'Export Python + notebook',exact:true}).click();
      await (await download).saveAs(path.join(out,preset+'-program.zip'));
    }
    if (job) {
      const deadline=Date.now()+3*60*60*1000;
      let run;
      do {
        run=await (await page.request.get(origin+'/api/jobs/'+job.id)).json();
        if (!['running','queued'].includes(run.status)) break;
        await new Promise(resolve=>setTimeout(resolve,4000));
      } while (Date.now()<deadline);
      assert.equal(run.status,'completed',run.error || run.logs);
      const report=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/report')).json();
      assert.equal(report.actual_device,'cuda:0');
      assert.equal(report.valid_column.length,3);
      assert(report.valid_column.every(Boolean));
      await lab.getByRole('button',{name:'Y splitter',exact:true}).click();
      await lab.getByLabel('Saved run').selectOption(job.id);
      await lab.getByText('GPU timing for this run',{exact:true}).click();
      const stages=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/performance')).json();
      assert.equal(stages.length,6);
      assert(stages.every(s=>s.synchronized_step_seconds>0 && s.cuda_event_elapsed_seconds>0));
      const dl=page.waitForEvent('download');
      await lab.getByRole('link',{name:'Export Python, notebook and measured arrays'}).click();
      await (await dl).saveAs(path.join(out,'splitter-result.zip'));
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({passed:true,splitter_gpu_job:Boolean(job),
      timings_and_csv:['ring-3d','grating-3d'],exports:['wall-tensor','mirror-y'],page_errors:errors},null,2));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
