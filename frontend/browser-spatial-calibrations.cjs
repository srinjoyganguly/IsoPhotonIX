/** Exercise the full-vector CUDA route and compare the canvas values to native sampled data. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async()=>{
  const origin = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const directory = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/spatial-calibration');
  fs.mkdirSync(directory,{recursive:true});
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:960},acceptDownloads:true});
    page.setDefaultTimeout(180000);
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin);
    await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const panel=page.getByRole('region',{name:'Calibrated components',exact:true});
    await panel.getByRole('button',{name:'3D finite-width grating',exact:true}).click();
    await panel.getByLabel('3D geometry').selectOption('compact');
    await panel.getByLabel('Mesh resolution').selectOption('12');
    await panel.locator('input[value="fdtdx:gpu"]').check();
    assert.equal(await panel.getByRole('radio').count(),1);
    const designDownload=page.waitForEvent('download');
    await panel.getByRole('button',{name:/Export Python/i}).click();
    await (await designDownload).saveAs(path.join(directory,'compact-program.zip'));
    const created=page.waitForResponse(response=>response.url().endsWith('/api/calibrations/jobs') &&
      response.request().method()==='POST');
    await panel.getByRole('button',{name:'Run calibrated study',exact:true}).click();
    const response=await created;
    const job=await response.json();
    assert.equal(response.status(),202,JSON.stringify(job));
    fs.writeFileSync(path.join(directory,'started.json'),JSON.stringify({job_id:job.id,config:job.config},null,2)+'\n');
    await page.waitForFunction(async id=>{
      const r=await fetch('/api/jobs/'+id);
      const job=await r.json();
      return !['running','queued'].includes(job.status);
    },job.id,{timeout:6*3600*1000,polling:3000});
    const saved=await (await page.request.get(origin+'/api/jobs/'+job.id)).json();
    assert.equal(saved.status,'completed',saved.error);
    const fields=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/field')).json();
    const report=await (await page.request.get(origin+'/api/calibrations/jobs/'+job.id+'/report')).json();
    const local=panel.getByRole('region',{name:'Measured field plane'});
    await local.locator('canvas').waitFor();
    const x=Math.floor(fields.horizontal_um.length/2), y=Math.floor(fields.vertical_um.length/2);
    assert((await local.getByLabel('Field sample').textContent()).endsWith(fields.values[y][x].toPrecision(7)));
    await local.locator('canvas').focus(); await page.keyboard.press('ArrowRight');
    assert((await local.getByLabel('Field sample').textContent()).endsWith(fields.values[y][x+1].toPrecision(7)));
    const spectrum=panel.getByRole('region',{name:'Measured spectrum'}).last();
    await spectrum.getByLabel('Wavelength sample').selectOption({label:'1.55000 µm'});
    assert.equal(await spectrum.getByLabel('Measured power fraction').textContent(),
        report.result.center_beam_to_guide.toPrecision(7));
    const resultDownload=page.waitForEvent('download');
    await panel.getByRole('link',{name:'Export Python, notebook and measured arrays'}).click();
    await (await resultDownload).saveAs(path.join(directory,'compact-results.zip'));
    await local.screenshot({path:path.join(directory,'measured-field.png')});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(directory,'report.json'),JSON.stringify({passed:true,job_id:job.id,checked_at:new Date().toISOString(),
      actual_device:report.actual_device,device_name:report.device_name,geometry:report.settings.geometry,
      resolution:report.settings.resolution,coupling_at_1550:report.result.center_beam_to_guide,
      field_component:fields.component,field_shape:fields.source_shape,center_sample:fields.values[y][x],
      checked_adjacent_sample:fields.values[y][x+1],sample_units:'solver units',browser_errors:errors},null,2)+'\n');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
