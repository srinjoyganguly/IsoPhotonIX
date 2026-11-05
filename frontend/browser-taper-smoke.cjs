/** Exercise the actual library-to-field-study journey. Start the local workbench first.
 * PHOTONICS_TAPER_PREFLIGHT=1 checks setup/export without submitting a solver.
 * PHOTONICS_TAPER_DESIGN can point at taper-geometry.json from the first run
 * so different execution targets reuse identical component and solid IDs.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/taper');
const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
const target = process.env.PHOTONICS_TEST_TARGET || 'meep:cpu';
const preflight = process.env.PHOTONICS_TAPER_PREFLIGHT === '1';
const fromDesign = process.env.PHOTONICS_TAPER_DESIGN;

function nearest(values, x) {
  return values.reduce((best, value, i) => Math.abs(value - x) < Math.abs(values[best] - x) ? i : best, 0);
}

async function inspectField(panel, data, peak) {
  const values = data.field;
  let strongest = { x: 0, y: 0, value: 0 };
  for (let y = 0; y < values.length; y++) {
    for (let x = 0; x < values[y].length; x++) {
      assert.ok(Number.isFinite(values[y][x]));
      if (Math.abs(values[y][x]) > Math.abs(strongest.value)) strongest = { x, y, value: values[y][x] };
    }
  }
  const canvas = panel.locator(`canvas[data-plane="${data.plane}"][data-component="${data.component}"][data-slice-index="${data.index}"]`);
  await canvas.waitFor();
  const pixel = await canvas.evaluate((element, {x, y}) => [...element.getContext('2d').getImageData(x, element.height - 1 - y, 1, 1).data], strongest);
  const fraction = Math.min(1, Math.abs(strongest.value / (peak || 1)));
  const color = strongest.value < 0 ? [73, 141, 249] : [255, 157, 104];
  const expected = [19, 28, 41].map((channel, i) => Math.round(channel + (color[i] - channel) * fraction));
  assert.deepEqual(pixel, [...expected, 255]);
  return Math.abs(strongest.value);
}

async function main() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome'});
  const page = await browser.newPage({viewport:{width:1500, height:1050}, acceptDownloads:true});
  page.setDefaultTimeout(30000);
  const library = page.locator('.component-library');
  const panel = page.getByRole('region', {name:'Finite-height 3D simulation', exact:true});
  const run = panel.getByRole('button', {name:'Run 3D simulation', exact:true});
  const errors = [], submissions = [], checks = [];
  const button = (name, area=page) => area.getByRole('button', {name, exact:true});
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', r => {
    if (r.method() === 'POST' && /\/api\/(?:jobs|jobs3d|(?:ports|ports3d|taper)\/jobs)$/.test(r.url())) submissions.push(r.url());
  });
  async function json(url) {
    const response = await page.request.get(base+url);
    assert.equal(response.status(),200,url);
    return response.json();
  }
  async function download(locator, name) {
    const pending = page.waitForEvent('download');
    await locator.click();
    const file = path.join(output,name);
    await (await pending).saveAs(file);
    assert.ok(fs.statSync(file).size > 100, name);
    if (name.endsWith('.zip')) assert.equal(fs.readFileSync(file).subarray(0,2).toString(),'PK');
    return file;
  }
  try {
    await page.goto(base);
    await library.locator(':scope > summary').click();
    let design;
    if (fromDesign) {
      design = JSON.parse(fs.readFileSync(fromDesign,'utf8'));
      await page.getByLabel('Import geometry file',{exact:true}).setInputFiles(fromDesign);
    } else {
      await button('Choose component Linear taper',library).click();
      await library.getByLabel('Component Length',{exact:true}).fill('3');
      await library.getByLabel('Component Input width',{exact:true}).fill('0.5');
      await library.getByLabel('Component Output width',{exact:true}).fill('1.15');
      await library.getByLabel('Component Finite core height',{exact:true}).fill('0.3');
      const built = page.waitForResponse(r=>r.url().endsWith('/api/library/build') && r.request().method()==='POST');
      await button('Open as new design',library).click();
      const response = await built;
      assert.equal(response.status(),200,await response.text());
      design = (await response.json()).geometry;
    }
    assert.equal(design.components[0].parameters.length_um,3);
    assert.equal(design.components[0].parameters.height_um,.3);
    await button('Set up taper simulation',library).click();
    await panel.getByText('3D study validated',{exact:true}).waitFor();
    assert.ok(await run.isDisabled(), 'Explicit target required.');
    await panel.getByLabel('3D Resolution',{exact:true}).fill('16');
    await panel.getByLabel('3D Duration',{exact:true}).fill('40');
    await panel.getByText('3D study validated',{exact:true}).waitFor();
    const actualDesign = JSON.parse(fs.readFileSync(await download(button('Export geometry'),'taper-geometry.json'),'utf8'));
    assert.deepEqual(actualDesign,design);
    const choice = panel.locator(`input[name="study3d-target"][value="${target}"]`);
    await choice.waitFor();
    if (await choice.isDisabled()) throw new Error(`Requested ${target} unavailable: ${await choice.locator('..').innerText()}`);
    await choice.check();
    await page.waitForFunction(()=>!document.querySelector('.study3d-run-buttons button').disabled);
    const draftPath = await download(button('Export study JSON',panel),'taper-study.json');
    const config = JSON.parse(fs.readFileSync(draftPath,'utf8'));
    assert.equal(config.schema_version,5);
    assert.equal(config.study_type,'taper-3d');
    assert.equal(`${config.solver}:${config.device}`,target);
    assert.deepEqual(config.geometry,design);
    assert.equal(config.duration_fs,40);
    const validated = await page.request.post(base+'/api/taper/validate',{data:config});
    assert.equal(validated.status(),200,await validated.text());
    const layout = (await validated.json()).physical_layout;
    assert.equal(layout.width_out,1.15);
    const geometrySummary = await panel.getByLabel('Validated taper simulation layout',{exact:true}).innerText();
    assert.ok(geometrySummary.includes(layout.domain.x.toFixed(3)));
    assert.ok(geometrySummary.includes(layout.source_x.toFixed(3)));
    await download(button('Export Python + notebook',panel),'taper-study-python-notebook.zip');
    checks.push('User-selected solver/device, exact edited geometry and visible physical cell/source; config and Python/notebook downloads.');
    let jobId = null, actualDevice = null, deviceName = null, observedWidths = null, volumeShape = null;
    if (!preflight) {
      const pending = page.waitForResponse(r=>r.url().endsWith('/api/taper/jobs') && r.request().method()==='POST');
      await run.click();
      const response = await pending;
      assert.equal(response.status(),202,await response.text());
      const created = await response.json();
      jobId = created.id;
      console.log(JSON.stringify({submitted:jobId,target}));
      assert.deepEqual(created.config,config);
      await panel.locator('.status-tag.completed,.status-tag.failed,.status-tag.cancelled').waitFor({timeout:330000});
      const job = await json(`/api/jobs/${jobId}`);
      assert.equal(job.status,'completed',job.error || job.logs);
      actualDevice = job.result.actual_device;
      deviceName = job.result.device_name;
      if (target.endsWith(':gpu')) assert.match(actualDevice,/cuda|gpu/i);
      else assert.match(actualDevice,/cpu/i);
      assert.deepEqual(job.config,config);
      const volume = await json(`/api/jobs/${jobId}/volume`);
      volumeShape = volume.shape;
      for (const [plane,axis] of [['xy',volume.z_um],['xz',volume.y_um],['yz',volume.x_um]]) {
        await panel.getByLabel('3D slice plane',{exact:true}).selectOption(plane);
        for (const component of ['Ex','Ey','Ez']) {
          await panel.getByLabel('3D field component',{exact:true}).selectOption(component);
          const index = Math.floor(axis.length/2);
          const data = await json(`/api/jobs/${jobId}/slice?plane=${plane}&index=${index}&component=${component}`);
          const peak = await inspectField(panel,data,volume.field_peaks_by_component[component]);
          if (component === 'Ey') assert.ok(peak > 0);
        }
      }
      await panel.getByLabel('3D slice plane',{exact:true}).selectOption('xy');
      await button('Material grid',panel).click();
      await panel.locator('canvas[aria-label^="Actual 3D relative permittivity XY slice"]').waitFor();
      const data = await json(`/api/jobs/${jobId}/slice?plane=xy&index=${Math.floor(volume.z_um.length/2)}&component=Ez`);
      const threshold = (config.cladding_index**2 + layout.core_index**2)/2;
      observedWidths = [layout.x_in+.3, layout.x_out-.3].map(x => {
        const column = nearest(data.u_um,x);
        return data.epsilon.filter(row=>row[column]>threshold).length;
      });
      assert.ok(observedWidths[1] > observedWidths[0],`Expected widening material cross-section: ${observedWidths}`);
      const epsMax = Math.max(...data.epsilon.flat());
      const epsMin = Math.min(...data.epsilon.flat());
      const x = nearest(data.u_um,0), y = nearest(data.v_um,0);
      const materialPixel = await panel.locator('.slice3d-canvas').evaluate((c,p)=>[...c.getContext('2d').getImageData(p.x,c.height-1-p.y,1,1).data],{x,y});
      const f = (data.epsilon[y][x]-epsMin)/(epsMax-epsMin);
      const color = [22+67*f,37+182*f,54+143*f];
      assert.ok(color.every((value,i)=>Math.abs(value-materialPixel[i])<=1));
      const savedConfig = await download(button('Save run configuration',panel),'taper-run-config.json');
      assert.deepEqual(JSON.parse(fs.readFileSync(savedConfig,'utf8')),config);
      await download(panel.getByRole('link',{name:'Export 3D results .zip',exact:true}),'taper-results.zip');
      await download(panel.getByRole('link',{name:'Export taper run Python + notebook',exact:true}),'taper-run-python-notebook.zip');
      await panel.screenshot({path:path.join(output,'taper-result.png')});
      await button('Material grid',panel).click();
      await panel.getByLabel('3D field component',{exact:true}).selectOption('Ey');
      await panel.locator('canvas[aria-label^="Actual 3D Ey XY slice"]').waitFor();
      await panel.locator('.volume3d-results').screenshot({path:path.join(output,'taper-field.png')});
      checks.push('Real worker on chosen target; actual nine component/plane slices agree with rendered samples; material grid widens between taper faces; immutable result and replay exports.');
    }
    // Lock the access-guide model before adding an incompatible extra solid.
    await panel.getByLabel('3D excitation model',{exact:true}).selectOption('auto');
    await button('Choose component Strip waveguide',library).click();
    await library.getByLabel('Component Length',{exact:true}).fill('1.5');
    await library.getByLabel('Component Finite core height',{exact:true}).fill('0.3');
    const translated=page.waitForResponse(r=>r.url().endsWith('/api/layout/validate')&&r.request().method()==='POST'&&r.request().postDataJSON().geometry.components?.length===2);
    await button('Add to layout',library).click();
    const transition=await translated;assert.equal(transition.status(),200,await transition.text());
    assert.equal(transition.request().postDataJSON().study_type,'layout-3d');
    assert.ok(await run.isDisabled(),'Target must be selected again for the new physical model.');
    checks.push('Adding a strip to a taper switches auto selection to the finite dipole layout and requires an explicit target for that model.');
    await panel.getByLabel('3D excitation model').selectOption('taper-3d');
    await page.getByLabel('New shape',{exact:true}).selectOption('box');
    await button('+ Add').click();
    await panel.getByText('Study needs attention',{exact:true}).waitFor();
    assert.ok(await run.isDisabled());
    if (jobId) {
      await panel.getByText('The editor differs from this run.',{exact:false}).waitFor();
      assert.deepEqual((await json(`/api/jobs/${jobId}`)).config,config);
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
    assert.deepEqual(errors,[]);
    assert.equal(submissions.length,preflight ? 0 : 1);
    checks.push('A second solid prevents submission; small-screen layout fits; no browser exception or unexpected job.');
    const record = {passed:true,checked_at:new Date().toISOString(),target,preflight,new_solver_jobs:submissions.length,
      run_id:jobId,actual_device:actualDevice,device_name:deviceName,shape:volumeShape,observed_core_width_cells:observedWidths,
      parameters:design.components[0].parameters,checks,browser_errors:errors,browser:await browser.version()};
    fs.writeFileSync(path.join(output,'taper-browser.json'),JSON.stringify(record,null,2)+'\n');
    console.log(JSON.stringify(record));
  } catch (error) {
    await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});
    throw error;
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
