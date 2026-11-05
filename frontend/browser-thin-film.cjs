/** Exercise the material inspector and capture a portable real analytical spectrum. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/thin-film');
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true,channel:process.env.PHOTONICS_BROWSER_CHANNEL||'chrome'});
  const page = await browser.newPage({viewport:{width:1400,height:960},acceptDownloads:true});
  const errors = [];
  page.on('pageerror',e=>errors.push(e.message));
  const label = name => page.getByLabel(name,{exact:true});
  const button = name => page.getByRole('button',{name,exact:true});
  async function calculate(status) {
    const ready = page.waitForResponse(r=>new URL(r.url()).pathname==='/api/materials/film');
    await button('Calculate film spectrum').click();
    const response = await ready;
    assert.equal(response.status(),status);
    return {input:response.request().postDataJSON(),result:await response.json()};
  }
  try {
    await page.goto(process.env.PHOTONICS_URL || 'http://127.0.0.1:8000');
    await page.locator('.geometry-3d-host canvas').waitFor();
    const material = page.getByRole('region',{name:'Optical material editor',exact:true});
    await material.locator('.material-custom > summary').click();
    await label('Custom material model').selectOption('tabulated-nk');
    await label('Custom material name').fill('Measured passive film');
    await label('Material CSV samples').fill('wavelength_um,n,k\n1.35,2.1,0.12\n1.55,2,0.08\n1.75,1.9,0.04');
    await button('Save as new & assign').click();
    await label('Thickness Z').fill('0.4');
    await label('Thickness Z').press('Enter');
    await material.locator('.film-study > summary').click();
    await label('Film Start λ (µm)').fill('1.45');
    await label('Film End λ (µm)').fill('1.55');
    await label('Film Samples').fill('3');
    const {input,result} = await calculate(200);
    assert.equal(input.thickness_um,.4);
    assert.equal(input.material.model,'tabulated-nk');
    assert.ok(Math.abs(result.samples.at(-1).R-.0509883321448908)<1e-12);
    assert.ok(Math.abs(result.samples.at(-1).T-.7231774098323366)<1e-12);
    await label('Film sample').fill('2');
    const panel=page.getByRole('region',{name:'Film spectrum results'});
    await panel.getByText('5.09883%',{exact:true}).waitFor();
    const pending=page.waitForEvent('download');
    await button('Export film Python + notebook').click();
    await (await pending).saveAs(path.join(output,'planar-film.zip'));
    fs.writeFileSync(path.join(output,'gui-result.json'),JSON.stringify(result));
    await label('Film Start λ (µm)').fill('1.3');
    assert.equal(await panel.count(),0);
    const rejected=await calculate(422);
    assert.match(rejected.result.detail,/outside the material table/);
    await label('Film Start λ (µm)').fill('1.45');
    await calculate(200);
    await panel.waitFor();
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);
    const report={passed:true,checked_at:new Date().toISOString(),browser:browser.version(),widths:[1400,390],
      checks:['Assigned a tabulated n,k material, edited the real solid thickness and ran the film calculation.',
        'GUI showed the independently checked 1.55 µm power samples and hid stale results after a band edit.',
        'Out-of-table wavelengths were rejected. GUI downloaded a Python, notebook, CSV and full precision JSON archive.'],
      wavelength_count:result.samples.length,last_sample:result.samples.at(-1),browser_errors:errors};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
