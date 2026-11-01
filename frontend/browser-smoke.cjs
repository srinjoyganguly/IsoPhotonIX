/** Optional browser verification. Requires playwright 1.62.1 and Chromium/Chrome.
 * PHOTONICS_URL defaults to http://127.0.0.1:8000.
 * PHOTONICS_TEST_TARGET defaults to fdtdx:gpu; select an actually installed target.
 * PHOTONICS_BROWSER_CHANNEL=chrome uses an installed Google Chrome.
 * PHOTONICS_ARTIFACT_DIR controls screenshots/downloads; defaults to test-results/.
 * Run against the real local backend: this test does not mock solver data.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, ...(process.env.PHOTONICS_BROWSER_CHANNEL ? {channel: process.env.PHOTONICS_BROWSER_CHANNEL} : {}) });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  try {
    await page.goto(process.env.PHOTONICS_URL || 'http://127.0.0.1:8000');
    await page.getByText('Parameters validated', {exact:false}).waitFor();
    const run = page.getByRole('button', { name: 'Run simulation' });
    assert.equal(await run.isDisabled(), true, 'Run must require explicit target selection.');
    await page.screenshot({ path: path.join(output, '01-ready.png'), fullPage:true });
    await page.getByLabel('Core index', {exact:true}).fill('0');
    await page.locator('.validation.invalid').waitFor();
    assert.equal(await run.isDisabled(), true, 'Invalid physics must not run.');
    await page.getByLabel('Core index', {exact:true}).fill('2');
    await page.getByLabel('Core width', {exact:true}).fill('0.65');
    await page.getByText('Parameters validated', {exact:false}).waitFor();
    const target = process.env.PHOTONICS_TEST_TARGET || 'fdtdx:gpu';
    await page.locator(`input[name="execution-target"][value="${target}"]`).check({force:true});
    await run.click();
    await page.locator('.status-tag.completed').waitFor({timeout:180000});
    await page.locator('.field-canvas').waitFor();
    assert.equal(await page.getByRole('tab', {name:'Ez field'}).getAttribute('aria-selected'), 'true');
    await page.locator('.field-canvas').hover();
    await page.screenshot({path:path.join(output,'02-real-fields.png'),fullPage:true});
    await page.getByRole('tab', {name:'Material grid'}).click();
    await page.screenshot({path:path.join(output,'03-material-grid.png'),fullPage:true});
    await page.getByRole('tab', {name:'Ez field'}).click();
    const downloadEvent=page.waitForEvent('download');
    await page.getByRole('link', {name:'Export results .zip'}).click();
    const download=await downloadEvent;
    const zipPath=path.join(output,download.suggestedFilename());
    await download.saveAs(zipPath);
    assert.ok(fs.statSync(zipPath).size>1000,'Actual numerical export must be nonempty.');
    await page.getByLabel('Core width',{exact:true}).fill('0.7');
    await page.getByText('The editor has changed.',{exact:false}).waitFor();
    assert.ok((await page.locator('.run-facts').textContent()).includes('0.65 µm'),'Editing must preserve the run snapshot.');
    await page.getByRole('button',{name:'Open bundled example'}).click();
    assert.equal(await run.isDisabled(),true,'Opening an example must require explicit target confirmation.');
    assert.deepEqual(errors,[],'Browser must not produce uncaught exceptions.');
    const report={passed:true,target,checks:['explicit target required','physical validation','edited configuration run','actual field canvas','discretized material view','ZIP export','immutable run configuration','example reset clears target'],browser_errors:errors};
    fs.writeFileSync(path.join(output,'browser-smoke.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
