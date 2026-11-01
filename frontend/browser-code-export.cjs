/** Real editor exports for every supported schema; no solver submissions.
 * Uses the running local API. A deliberate transport error tests visible retry.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/code-export');
  fs.mkdirSync(output, { recursive: true });
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, acceptDownloads: true });
  page.setDefaultTimeout(90000);
  const errors = [], exports = []; let jobs = 0;
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (r.method() === 'POST' && /\/api\/(?:jobs3d|jobs|ports\/jobs|ports3d\/jobs)$/.test(r.url())) jobs++; });
  async function download(button, kind, name) {
    const posted = page.waitForRequest(r => r.method() === 'POST' && r.url().endsWith(`/api/reproduce/${kind}`));
    const ready = page.waitForEvent('download'); await button.click();
    const request = await posted, file = await ready;
    const destination = path.join(output, `${name}.zip`); await file.saveAs(destination);
    const bytes = fs.readFileSync(destination); assert.equal(bytes.subarray(0, 2).toString(), 'PK'); assert.ok(bytes.length > 1000);
    exports.push({ kind, file: `${name}.zip`, config: request.postDataJSON() });
  }
  try {
    await page.goto(base);
    const geometryButton = page.getByRole('button', { name: 'Export geometry Python + notebook', exact: true });
    await download(geometryButton, 'geometry', 'design');
    assert.ok(exports[0].config.objects.some(o => o.kind === 'ring'), 'A general geometry draft must export intact.');
    const field3d = page.getByRole('region', { name: 'Finite-height 3D simulation', exact: true });
    const fieldCode = field3d.getByRole('button', { name: 'Export Python + notebook', exact: true });
    assert.ok(await fieldCode.isDisabled(), 'Unsupported simulation geometry must not export as an executable study.');
    await field3d.getByRole('button', { name: 'Open runnable 3D waveguide', exact: true }).click();
    await field3d.getByText('3D study validated', { exact: true }).waitFor();
    assert.ok(await fieldCode.isDisabled(), 'Export requires an explicit execution target.');
    await field3d.locator('input[name="study3d-target"][value="meep:cpu"]').check();
    await download(fieldCode, 'study', 'field3d');
    assert.equal(exports.at(-1).config.schema_version, 2);

    await page.getByRole('button', { name: '3D ports & S-matrix', exact: true }).click();
    const modal = page.getByRole('region', { name: 'Finite-height 3D modal ports', exact: true });
    await modal.getByRole('button', { name: 'Open 3D gap example', exact: true }).click();
    await modal.getByText('3D modal study validated', { exact: true }).waitFor();
    await modal.locator('input[name="modal-target"][value="fdtdx:gpu"]').check();
    await modal.getByLabel('Modal Gap length (µm)', { exact: true }).fill('.375');
    await modal.getByText('3D modal study validated', { exact: true }).waitFor();
    await download(modal.getByRole('button', { name: 'Export Python + notebook', exact: true }), 'study', 'modal3d');
    assert.equal(exports.at(-1).config.schema_version, 4); assert.equal(exports.at(-1).config.gap_um, .375);
    assert.equal(exports.at(-1).config.device, 'gpu');
    await modal.locator('.modal-preview>summary').click();
    const canvas = modal.locator('.modal-preview canvas'); await canvas.waitFor();
    const before = await canvas.getAttribute('data-camera');
    await modal.locator('.modal-preview').getByRole('button', { name: 'Zoom in', exact: true }).click();
    await page.waitForFunction(previous => document.querySelector('.modal-preview canvas')?.dataset.camera !== previous, before);
    const zoomed = await canvas.getAttribute('data-camera'); const bounds = await canvas.boundingBox();
    await page.mouse.move(bounds.x + bounds.width * .5, bounds.y + bounds.height * .5);
    await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width * .6, bounds.y + bounds.height * .6, { steps: 8 }); await page.mouse.up();
    await page.waitForFunction(previous => document.querySelector('.modal-preview canvas')?.dataset.camera !== previous, zoomed);

    await page.getByRole('button', { name: '2D simulation', exact: true }).click();
    await page.getByText('Parameters validated', { exact: false }).waitFor();
    await page.locator('label.target-card').filter({ has: page.locator('input[value="meep:cpu"]') }).click();
    const slabCode = page.locator('.execution-panel .code-export button');
    await download(slabCode, 'study', 'slab2d'); assert.equal(exports.at(-1).config.schema_version, 1);

    await page.getByRole('button', { name: 'Ports · 2D', exact: true }).click();
    const ports = page.getByRole('region', { name: '2D port measurements', exact: true });
    await ports.locator('input[value="meep:cpu"]').check();
    const portsCode = ports.getByRole('button', { name: 'Export Python + notebook', exact: true });
    await download(portsCode, 'study', 'ports2d'); assert.equal(exports.at(-1).config.schema_version, 3);
    await page.route('**/api/reproduce/study', route => route.abort('failed'));
    await portsCode.click(); await ports.getByRole('alert').filter({ hasText: 'Code export unavailable' }).waitFor();
    await page.unroute('**/api/reproduce/study'); await download(portsCode, 'study', 'ports2d-retry');
    assert.equal(await ports.locator('.code-export-error').count(), 0);
    assert.equal(jobs, 0); assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'code-export-browser.json'), JSON.stringify({ passed: true, checked_at: new Date().toISOString(), new_solver_jobs: jobs, exports,
      checks: ['Complete geometry draft and four executable schema exports', 'Explicit target and physical validation gates', 'Edited modal settings and GPU target preserved', 'Real gap-preview WebGL zoom and orbit', 'Visible transport failure and successful retry', 'No solver submission or browser exception'], browser_errors: errors }, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, exports: exports.length, new_solver_jobs: jobs }));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
