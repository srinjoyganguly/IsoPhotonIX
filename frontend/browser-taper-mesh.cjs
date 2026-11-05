// Compare genuine completed jobs served by a local workbench.
// node browser-taper-mesh.cjs URL COARSE_ID FINE_ID OUTPUT_DIR [INCOMPATIBLE_COARSE_ID]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const [url, coarse, fine, output, incompatible] = process.argv.slice(2);
  assert.ok(url && coarse && fine && output, 'Supply the local URL, coarse and fine run IDs, and output directory.');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  try {
    await page.goto(url);
    await page.getByRole('button', { name: '3D ports & S-matrix', exact: true }).click();
    await page.getByText('Taper mesh sensitivity', { exact: true }).click();
    const panel = page.locator('details.modal-mesh');
    const coarseSelect = panel.getByLabel('Coarse taper mesh run');
    const fineSelect = panel.getByLabel('Fine taper mesh run');
    const compare = panel.getByRole('button', { name: 'Compare taper meshes', exact: true });
    assert.ok(await compare.isDisabled());
    if (incompatible) {
      await coarseSelect.selectOption(incompatible);
      await fineSelect.selectOption(fine);
      await compare.click();
      await panel.getByRole('alert').waitFor();
      assert.equal(await panel.locator('.modal-mesh-report').count(), 0);
    }
    await coarseSelect.selectOption(coarse);
    await fineSelect.selectOption(fine);
    const response = page.waitForResponse(r => r.url().endsWith('/api/taper-ports/compare-mesh') && r.status() === 200);
    await compare.click();
    const report = await (await response).json();
    const region = panel.getByRole('region', { name: 'Taper mesh comparison' });
    await region.waitFor();
    assert.equal(report.coarse_run_id, coarse);
    assert.equal(report.fine_run_id, fine);
    for (const [name, metrics] of Object.entries(report.entries)) {
      const text = await region.locator(`[data-mesh-entry="${name}"] td`).allTextContents();
      assert.deepEqual(text, [metrics.maximum_absolute_complex_change,
        metrics.maximum_absolute_power_change,metrics.maximum_absolute_phase_change_rad]
        .map(v => v === null ? 'Undefined' : v.toPrecision(6)));
    }
    const download = page.waitForEvent('download');
    await region.getByRole('button', { name: 'Download mesh comparison JSON' }).click();
    const file = await download;
    const filename = path.join(output, file.suggestedFilename());
    await file.saveAs(filename);
    assert.deepEqual(JSON.parse(fs.readFileSync(filename, 'utf8')), report);
    await panel.screenshot({ path: path.join(output, 'taper-mesh.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await coarseSelect.selectOption('');
    assert.equal(await region.count(), 0);
    assert.ok(await compare.isDisabled());
    assert.deepEqual(errors, []);
    const verification = { passed: true, checked_at: new Date().toISOString(), coarse_run_id: coarse, fine_run_id: fine,
      compared_bins: report.compared_bins, incompatibility_shown: Boolean(incompatible), four_entries_equal_to_saved_response: true,
      downloaded_json_equals_response: true, stale_result_cleared: true, fits_390px: true, browser_errors: errors };
    fs.writeFileSync(path.join(output, 'taper-mesh-browser.json'), JSON.stringify(verification, null, 2) + '\n');
    console.log(JSON.stringify(verification));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
