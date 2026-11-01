/** Read-only slice race regression against an existing real 3D run.
 * Set PHOTONICS_RUN_ID. Optional PHOTONICS_URL, PHOTONICS_BROWSER_CHANNEL,
 * PHOTONICS_ARTIFACT_DIR. Actual responses are delayed, never replaced by fake data.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
(async () => {
  const id = process.env.PHOTONICS_RUN_ID;
  assert.ok(id, 'Set PHOTONICS_RUN_ID to a completed real 3D run.');
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/slice3d-regression'); fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, ...(process.env.PHOTONICS_BROWSER_CHANNEL ? { channel: process.env.PHOTONICS_BROWSER_CHANNEL } : {}) });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 }, acceptDownloads: true });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const panel = page.getByRole('region', { name: 'Finite-height 3D simulation' });
  try {
    await page.goto(base);
    await panel.getByLabel('Recent 3D run', { exact: true }).selectOption(id);
    await panel.locator('.slice3d-canvas[data-plane="xy"][data-component="Ey"]').waitFor();
    await panel.getByText('Exploratory grid · Commercial solver agreement: not validated', { exact: true }).waitFor();
    await page.route(`**/api/jobs/${id}/slice?**`, async route => {
      const response = await route.fetch();
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({ response });
    });
    await panel.getByLabel('3D field component', { exact: true }).selectOption('Ex');
    assert.equal(await panel.locator('.slice3d-canvas').count(), 0, 'Old Ey pixels must disappear before Ex loads.');
    await panel.getByLabel('3D slice plane', { exact: true }).selectOption('xz');
    await panel.getByLabel('3D field component', { exact: true }).selectOption('Ez');
    await panel.getByLabel('3D slice plane', { exact: true }).selectOption('yz');
    await panel.locator('.slice3d-canvas[data-plane="yz"][data-component="Ez"]').waitFor();
    await page.waitForTimeout(600);
    const canvas = panel.locator('.slice3d-canvas');
    assert.equal(await canvas.getAttribute('data-plane'), 'yz'); assert.equal(await canvas.getAttribute('data-component'), 'Ez');
    const index = Number(await canvas.getAttribute('data-slice-index'));
    const sliceResponse = await page.request.get(`${base}/api/jobs/${id}/slice?plane=yz&index=${index}&component=Ez`); assert.ok(sliceResponse.ok()); const slice = await sliceResponse.json();
    const volumeResponse = await page.request.get(`${base}/api/jobs/${id}/volume`); assert.ok(volumeResponse.ok()); const volume = await volumeResponse.json();
    const pixels = await canvas.evaluate(element => { const x = Math.floor(element.width * .61), y = Math.floor(element.height * .44); return { x, y, height: element.height, rgba: [...element.getContext('2d').getImageData(x, y, 1, 1).data] }; });
    const value = slice.field[pixels.height - pixels.y - 1][pixels.x], normal = Math.max(-1, Math.min(1, value / (volume.field_peaks_by_component.Ez || 1)));
    const baseColor = [19, 28, 41], color = normal < 0 ? [73, 141, 249] : [255, 157, 104];
    assert.deepEqual(pixels.rgba, [...baseColor.map((channel, i) => Math.round(channel + (color[i] - channel) * Math.abs(normal))), 255]);
    let comparison;
    const referenceId = process.env.PHOTONICS_REFERENCE_RUN_ID;
    if (referenceId) {
      await panel.locator('.study3d-comparison > summary').click();
      await panel.getByLabel('3D comparison reference', { exact: true }).selectOption(referenceId);
      const comparisonResponse = page.waitForResponse(response => response.url().endsWith('/api/compare3d') && response.request().method() === 'POST');
      await panel.getByRole('button', { name: 'Compare saved runs', exact: true }).click();
      const response = await comparisonResponse; assert.equal(response.status(), 200); comparison = await response.json();
      await panel.locator('.study3d-comparison-result').waitFor();
      assert.equal(comparison.reference_run_id, referenceId); assert.equal(comparison.candidate_run_id, id);
      assert.equal(comparison.kind, 'same-study-repeatability'); assert.equal(comparison.commercial_agreement, 'not_validated');
      assert.equal(await panel.locator('.comparison-overall').innerText(), comparison.overall_relative_l2.toExponential(5));
      for (const component of ['Ex', 'Ey', 'Ez']) {
        const row = panel.locator('.study3d-comparison tbody tr').filter({ has: page.getByRole('rowheader', { name: component, exact: true }) });
        assert.ok((await row.innerText()).includes(comparison.components[component].relative_l2.toExponential(5)));
        assert.ok((await row.innerText()).includes(comparison.components[component].max_abs_error.toExponential(5)));
      }
      const downloadPromise = page.waitForEvent('download'); await panel.getByRole('button', { name: 'Export comparison JSON', exact: true }).click();
      const download = await downloadPromise, file = path.join(output, download.suggestedFilename()); await download.saveAs(file);
      assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), comparison);
      await panel.locator('.study3d-comparison').screenshot({ path: path.join(output, 'same-study-comparison.png') });
      await panel.getByLabel('3D slice plane', { exact: true }).selectOption('xz');
      await panel.getByLabel('3D field component', { exact: true }).selectOption('Ey');
      await panel.locator('.slice3d-canvas[data-plane="xz"][data-component="Ey"]').waitFor();
      // Restore the actual saved design/settings only for the final screenshot;
      // explicit target selection remains a separate simulated user action.
      const jobResponse = await page.request.get(`${base}/api/jobs/${id}`); assert.ok(jobResponse.ok()); const saved = (await jobResponse.json()).config;
      await page.getByLabel('Import geometry file', { exact: true }).setInputFiles({ name: 'saved-geometry.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(saved.geometry)) });
      for (const [key, label] of [['wavelength_um', 'Wavelength'], ['cladding_index', 'Cladding index'], ['resolution', 'Resolution'], ['duration_fs', 'Duration'], ['pml_um', 'PML thickness']]) await panel.getByLabel(`3D ${label}`, { exact: true }).fill(String(saved[key]));
      await panel.locator(`input[name="study3d-target"][value="${saved.solver}:${saved.device}"]`).check();
      await panel.getByText('3D study validated', { exact: true }).waitFor();
      assert.equal(await panel.locator('.snapshot-note').count(), 0, 'Restored editor must match the immutable saved study.');
      await panel.screenshot({ path: path.join(output, '3d-comparison-panel.png') });
    }
    assert.deepEqual(errors, []);
    const report = { passed: true, checked_at: new Date().toISOString(), run_id: id, checks: ['Visible commercial-validation status', 'Old samples hidden immediately on component change', 'Rapid plane/component changes discard stale delayed responses', 'Final canvas sample matches the real saved Ez YZ slice and volume scale', ...(comparison ? ['Same-study CPU/GPU numerical errors match the comparison API', 'Comparison export equals the exact API result and retains unvalidated commercial status'] : [])], ...(comparison ? { comparison: { reference_run_id: referenceId, candidate_run_id: id, overall_relative_l2: comparison.overall_relative_l2 } } : {}), browser_errors: errors, new_solver_run: false, transport: 'Unmodified actual backend responses delayed 500 ms' };
    fs.writeFileSync(path.join(output, 'slice3d-regression.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
