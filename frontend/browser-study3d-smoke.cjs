/** Real finite-height 3D browser journey. No solver, API, or field mocks.
 * Install Playwright separately, start the local service, and choose an installed
 * PHOTONICS_TEST_TARGET (default meep:cpu). PHOTONICS_BROWSER_CHANNEL=chrome
 * uses installed Chrome. Output defaults to test-results/study3d.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/study3d');
  fs.mkdirSync(output, { recursive: true });
  const target = process.env.PHOTONICS_TEST_TARGET || 'meep:cpu';
  const baseUrl = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const browser = await chromium.launch({ headless: true, ...(process.env.PHOTONICS_BROWSER_CHANNEL ? { channel: process.env.PHOTONICS_BROWSER_CHANNEL } : {}) });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 }, acceptDownloads: true });
  const errors = [], checks = [];
  page.on('pageerror', error => errors.push(error.message));
  const panel = page.getByRole('region', { name: 'Finite-height 3D simulation' });
  const run = panel.getByRole('button', { name: 'Run 3D simulation', exact: true });
  async function data(url) { const response = await page.request.get(new URL(url, baseUrl).href); assert.ok(response.ok(), `GET ${url} failed`); return response.json(); }
  async function waitSlice(plane, component) { await panel.locator(`canvas[aria-label^="Actual 3D ${component} ${plane.toUpperCase()} slice"]`).waitFor(); }
  async function checkPixels(slice, peak) {
    const pixel = await panel.locator('.slice3d-canvas').evaluate(element => {
      const context = element.getContext('2d'), x = Math.floor(element.width * .61), y = Math.floor(element.height * .44);
      return { x, y, width: element.width, height: element.height, rgba: [...context.getImageData(x, y, 1, 1).data] };
    });
    const value = slice.field[pixel.height - pixel.y - 1][pixel.x];
    const normal = Math.max(-1, Math.min(1, value / (peak || 1))), base = [19, 28, 41], color = normal < 0 ? [73, 141, 249] : [255, 157, 104];
    assert.deepEqual(pixel.rgba, [...base.map((channel, i) => Math.round(channel + (color[i] - channel) * Math.abs(normal))), 255], 'Canvas sample must equal its actual saved solver sample and fixed scale.');
    assert.equal(pixel.height, slice.field.length); assert.equal(pixel.width, slice.field[0].length);
  }
  try {
    await page.goto(baseUrl);
    await panel.getByText('Study needs attention', { exact: true }).waitFor();
    assert.equal(await run.isDisabled(), true, 'Unsupported ring draft must not run as a waveguide.');
    await panel.getByRole('button', { name: 'Open runnable 3D waveguide', exact: true }).click();
    await panel.getByText('3D study validated', { exact: true }).waitFor();
    assert.equal(await run.isDisabled(), true, 'A validated study must still require target selection.');
    await page.getByLabel('Size Y', { exact: true }).fill('0.875'); await page.getByLabel('Size Y', { exact: true }).press('Enter');
    await panel.getByText('3D study validated', { exact: true }).waitFor();
    await panel.locator(`input[name="study3d-target"][value="${target}"]`).check();
    await page.waitForFunction(() => ![...document.querySelectorAll('.study3d button')].find(element => element.textContent === 'Run 3D simulation').disabled);
    const start = page.waitForResponse(response => response.url().endsWith('/api/jobs3d') && response.request().method() === 'POST');
    await run.click(); const response = await start; assert.equal(response.status(), 202); const created = await response.json();
    assert.equal(created.config.geometry.objects.find(object => object.id === 'core').size.y, .875);
    assert.equal(`${created.config.solver}:${created.config.device}`, target);
    await panel.locator('.status-tag.completed').waitFor({ timeout: 300000 });
    await waitSlice('xy', 'Ey');
    const job = await data(`/api/jobs/${created.id}`), volume = await data(`/api/jobs/${created.id}/volume`);
    assert.equal(job.config.schema_version, 2); assert.equal(volume.shape.length, 3);
    if (target.endsWith(':gpu')) assert.match(JSON.stringify(job.result.actual_device), /cuda|gpu/i);
    const actualTime = volume.metadata.actual_duration_fs ?? volume.metadata.simulated_time_fs;
    assert.ok((await panel.locator('.slice3d-header').innerText()).includes(`${actualTime.toFixed(2)} fs`), 'Final snapshot must show its actual solver time.');
    checks.push('Unsupported draft rejected; exact edited geometry submitted; explicit execution target; real completed 3D worker');

    for (const plane of ['xy', 'xz', 'yz']) {
      await panel.getByLabel('3D slice plane', { exact: true }).selectOption(plane);
      const axis = plane === 'xy' ? volume.z_um : plane === 'xz' ? volume.y_um : volume.x_um;
      const index = Math.floor(axis.length / 2);
      for (const component of ['Ex', 'Ey', 'Ez']) {
        await panel.getByLabel('3D field component', { exact: true }).selectOption(component); await waitSlice(plane, component);
        const slice = await data(`/api/jobs/${created.id}/slice?plane=${plane}&index=${index}&component=${component}`);
        await checkPixels(slice, volume.field_peaks_by_component[component]);
        assert.ok((await panel.locator('.slice3d-position').innerText()).includes(`${slice.coordinate_um.toFixed(4)} µm`));
      }
    }
    await panel.getByLabel('3D field component', { exact: true }).selectOption('Ez'); await waitSlice('yz', 'Ez');
    const index = Math.max(0, Math.floor(volume.x_um.length / 2) - 2);
    await panel.getByLabel('3D slice index', { exact: true }).fill(String(index));
    await page.waitForFunction(expected => document.querySelector('.slice3d-position strong').textContent.includes(expected), `${volume.x_um[index].toFixed(4)} µm`);
    await waitSlice('yz', 'Ez');
    await panel.locator(`.slice3d-canvas[data-slice-index="${index}"]`).waitFor();
    const slice = await data(`/api/jobs/${created.id}/slice?plane=yz&index=${index}&component=Ez`);
    await checkPixels(slice, volume.field_peaks_by_component.Ez);
    await panel.getByRole('button', { name: 'Material grid', exact: true }).click();
    await panel.locator('canvas[aria-label^="Actual 3D relative permittivity YZ slice"]').waitFor();
    checks.push('Actual XY/XZ/YZ slices; per-component scale; exact API-to-canvas samples; physical slice coordinate; Ez selection; material grid');

    await page.getByLabel('Size Y', { exact: true }).fill('1'); await page.getByLabel('Size Y', { exact: true }).press('Enter');
    await panel.getByText('The editor differs from this run.', { exact: false }).waitFor();
    assert.equal((await data(`/api/jobs/${created.id}`)).config.geometry.objects.find(object => object.id === 'core').size.y, .875);
    const passport = await data(`/api/jobs/${created.id}/passport`); assert.ok(passport.checks.every(check => check.passed));
    const downloadPromise = page.waitForEvent('download'); await panel.getByRole('link', { name: 'Export 3D results .zip', exact: true }).click();
    const download = await downloadPromise; const zip = path.join(output, download.suggestedFilename()); await download.saveAs(zip); assert.ok(fs.statSync(zip).size > 10000);
    await panel.getByRole('button', { name: 'Material grid', exact: true }).click();
    await panel.getByLabel('3D slice plane', { exact: true }).selectOption('xz');
    await panel.getByLabel('3D field component', { exact: true }).selectOption('Ey'); await waitSlice('xz', 'Ey');
    await page.getByLabel('Size Y', { exact: true }).fill('0.875'); await page.getByLabel('Size Y', { exact: true }).press('Enter');
    await panel.getByText('3D study validated', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, '3d-simulation-result.png'), fullPage: true });
    await panel.screenshot({ path: path.join(output, '3d-computed-panel.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile study must not overflow horizontally.');
    assert.deepEqual(errors, []);
    checks.push('Immutable run geometry after editor changes; run passport observed checks; actual result ZIP export; responsive 390px; no browser exceptions');
    const report = { passed: true, checked_at: new Date().toISOString(), target, run_id: created.id, shape: volume.shape, actual_device: job.result.actual_device, device_name: job.result.device_name, browser: await browser.version(), checks, browser_errors: errors };
    fs.writeFileSync(path.join(output, 'study3d-browser.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
  } catch (error) { await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {}); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
