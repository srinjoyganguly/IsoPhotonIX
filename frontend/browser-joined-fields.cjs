/** Exercise a two-part design through the visible port, editing and solver controls. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/joined-fields');
  const target = process.env.PHOTONICS_TEST_TARGET || 'meep:cpu';
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, acceptDownloads: true });
  page.setDefaultTimeout(45000);
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  const saved = () => page.evaluate(() => JSON.parse(localStorage.getItem('isophotonix.geometry.v1')));
  async function post(url, data) {
    const response = await page.request.post(base + url, { data });
    assert.equal(response.status(), 200, await response.text());
    return response.json();
  }
  async function run(label, geometry) {
    const panel = page.getByRole('region', { name: 'Finite-height 3D simulation', exact: true });
    await panel.getByLabel('3D Resolution', { exact: true }).fill('16');
    await panel.getByLabel('3D Duration', { exact: true }).fill('24');
    for (const [axis, value] of Object.entries({ x: '-1.5', y: '0', z: '0' })) {
      await panel.getByLabel('Dipole ' + axis + ' (µm)', { exact: true }).fill(value);
    }
    await panel.locator('input[value="' + target + '"]').check();
    const pending = page.waitForResponse(r => r.url().endsWith('/api/layout/jobs') && r.request().method() === 'POST');
    await panel.getByRole('button', { name: 'Run 3D simulation', exact: true }).click();
    const response = await pending;
    assert.equal(response.status(), 202, await response.text());
    const job = await response.json();
    assert.deepEqual(job.config.geometry, geometry);
    const deadline = Date.now() + 300000;
    let value;
    do {
      const status = await page.request.get(base + '/api/jobs/' + job.id);
      assert.equal(status.status(), 200);
      value = await status.json();
      if (!['queued', 'running'].includes(value.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
    } while (Date.now() < deadline);
    assert.equal(value.status, 'completed', value.error);
    assert.equal(value.result.requested_solver + ':' + value.result.requested_device, target);
    await panel.locator('.status-tag.completed').waitFor();
    await panel.getByLabel('3D field component', { exact: true }).selectOption('Ey');
    await panel.locator('canvas[aria-label^="Actual 3D Ey"]').first().waitFor();
    return { label, id: job.id, actual_device: value.result.actual_device, device_name: value.result.device_name,
      design_epsilon_sha256: value.result.geometry_translation.epsilon_xyz_sha256 };
  }
  try {
    await page.goto(base);
    const dimensions = { length_um: 2, width_um: .5, height_um: .35 };
    const first = await post('/api/library/build', { component_id: 'strip', parameters: dimensions, placement: { x: -1 } });
    first.geometry.domain = { x: 7, y: 4.5, z: 3 };
    const two = await post('/api/library/place', { geometry: first.geometry, component_id: 'directional_coupler',
      parameters: { ...dimensions, gap_um: .25 }, placement: { x: .5, y: -.1, rotation_deg: 90 } });
    await page.getByLabel('Import geometry file').setInputFiles({ name: 'layout.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(two.geometry)) });
    const library = page.locator('.component-library');
    await library.locator(':scope>summary').click();
    const align = library.locator('.port-alignment');
    await align.locator('summary').click();
    await align.getByLabel('Moving component', { exact: true }).selectOption(two.component.id);
    await align.getByLabel('Moving port', { exact: true }).selectOption('lower_in');
    await align.getByLabel('Fixed port', { exact: true }).selectOption('out');
    const receipt = page.waitForResponse(r => r.url().endsWith('/api/library/align'));
    await align.getByRole('button', { name: 'Align ports' }).click();
    assert.equal((await receipt).status(), 200);
    await align.getByRole('status').waitFor();
    const joined = await saved();
    const moving = joined.components.find(c => c.id === two.component.id);
    assert.equal(moving.placement.rotation_deg, 0);
    const port = moving.ports.find(p => p.name === 'lower_in');
    assert.ok(Math.hypot(port.x, port.y, port.z) < 1e-10 && port.angle_deg === 180);
    await library.locator(':scope>summary').click();
    const runs = [await run('joined', joined)];
    await page.getByRole('button', { name: 'Delete component', exact: true }).click();
    const deleted = await saved();
    assert.deepEqual(deleted.components, [joined.components[0]]);
    assert.equal(deleted.objects.length, 1);
    runs.push(await run('deleted', deleted));
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.deepEqual(await saved(), joined);
    runs.push(await run('restored', joined));
    assert.notEqual(runs[0].design_epsilon_sha256, runs[1].design_epsilon_sha256);
    assert.equal(runs[0].design_epsilon_sha256, runs[2].design_epsilon_sha256);
    for (const g of [joined, deleted]) {
      const id = g.components.length === 2 ? 'joined' : 'deleted';
      fs.writeFileSync(path.join(output, id + '.json'), JSON.stringify(g));
    }
    assert.deepEqual(errors, []);
    const report = { passed: true, checked_at: new Date().toISOString(), target, runs, browser_errors: errors };
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
