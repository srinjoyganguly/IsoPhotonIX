/** Actual WebGL and editor interaction checks; no mocked solver or rendering data.
 * Requires Playwright and Chrome/Chromium. Start the local application first.
 * PHOTONICS_URL, PHOTONICS_BROWSER_CHANNEL, PHOTONICS_ARTIFACT_DIR are optional.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/geometry');
  fs.mkdirSync(output, { recursive: true });
  const launch = { headless: true, ...(process.env.PHOTONICS_BROWSER_CHANNEL ? { channel: process.env.PHOTONICS_BROWSER_CHANNEL } : {}) };
  const browser = await chromium.launch(launch);
  const context = await browser.newContext({ viewport: { width: 1600, height: 1050 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [], checks = [], requests = [];
  const url = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));
  const canvas = page.locator('.geometry-3d-host canvas');
  const button = name => page.getByRole('button', { name, exact: true });
  const input = name => page.getByLabel(name, { exact: true });
  const storageKey = 'photonics-workbench.geometry.v1';
  const saved = () => page.evaluate(key => JSON.parse(localStorage.getItem(key)), storageKey);
  async function edit(name, value) { await input(name).fill(String(value)); await input(name).press('Enter'); }
  async function select(name) { await page.locator('.geo-object-select').filter({ hasText: name }).click(); }
  async function camera() { return (await canvas.getAttribute('data-camera')).split(',').map(Number); }
  async function importJson(value, name = 'design.json') {
    await input('Import geometry file').setInputFiles({ name, mimeType: 'application/json', buffer: Buffer.from(typeof value === 'string' ? value : JSON.stringify(value)) });
  }
  try {
    await page.goto(url);
    await canvas.waitFor();
    await page.waitForFunction(() => !!document.querySelector('.geometry-3d-host canvas')?.dataset.camera);
    assert.equal(await page.locator('.geometry-3d-error').count(), 0);
    const initialCamera = await camera();
    const box = await canvas.boundingBox(); assert.ok(box);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 100, box.y + box.height / 2 + 45, { steps: 12 }); await page.mouse.up();
    const orbit = await camera(); assert.notDeepEqual(orbit, initialCamera, 'Drag must orbit the actual camera.');
    await page.mouse.wheel(0, -250);
    await page.waitForFunction(previous => document.querySelector('.geometry-3d-host canvas').dataset.camera !== previous, orbit.map(v => v.toFixed(4)).join(','));
    const zoomed = await camera();
    assert.ok(Math.hypot(...zoomed.slice(0, 3)) < Math.hypot(...orbit.slice(0, 3)), 'Wheel must zoom in.');
    await page.mouse.down({ button: 'right' }); await page.mouse.move(box.x + box.width / 2 + 145, box.y + box.height / 2 + 45, { steps: 8 }); await page.mouse.up({ button: 'right' });
    assert.notDeepEqual((await camera()).slice(3), [0, 0, 0], 'Right-drag must pan the target.');
    await button('Isometric view').click(); assert.deepEqual(await camera(), initialCamera);
    await button('Zoom in').click(); assert.notDeepEqual(await camera(), initialCamera);
    await button('Zoom out').click();
    for (const name of ['Top view (XY)', 'Front view (XZ)', 'Right view (YZ)', 'Isometric view']) {
      await button(name).click(); assert.equal(await button(name).getAttribute('aria-pressed'), 'true');
    }
    await button('Grid').click(); assert.equal(await button('Grid').getAttribute('aria-pressed'), 'false');
    await button('Domain').click(); assert.equal(await button('Domain').getAttribute('aria-pressed'), 'false');
    await button('Grid').click(); await button('Domain').click();
    checks.push('WebGL render; pointer orbit/wheel zoom/right-drag pan; camera presets and guide toggles');

    // Pick the bus in an actual top-view projection, independently of the scene list.
    await button('Top view (XY)').click();
    const top = await camera(), topBox = await canvas.boundingBox();
    const pixelsPerUm = topBox.height / (2 * Math.tan(19 * Math.PI / 180) * top[2]);
    await canvas.click({ position: { x: topBox.width / 2, y: topBox.height / 2 + 2.05 * pixelsPerUm } });
    assert.equal(await input('Object name').inputValue(), 'Bus waveguide', 'Raycast must select the actual clicked solid.');
    await select('Ring core');
    await edit('Outer radius', 1.8);
    assert.equal((await saved()).objects.find(object => object.id === 'ring').radius, 1.8);
    await edit('Outer radius', .5);
    await page.getByRole('alert').filter({ hasText: 'ring inner radius' }).waitFor();
    assert.equal((await saved()).objects.find(object => object.id === 'ring').radius, 1.8, 'Invalid ring edit must retain the previous valid draft.');
    await button('Dismiss geometry error').click();
    await input('Outer radius').fill('2.2'); await input('Outer radius').press('Escape');
    assert.equal(await input('Outer radius').inputValue(), '1.8', 'Escape must cancel the uncommitted edit.');
    await button('Undo').click(); assert.equal(await input('Outer radius').inputValue(), '1.6');
    await button('Redo').click(); assert.equal(await input('Outer radius').inputValue(), '1.8');
    await edit('Object name', 'Ring with spaces'); assert.equal(await input('Object name').inputValue(), 'Ring with spaces');
    checks.push('Raycast selection; validated dimensions; invalid edit retention; Escape cancel; undo/redo; editable names');

    await button('Hide Ring with spaces').click(); assert.equal(await canvas.getAttribute('data-visible-objects'), '2');
    await button('Show Ring with spaces').click(); assert.equal(await canvas.getAttribute('data-visible-objects'), '3');
    await button('Duplicate').click(); assert.equal((await saved()).objects.length, 4);
    await edit('Position X', 2); await button('Delete object').click(); assert.equal((await saved()).objects.length, 3);
    await button('Undo').click(); assert.equal((await saved()).objects.length, 4);
    await page.getByRole('button', { name: 'Directional coupler Two guides with an editable gap', exact: true }).click();
    assert.equal((await saved()).objects.length, 3);
    await page.getByRole('button', { name: 'Strip waveguide A finite dielectric guide', exact: true }).click();
    assert.equal((await saved()).objects.length, 2);
    await input('New shape').selectOption('cylinder'); await button('+ Add').click();
    await edit('Outer radius', .8); await edit('Position X', 2);
    assert.equal((await saved()).objects.at(-1).kind, 'cylinder');
    await input('New shape').selectOption('box'); await button('+ Add').click();
    await edit('Rotation Z', 45); assert.equal((await saved()).objects.at(-1).rotation_deg, 45);
    await edit('Position X', 100);
    await page.getByRole('alert').filter({ hasText: 'outside the domain' }).waitFor();
    assert.equal((await saved()).objects.at(-1).position.x, 0);
    await button('Dismiss geometry error').click();
    checks.push('Visibility; duplicate/delete/undo; strip and coupler templates; cylinder and box creation; transform containment');

    const before = await saved();
    const downloadPromise = page.waitForEvent('download'); await button('Export geometry').click();
    const download = await downloadPromise, file = path.join(output, 'geometry-roundtrip.json'); await download.saveAs(file);
    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), before);
    await importJson('{"schema":"bad"}');
    await page.getByRole('alert').filter({ hasText: 'Import failed' }).waitFor(); assert.deepEqual(await saved(), before);
    await button('Dismiss geometry error').click();
    await importJson({ ...before, name: 'Imported local geometry', mode: 'slab-preview' });
    assert.equal((await saved()).mode, 'draft-3d', 'Imported labels cannot confer slab provenance.');
    await page.reload(); await canvas.waitFor();
    assert.equal(await input('Design name').inputValue(), 'Imported local geometry');
    assert.deepEqual((await saved()).objects, before.objects);
    checks.push('JSON export/roundtrip; malformed import retention; imported preview treated as draft; browser reload recovery');

    // A second tab cannot silently replace this tab's next autosave.
    const other = await context.newPage(); await other.goto(url);
    await other.getByLabel('Design name', { exact: true }).fill('Second tab design');
    await other.getByLabel('Design name', { exact: true }).press('Enter');
    await page.getByText('Another tab changed the saved draft.', { exact: false }).waitFor();
    await edit('Design name', 'First tab kept in memory');
    assert.equal((await saved()).name, 'Second tab design');
    await button('Save this draft').click(); assert.equal((await saved()).name, 'First tab kept in memory');
    await other.close(); checks.push('Cross-tab draft conflict pauses autosave until explicit save');

    await button('Preview current 2D slab').click();
    await page.getByText('Extruded view of the 2D slab', { exact: true }).waitFor();
    assert.equal((await saved()).mode, 'slab-preview');
    assert.equal(await page.getByRole('button', { name: 'Run simulation', exact: true }).isVisible(), false);
    assert.equal(await page.getByRole('button', { name: 'Run 3D simulation', exact: true }).isDisabled(), true, 'An illustrative 2D extrusion must not be executable as a 3D study.');
    const beforeSwitch = await camera();
    await button('2D simulation').click();
    await page.getByText('Parameters validated', { exact: false }).waitFor();
    assert.equal(await button('Run simulation').isDisabled(), true, 'Switching workspaces must not choose an execution target.');
    await button('3D design').click();
    assert.deepEqual(await camera(), beforeSwitch, 'Tab switch must preserve the design camera.');
    checks.push('Illustrative slab labeling; illustrative extrusion cannot run as 3D; workspace switch retains camera and explicit target requirement');

    await page.getByRole('button', { name: 'Ring & bus An annulus beside a straight guide', exact: true }).click();
    await page.screenshot({ path: path.join(output, '3d-workspace.png'), fullPage: true });
    for (const [width, height] of [[1280, 900], [800, 1000], [390, 844]]) {
      await page.setViewportSize({ width, height });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Layout at ${width}px must not overflow horizontally.`);
      await button('Fit view').click();
    }
    await page.screenshot({ path: path.join(output, '3d-mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1600, height: 1050 });
    await canvas.evaluate(element => element.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext());
    await page.getByText('3D view unavailable', { exact: true }).waitFor();
    await button('Reopen viewport').click(); await canvas.waitFor();
    await page.waitForFunction(() => !document.querySelector('.geometry-3d-error'));
    assert.equal(await canvas.getAttribute('data-visible-objects'), '3');
    assert.deepEqual(errors, [], 'No uncaught browser errors.');
    assert.ok(requests.every(value => new URL(value).origin === new URL(url).origin), 'Application assets and API must stay local.');
    checks.push('Responsive layouts 1600/1280/800/390; actual WebGL context loss/recovery; no external application requests');

    // Verify the editor remains usable when WebGL is unavailable at startup.
    const noGl = await chromium.launch({ ...launch, args: ['--disable-webgl'] });
    try {
      const fallback = await noGl.newPage(); await fallback.goto(url);
      await fallback.getByText('3D view unavailable', { exact: true }).waitFor();
      const item = fallback.getByLabel('Object name', { exact: true });
      await item.fill('Editable without WebGL'); await item.press('Enter');
      await fallback.waitForFunction(() => document.querySelector('input[aria-label="Object name"]').value === 'Editable without WebGL');
      assert.equal(await item.inputValue(), 'Editable without WebGL');
      checks.push('WebGL unavailable: visible explanation and functional parameter editor');
    } finally { await noGl.close(); }
    const report = { passed: true, checked_at: new Date().toISOString(), browser: await browser.version(), checks, browser_errors: errors, scope: '3D geometry editor only; no 3D solver claim' };
    fs.writeFileSync(path.join(output, 'geometry-browser.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
