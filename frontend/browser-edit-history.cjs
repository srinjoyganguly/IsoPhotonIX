/** Exercise real component recipes, rendered ports, editor selection, downloads and keyboard input. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/edit-history');
  fs.mkdirSync(output, { recursive: true });
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1380, height: 950 }, acceptDownloads: true });
  page.setDefaultTimeout(30000);
  const checks = [], errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const label = name => page.getByLabel(name, { exact: true });
  const button = name => page.getByRole('button', { name, exact: true });
  const storageKey = 'photonics-workbench.geometry.v1';
  const saved = () => page.evaluate(key => JSON.parse(localStorage.getItem(key)), storageKey);
  const selectedName = () => label('Object name').inputValue();
  async function expectScene(expected) {
    await page.waitForFunction(({ expected, storageKey }) => {
      const ordered = value => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item)
        ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
      return ordered(JSON.parse(localStorage.getItem(storageKey))) === ordered(expected);
    }, { expected, storageKey });
    const ports = (expected.components ?? []).reduce((sum, item) => sum + item.ports.length, 0);
    const solids = expected.objects.filter(item => item.visible).length;
    await page.waitForFunction(({ ports, solids }) => {
      const canvas = document.querySelector('.geometry-3d-host canvas');
      return canvas?.dataset.visibleObjects === String(solids) && canvas.dataset.layoutPorts === String(ports);
    }, { ports, solids });
  }
  async function generate(route, payload) {
    const response = await page.request.post(base + route, { data: payload });
    assert.ok(response.ok(), await response.text());
    return (await response.json()).geometry;
  }
  async function importScene(value) {
    await label('Import geometry file').setInputFiles({
      name: 'component-history.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(value)),
    });
    await expectScene(value);
  }
  async function exportedScene() {
    const ready = page.waitForEvent('download');
    await button('Export geometry').click();
    const target = path.join(output, 'after-delete.json');
    await (await ready).saveAs(target);
    return JSON.parse(fs.readFileSync(target, 'utf8'));
  }
  try {
    await page.goto(base);
    await page.locator('.geometry-3d-host canvas').waitFor();
    assert.ok(await button('Undo').isDisabled() && await button('Redo').isDisabled());
    const slot = await generate('/api/library/build', { component_id: 'slot' });
    const scene = await generate('/api/library/place', {
      component_id: 'directional_coupler', geometry: slot, placement: { y: 2 }, fit_domain: true,
    });
    const [keep, target] = scene.components;
    const targets = new Set(target.object_ids);
    assert.ok(targets.size >= 2 && target.ports.length > 0);
    const members = scene.objects.filter(object => targets.has(object.id));
    const survivors = scene.objects.filter(object => !targets.has(object.id));
    const primitive = { ...members[0], id: 'unowned-witness', name: 'Independent solid' };
    delete primitive.component_id;
    members.at(-1).visible = false;
    scene.objects = [members[0], survivors[0], primitive, ...members.slice(1), ...survivors.slice(1)];
    await importScene(scene);
    await label('Select component').selectOption(target.id);
    const member = members.at(-1);
    await page.locator('.geo-object-select').filter({ hasText: member.name }).click();
    assert.equal(await selectedName(), member.name);
    assert.equal(await label('Select component').inputValue(), target.id);

    // Undo and Delete on a text field belong to the text field.
    const name = label('Object name');
    await name.press('End');
    await name.pressSequentially(' x');
    await name.press('Control+z');
    assert.equal(await name.inputValue(), member.name);
    await name.press('Control+a');
    await name.press('Delete');
    assert.equal(await name.inputValue(), '');
    assert.deepEqual(await saved(), scene);
    await name.press('Escape');
    assert.equal(await selectedName(), member.name);
    checks.push('Native text undo and delete retained the design and component selection.');

    await button('Delete component').click();
    const removed = { ...scene, objects: scene.objects.filter(object => !targets.has(object.id)), components: [keep] };
    await expectScene(removed);
    assert.ok(await button('Delete object').isDisabled());
    assert.deepEqual(await exportedScene(), removed);
    assert.deepEqual(removed.materials, scene.materials);
    await button('Undo').click();
    await expectScene(scene);
    assert.equal(await selectedName(), member.name);
    assert.equal(await label('Select component').inputValue(), target.id);
    await page.keyboard.press('Control+Shift+z');
    await expectScene(removed);
    checks.push('Selected hidden member deleted its complete recipe and all ports, preserved interleaved solids and shared material, exported exact geometry, and restored original selection.');

    await button('Undo').click();
    await button('Draw in XY').click();
    await button('Select / move').click();
    await page.locator('.geo-object-select').filter({ hasText: member.name }).click();
    await page.keyboard.press('Delete');
    await expectScene(removed);
    await page.keyboard.press('Control+z');
    await expectScene(scene);
    assert.equal(await selectedName(), member.name);
    await page.keyboard.press('Control+y');
    await expectScene(removed);
    await page.keyboard.press('Control+z');
    await expectScene(scene);
    await label('Design name').fill('New history branch');
    await label('Design name').press('Enter');
    await expectScene({ ...scene, name: 'New history branch' });
    assert.ok(await button('Redo').isDisabled());
    await button('Undo').click();
    await expectScene(scene);
    checks.push('Delete and both redo shortcuts worked in XY; a committed rename after undo discarded the obsolete redo branch.');

    // Requesting a partial edit is explicit. Undo restores the individual solid first, then its recipe.
    await button('Convert component to editable solids').click();
    const detached = { ...scene, components: [keep], objects: scene.objects.map(object => {
      if (!targets.has(object.id)) return object;
      const copy = { ...object }; delete copy.component_id; return copy;
    }) };
    await expectScene(detached);
    await button('Delete object').click();
    await expectScene({ ...detached, objects: detached.objects.filter(object => object.id !== member.id) });
    await button('Undo').click();
    await expectScene(detached);
    await button('Undo').click();
    await expectScene(scene);
    checks.push('Explicit detachment permitted a single-solid delete; two undos restored first the solid and then its parametric membership and ports.');

    // A mounted but hidden editor must not consume commands in the separate simulation workspace.
    await button('2D simulation').click();
    await page.locator('body').evaluate(body => { document.activeElement.blur(); body.focus(); });
    await page.keyboard.press('Delete');
    await page.keyboard.press('Control+z');
    assert.deepEqual(await saved(), scene);
    await button('3D design').click();
    await button('3D view').click();
    await importScene(slot);
    await button('Delete component').click();
    const empty = { ...slot, objects: [], components: [] };
    await expectScene(empty);
    assert.ok(await button('Delete object').isDisabled());
    await button('Undo').click();
    await expectScene(slot);
    await button('Redo').click();
    await expectScene(empty);
    await page.reload();
    await page.locator('.geometry-3d-host canvas').waitFor();
    await expectScene(empty);
    assert.ok(await button('Delete object').isDisabled());
    await button('Undo').click();
    await expectScene(slot);
    assert.equal(await selectedName(), slot.objects[0].name);
    await page.reload();
    await page.locator('.geometry-3d-host canvas').waitFor();
    await button('Redo').click();
    await expectScene(empty);
    await button('+ Add').click();
    assert.equal((await saved()).objects.length, 1);
    await button('Undo').click();
    await expectScene(empty);
    checks.push('Delete to an empty scene and undo/redo both survived same-tab reloads. The restored selection and subsequent new geometry remained editable.');

    // Even a new tab with a live opener, which initially copies sessionStorage, starts its own history.
    const popupReady = page.waitForEvent('popup');
    await page.evaluate(url => window.open(url, '_blank'), base);
    const other = await popupReady;
    await other.locator('.geometry-3d-host canvas').waitFor();
    assert.ok(await other.getByRole('button', { name: 'Undo', exact: true }).isDisabled());
    await other.getByLabel('Design name', { exact: true }).fill('Another tab');
    await other.getByLabel('Design name', { exact: true }).press('Enter');
    await page.getByText('Another tab changed the saved draft.', { exact: false }).waitFor();
    await page.reload();
    await page.locator('.geometry-3d-host canvas').waitFor();
    assert.equal(await label('Design name').inputValue(), 'Another tab');
    assert.ok(await button('Undo').isDisabled() && await button('Redo').isDisabled());
    await other.close();
    checks.push('A fresh tab ignored inherited undo history; external draft conflict paused autosave and reload rejected the old checkpoint.');

    // Force only a sessionStorage quota failure. Normal draft persistence and in-memory undo must still work.
    await page.evaluate(() => {
      const setItem = Storage.prototype.setItem;
      Storage.prototype.setItem = function (key, value) {
        if (this === sessionStorage && key === 'photonics-workbench.edit-history.v1') throw new DOMException('Quota', 'QuotaExceededError');
        return setItem.call(this, key, value);
      };
    });
    await button('+ Add').click();
    await page.getByText('Draft saved · undo available until reload', { exact: true }).waitFor();
    const afterAdd = await saved();
    assert.equal(afterAdd.objects.length, 1);
    await button('Undo').click();
    assert.equal((await saved()).objects.length, 0);
    await button('Redo').click();
    await expectScene(afterAdd);
    await page.reload();
    await expectScene(afterAdd);
    assert.ok(await button('Undo').isDisabled());
    checks.push('A full tab history store preserved the separately saved draft and in-memory undo, and prevented recovery of stale history.');
    assert.deepEqual(errors, []);
    const report = { passed: true, checked_at: new Date().toISOString(), checks,
      fixture: { target_family: target.family, target_solids: targets.size, target_ports: target.ports.length,
        preserved_family: keep.family, survivor_solids: survivors.length + 1 }, browser_errors: errors };
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
