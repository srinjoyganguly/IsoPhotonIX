/** Verify the local editor’s real library placement, saved links and edit history. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const path = require('node:path');
const { chromium } = require('playwright');

const storageKey = 'photonics-workbench.geometry.v1';
const tolerance = 1e-7;

function endpoints(edge) {
  return [edge.a, edge.b].map(end => [end.component_id, end.port]).sort();
}

function port(scene, id, name) {
  const component = scene.components.find(item => item.id === id);
  assert.ok(component, `Missing component ${id}`);
  const selected = component.ports.find(item => item.name === name);
  assert.ok(selected, `Missing port ${id}/${name}`);
  return selected;
}

function separation(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function angleError(a, b) {
  return Math.abs((((a.angle_deg - b.angle_deg) % 360 + 360) % 360) - 180);
}

function withoutLinks(scene) {
  const { connections, ...rest } = scene;
  return rest;
}

(async () => {
  const url = process.env.PHOTONICS_URL;
  assert.ok(url, 'Set PHOTONICS_URL to the isolated Workbench server.');
  const server = new URL(url);
  assert.ok(['127.0.0.1', 'localhost'].includes(server.hostname) && server.port && server.port !== '8000',
    'Circuit QA requires an explicit local port other than the primary service.');
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/circuit-connections');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1512, height: 1040 }, acceptDownloads: true });
  page.setDefaultTimeout(30000);
  const library = page.locator('.component-library');
  const panel = page.locator('details.port-alignment');
  const rows = panel.getByRole('list', { name: 'Saved connections' }).getByRole('listitem');
  const button = (name, scope = page) => scope.getByRole('button', { name, exact: true });
  const label = name => page.getByLabel(name, { exact: true });
  const errors = [], jobs = [], checks = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.method() === 'POST' && /^\/api\/.*jobs(?:\/|$)/.test(new URL(request.url()).pathname)) jobs.push(request.url());
  });

  async function openDetails(locator) {
    if (!(await locator.evaluate(node => node.open))) await locator.locator(':scope > summary').click();
  }
  async function saved() {
    return page.evaluate(key => JSON.parse(localStorage.getItem(key)), storageKey);
  }
  async function waitSaved(expected) {
    await page.waitForFunction(({ key, expected }) => {
      const canonical = value => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
        ? Object.fromEntries(Object.keys(item).sort().map(name => [name, item[name]])) : item);
      return canonical(JSON.parse(localStorage.getItem(key))) === canonical(expected);
    }, { key: storageKey, expected });
  }
  async function action(locator, endpoint) {
    const pending = page.waitForResponse(response => new URL(response.url()).pathname === endpoint && response.request().method() === 'POST');
    await locator.click();
    const response = await pending;
    const data = await response.json();
    assert.ok(response.ok(), `${endpoint}: ${response.status()} ${JSON.stringify(data)}`);
    assert.ok(data.geometry, `No validated geometry for ${endpoint}`);
    await waitSaved(data.geometry);
    return data.geometry;
  }
  async function choosePorts(moving, movingPort, fixed, fixedPort) {
    await openDetails(library);
    await openDetails(panel);
    await label('Moving component').selectOption(moving);
    await label('Moving port').selectOption(movingPort);
    await label('Fixed component').selectOption(fixed);
    await label('Fixed port').selectOption(fixedPort);
  }
  async function verifyLink(scene, state, edge, fixed, moving) {
    assert.equal(scene.connections.length, 1);
    assert.equal(scene.connections[0].id, edge.id);
    assert.deepEqual(endpoints(scene.connections[0]), [[fixed, 'out'], [moving, 'in']].sort());
    const p = port(scene, fixed, 'out'), q = port(scene, moving, 'in');
    const distance = separation(p, q);
    const error = angleError(p, q);
    const response = await page.request.post(url + '/api/library/connections', { data: { geometry: scene } });
    assert.ok(response.ok(), await response.text());
    const diagnostics = (await response.json()).connections;
    assert.equal(diagnostics.length, 1);
    assert.equal(diagnostics[0].state, state);
    assert.equal(diagnostics[0].id, edge.id);
    assert.ok(Math.abs(diagnostics[0].distance_um - distance) < tolerance);
    assert.ok(Math.abs(diagnostics[0].angle_error_deg - error) < tolerance);
    assert.equal(diagnostics[0].widths_match, true);
    if (state === 'aligned') {
      assert.ok(distance < tolerance && error < tolerance);
    } else {
      assert.ok(distance > .1 || error > 1);
    }
    await openDetails(library);
    await openDetails(panel);
    await rows.first().getByText(state === 'aligned' ? 'Aligned' : 'Needs route', { exact: true }).waitFor();
    assert.equal(await rows.count(), 1);
    const legend = page.getByRole('group', { name: 'Connection guide legend' });
    await legend.getByText(state === 'aligned' ? '1 aligned joint' : '1 planned link', { exact: true }).waitFor();
    return diagnostics[0];
  }
  async function captureCircuit(name) {
    // Capture one continuous region of the real editor with no CSS/DOM edits.
    // Both its saved-port status and the complete 3D canvas are inside the crop.
    await button('Fit view').click();
    for (let i = 0; i < 4; i++) await button('Zoom in').click();
    const domain = button('Domain');
    if ((await domain.getAttribute('aria-pressed')) === 'true') await domain.click();
    const clip = await page.evaluate(() => {
      const panel = document.querySelector('details.port-alignment').getBoundingClientRect();
      const workspace = document.querySelector('.geo-workspace').getBoundingClientRect();
      const canvas = document.querySelector('.geometry-3d-host canvas').getBoundingClientRect();
      const x = Math.max(0, Math.floor(Math.min(panel.left, workspace.left) + scrollX - 10));
      const y = Math.max(0, Math.floor(panel.top + scrollY - 10));
      const right = Math.min(document.documentElement.scrollWidth, Math.ceil(Math.max(panel.right, workspace.right) + scrollX + 10));
      const bottom = Math.ceil(canvas.bottom + scrollY + 44);
      return { x, y, width: right - x, height: bottom - y };
    });
    assert.ok(clip.width > 800 && clip.height > 400);
    await page.screenshot({ path: path.join(output, name), fullPage: true, clip, animations: 'disabled' });
  }
  async function exportGeometry(file) {
    const pending = page.waitForEvent('download');
    await button('Export geometry').click();
    const target = path.join(output, file);
    await (await pending).saveAs(target);
    return { target, geometry: JSON.parse(fs.readFileSync(target, 'utf8')) };
  }

  try {
    await page.goto(url);
    await page.locator('.geometry-3d-host canvas').first().waitFor();
    await openDetails(library);
    await button('Choose component Strip waveguide', library).click();
    const first = await action(button('Open as new design', library), '/api/library/build');
    assert.equal(first.components.length, 1);
    const fixed = first.components[0].id;
    await openDetails(library.locator('.library-placement'));
    await label('Component placement x').fill('8');
    await label('Component placement y').fill('7');
    await label('Component placement rotation_deg').fill('90');
    await label('Enlarge domain to fit when adding or updating').check();
    const placed = await action(button('Add to layout', library), '/api/library/place');
    assert.equal(placed.components.length, 2);
    const moving = placed.components.find(c => c.id !== fixed).id;
    await library.locator('.library-placement > summary').click();
    await label('Search photonic components').fill('Strip waveguide');
    assert.equal(placed.components.find(c => c.id === moving).placement.rotation_deg, 90);
    assert.ok(separation(port(placed, fixed, 'out'), port(placed, moving, 'in')) > 1);
    assert.equal((placed.connections ?? []).length, 0);
    checks.push('Two strip instances placed through the public geometry library, with distinct IDs, positions and orientations.');

    await choosePorts(moving, 'in', fixed, 'out');
    await label('Enlarge domain to fit aligned components').check();
    const legacy = await action(button('Align ports', panel), '/api/library/align');
    assert.equal((legacy.connections ?? []).length, 0);
    assert.ok(separation(port(legacy, fixed, 'out'), port(legacy, moving, 'in')) < tolerance);
    await button('Undo').click(); await waitSaved(placed);
    checks.push('Geometry-only alignment still moves the real solids without creating a connection.');

    await choosePorts(moving, 'in', fixed, 'out');
    const planned = await action(button('Connect without moving', panel), '/api/library/connect');
    assert.deepEqual(withoutLinks(planned), withoutLinks(placed));
    assert.equal(planned.connections.length, 1);
    const edge = planned.connections[0];
    assert.ok(typeof edge.id === 'string' && edge.id.length > 0);
    const plannedState = await verifyLink(planned, 'needs_route', edge, fixed, moving);
    await button('Layout ports').click();
    await page.getByRole('group', { name: 'Connection guide legend' }).waitFor({ state: 'hidden' });
    assert.equal(await button('Layout ports').getAttribute('aria-pressed'), 'false');
    assert.deepEqual(await saved(), planned);
    await button('Layout ports').click();
    await page.getByRole('group', { name: 'Connection guide legend' }).getByText('1 planned link', { exact: true }).waitFor();
    await captureCircuit('circuit-planned.png');
    const plannedFile = await exportGeometry('planned-circuit.json');
    assert.deepEqual(plannedFile.geometry, planned);
    checks.push('A planned connection preserves every solid and placement; UI and API report the independently calculated gap and angular error.');

    await label('Fixed port').selectOption('in');
    for (const command of ['Connect without moving', 'Connect and align', 'Align ports']) {
      assert.ok(await button(command, panel).isDisabled(), `Occupied endpoint must disable ${command}`);
    }
    await panel.getByText('One of these ports already has a different connection.', { exact: false }).waitFor();
    assert.deepEqual(await saved(), planned);
    await label('Fixed port').selectOption('out');
    checks.push('All three connection/alignment actions reject a port already owned by a different link; the existing design stays intact.');

    const aligned = await action(button('Connect and align', panel), '/api/library/connect');
    const alignedState = await verifyLink(aligned, 'aligned', edge, fixed, moving);
    assert.deepEqual(aligned.components.find(c => c.id === fixed), placed.components.find(c => c.id === fixed));
    assert.deepEqual(aligned.objects.filter(o => o.component_id === fixed), placed.objects.filter(o => o.component_id === fixed));
    assert.notDeepEqual(aligned.components.find(c => c.id === moving).placement, placed.components.find(c => c.id === moving).placement);
    assert.deepEqual(aligned.components.map(c => c.id), placed.components.map(c => c.id));
    assert.equal(aligned.objects.length, placed.objects.length);
    await choosePorts(fixed, 'out', moving, 'in');
    const reversed = await action(button('Connect without moving', panel), '/api/library/connect');
    assert.deepEqual(reversed, aligned);
    await captureCircuit('circuit-aligned.png');
    const alignedFile = await exportGeometry('aligned-circuit.json');
    assert.deepEqual(alignedFile.geometry, aligned);
    await page.reload(); await page.locator('.geometry-3d-host canvas').first().waitFor();
    await waitSaved(aligned); await verifyLink(aligned, 'aligned', edge, fixed, moving);
    checks.push('Aligning the same pair preserves its link ID, fixes only the moving instance, co-locates anti-parallel equal-width ports, and treats reversed endpoint order as the same link. Reload/export preserve the complete scene.');

    const disconnected = await action(button('Disconnect', rows.first()), '/api/library/disconnect');
    assert.equal((disconnected.connections ?? []).length, 0);
    assert.deepEqual(withoutLinks(disconnected), withoutLinks(aligned));
    await button('Undo').click(); await waitSaved(aligned); await verifyLink(aligned, 'aligned', edge, fixed, moving);
    await button('Redo').click(); await waitSaved(disconnected);
    await button('Undo').click(); await waitSaved(aligned);
    checks.push('Disconnect changes link metadata alone, and undo/redo restore the exact linked/unlinked designs.');

    await label('Select component').selectOption(moving);
    await button('Delete component').click();
    await page.waitForFunction(({ key, id }) => {
      const scene = JSON.parse(localStorage.getItem(key)); return scene.components.length === 1 && !scene.components.some(c => c.id === id);
    }, { key: storageKey, id: moving });
    const deleted = await saved();
    assert.equal((deleted.connections ?? []).length, 0);
    assert.deepEqual(deleted.components, aligned.components.filter(c => c.id !== moving));
    assert.deepEqual(deleted.objects, aligned.objects.filter(o => o.component_id !== moving));
    await button('Undo').click(); await waitSaved(aligned);
    await page.reload(); await page.locator('.geometry-3d-host canvas').first().waitFor();
    await button('Redo').click(); await waitSaved(deleted);
    await label('Import geometry file').setInputFiles(alignedFile.target);
    await waitSaved(aligned); await verifyLink(aligned, 'aligned', edge, fixed, moving);
    checks.push('Deleting a component removes its incident link and solids atomically; undo, reload/redo and reimport retain exact component/port IDs and geometry.');
    assert.deepEqual(errors, []); assert.deepEqual(jobs, []);
    fs.writeFileSync(path.join(output, 'circuit-browser.json'), JSON.stringify({ passed: true, checks, connection_id: edge.id,
      browser_version: browser.version(), component_ids: [fixed, moving],
      diagnostics: { planned: plannedState, aligned: alignedState },
      json_sha256: { planned: createHash('sha256').update(fs.readFileSync(plannedFile.target)).digest('hex'),
        aligned: createHash('sha256').update(fs.readFileSync(alignedFile.target)).digest('hex') },
      coordinates: { fixed: port(aligned, fixed, 'out'), moving: port(aligned, moving, 'in') }, new_solver_jobs: jobs.length }, null, 2)+'\n');
    console.log(JSON.stringify({ passed: true, checks: checks.length, new_solver_jobs: jobs.length }));
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
