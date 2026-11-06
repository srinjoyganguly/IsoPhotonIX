// QA only. Launch a dedicated test profile with --remote-debugging-port=0.
// node browser-desktop-window.cjs ORIGIN WINDOWS_PROFILE OUTPUT_DIR create|reopen
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const [origin, profile, output, action] = process.argv.slice(2);
  assert.ok(origin && profile && output && ['create','reopen','finish-run'].includes(action));
  fs.mkdirSync(output, { recursive: true });
  const port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').trim().split(/\r?\n/)[0];
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
  const context = browser.contexts()[0];
  const page = context.pages().find(p => p.url().startsWith(origin + '/'));
  assert.ok(page, 'The real desktop window must already be open at the requested origin.');
  const errors = [], external = [];
  page.on('pageerror', e => errors.push(e.message));
  await context.route('**/*', route => {
    if (route.request().url().startsWith(origin + '/')) return route.continue();
    external.push(route.request().url()); return route.abort();
  });
  await page.reload();
  const title = 'Finite taper desktop recovery';
  const field = page.getByRole('textbox', { name: 'Design name', exact: true });
  await field.waitFor({ timeout: 30000 });
  if (action === 'create') {
    assert.notEqual(await field.inputValue(), title);
    await field.fill(title);
    await field.press('Enter');
  }
  await page.waitForFunction(title => JSON.parse(localStorage.getItem('isophotonix.geometry.v1') || 'null')?.name === title, title);
  assert.equal(await field.inputValue(), title);
  const canvas = page.locator('.geometry-3d canvas').first();
  await canvas.waitFor({ timeout: 20000 });
  const visibleObjects = Number(await canvas.getAttribute('data-visible-objects'));
  assert.ok(visibleObjects > 0);
  const camera = await canvas.getAttribute('data-camera');
  await canvas.scrollIntoViewIfNeeded();
  await canvas.hover();
  await page.mouse.wheel(0, 180);
  await page.waitForFunction(initial => document.querySelector('.geometry-3d canvas')?.dataset.camera !== initial, camera);
  await page.getByRole('button', { name: 'Fit view', exact: true }).first().click();
  const session = await context.newCDPSession(page);
  const { targetInfo } = await session.send('Target.getTargetInfo');
  const { bounds } = await session.send('Browser.getWindowForTarget', { targetId: targetInfo.targetId });
  assert.ok(bounds.width > 300 && bounds.height > 200);
  await page.screenshot({ path: path.join(output, 'desktop-' + action + '.png') });
  const record = { passed:true, checked_at:new Date().toISOString(), action, origin, page_title:await page.title(),
    actual_chromium_window_bounds_px:{width:bounds.width,height:bounds.height},
    persisted_geometry_name:await field.inputValue(), visible_webgl_objects:visibleObjects, camera_zoom_checked:true,
    document_requests_outside_loopback:external, browser_errors:errors };
  assert.deepEqual(external, []);
  assert.deepEqual(errors, []);
  if (action === 'finish-run') {
    const example = await context.request.get(origin + '/api/taper-ports/example');
    assert.equal(example.status(), 200);
    const config = await example.json();
    const submitted = await context.request.post(origin + '/api/taper-ports/jobs', { data: {...config, solver:'meep', device:'cpu'} });
    assert.equal(submitted.status(), 202, await submitted.text());
    const job = await submitted.json();
    assert.ok(['queued', 'running'].includes(job.status));
    record.run_submitted_before_window_close = job.id;
  }
  fs.writeFileSync(path.join(output, 'desktop-' + action + '.json'), JSON.stringify(record, null, 2)+'\n');
  // Close only this application window. The normal desktop launcher observes
  // its exit and shuts down the specific server it created.
  await session.send('Target.closeTarget', { targetId: targetInfo.targetId });
  console.log(JSON.stringify(record));
})().catch(error => { console.error(error); process.exitCode=1; }).finally(()=>process.exit(process.exitCode||0));
