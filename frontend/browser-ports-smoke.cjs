/** Real local reference+device browser journey, or read-only inspection of an existing run.
 * PHOTONICS_SOLVER=meep|fdtdx, PHOTONICS_DEVICE=cpu|gpu (defaults meep/cpu).
 * PHOTONICS_PORT_RUN_ID skips solving and inspects that saved run.
 * Optional PHOTONICS_URL, PHOTONICS_BROWSER_CHANNEL, PHOTONICS_ARTIFACT_DIR.
 * Masked-bin and HTTP-error responses are UI-state fixtures only; all reported
 * physical evidence and exported files are fetched from the actual local worker.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const fmt = value => value === null ? 'Masked' : Number(value).toPrecision(6);
(async () => {
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/ports'); fs.mkdirSync(output, { recursive: true });
  const solver = process.env.PHOTONICS_SOLVER || 'meep', device = process.env.PHOTONICS_DEVICE || 'cpu';
  let id = process.env.PHOTONICS_PORT_RUN_ID || null;
  const readOnly = !!id;
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, acceptDownloads: true });
  const errors = [], observedStages = new Set(); let submissions = 0;
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().endsWith('/api/ports/jobs') && request.method() === 'POST') submissions++; });
  page.on('response', response => {
    if (id && response.url().endsWith(`/api/jobs/${id}`) && response.ok()) void response.json().then(value => { if (value.port_progress?.stage) observedStages.add(value.port_progress.stage); }).catch(() => {});
  });
  const panel = page.getByRole('region', { name: '2D port measurements', exact: true });
  async function enter() {
    await page.goto(base); await page.getByRole('navigation', { name: 'Workspace' }).getByRole('button', { name: 'Ports · 2D', exact: true }).click();
    await panel.getByText('Port study validated', { exact: true }).waitFor({ timeout: 70000 });
    await panel.locator('input[name="port-target"]').first().waitFor({ timeout: 70000 });
  }
  async function reopenSaved() {
    await enter(); await panel.getByLabel('Recent port run', { exact: true }).selectOption(id);
    await panel.locator(`.port-spectrum[data-run-id="${id}"]`).waitFor({ timeout: 30000 });
  }
  async function download(action) {
    const waiting = page.waitForEvent('download'); await action.click(); const item = await waiting;
    const filename = path.join(output, item.suggestedFilename()); await item.saveAs(filename); return filename;
  }
  try {
    await enter();
    assert.equal(await panel.locator('input[name="port-target"]:checked').count(), 0, 'No automatic target selection.');
    assert.equal(await panel.getByRole('button', { name: 'Run reference + device', exact: true }).isDisabled(), true);
    if (!readOnly) {
      await panel.getByRole('button', { name: 'Waveguide gap', exact: false }).click();
      await panel.getByLabel('Port Gap length', { exact: true }).waitFor();
      await panel.getByText('Port study validated', { exact: true }).waitFor();
      assert.equal(await panel.locator('input[name="port-target"]:checked').count(), 0, 'Opening a new example requires explicit target selection.');
      const target = panel.locator(`input[name="port-target"][value="${solver}:${device}"]`);
      assert.equal(await target.isEnabled(), true, `${solver}/${device} must be port-qualified by the actual runtime probe.`);
      if (device === 'gpu') assert.match(await target.locator('..').innerText(), /mode preparation.*CPU|source-mode preparation uses CPU/i);
      await target.check();
      await panel.getByLabel('Port Gap length', { exact: true }).fill('0.01');
      await panel.locator('.port-validation.invalid').waitFor();
      assert.equal(await panel.getByRole('button', { name: 'Run reference + device', exact: true }).isDisabled(), true);
      assert.equal(submissions, 0, 'Invalid parameters cannot submit a solver job.');
      await panel.getByLabel('Port Gap length', { exact: true }).fill('0.3125');
      await panel.getByText('Port study validated', { exact: true }).waitFor();
      const createdResponse = page.waitForResponse(response => response.url().endsWith('/api/ports/jobs') && response.request().method() === 'POST');
      await panel.getByRole('button', { name: 'Run reference + device', exact: true }).click();
      const response = await createdResponse; assert.equal(response.status(), 202); const created = await response.json(); id = created.id;
      assert.equal(created.config.gap_um, .3125); assert.equal(created.config.solver, solver); assert.equal(created.config.device, device);
      await panel.locator(`.port-spectrum[data-run-id="${id}"]`).waitFor({ timeout: 330000 });
      assert.equal(submissions, 1, 'One submitted job contains the matched reference and device solves.');
    } else {
      await panel.getByLabel('Recent port run', { exact: true }).selectOption(id);
      await panel.locator(`.port-spectrum[data-run-id="${id}"]`).waitFor({ timeout: 30000 });
    }
    const jobResponse = await page.request.get(`${base}/api/jobs/${id}`); assert.ok(jobResponse.ok()); const job = await jobResponse.json();
    assert.equal(job.status, 'completed'); assert.equal(job.config.study_type, 'waveguide-ports-2d');
    const actualResponse = await page.request.get(`${base}/api/jobs/${id}/spectrum`); assert.ok(actualResponse.ok()); const spectrum = await actualResponse.json();
    if (readOnly) {
      await panel.getByLabel('Import port configuration', { exact: true }).setInputFiles({ name: 'saved-port-study.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(job.config)) });
      await panel.getByText('Port study validated', { exact: true }).waitFor();
      await panel.locator(`input[name="port-target"][value="${job.config.solver}:${job.config.device}"]`).check();
      await panel.getByText('Port study validated', { exact: true }).waitFor();
      await panel.getByLabel('Port Core index', { exact: true }).fill('0.5'); await panel.locator('.port-validation.invalid').waitFor();
      assert.equal(await panel.getByRole('button', { name: 'Run reference + device', exact: true }).isDisabled(), true);
      assert.equal(submissions, 0);
      await panel.getByLabel('Port Core index', { exact: true }).fill(String(job.config.core_index)); await panel.getByText('Port study validated', { exact: true }).waitFor();
    }
    assert.equal(spectrum.metadata.spectra_valid, true, 'This physical demonstration must meet its recorded validity checks.');
    for (const phase of ['reference', 'device']) assert.equal(spectrum.metadata.settlement[phase].settled, true, `${phase} must meet its recorded settlement criteria.`);
    assert.ok(spectrum.valid_bin.every(Boolean)); assert.equal(spectrum.wavelength_um.length, job.config.frequency_points);
    assert.match(String(spectrum.metadata.actual_device), job.config.device === 'gpu' ? /cuda|gpu/i : /cpu/i);
    for (let index = 0; index < spectrum.wavelength_um.length; index++) {
      assert.ok(Math.abs(spectrum.reflectance[index] + spectrum.reflected_flux[index] / spectrum.incident_flux[index]) < 2e-5);
      assert.ok(Math.abs(spectrum.transmittance[index] - spectrum.transmitted_flux[index] / spectrum.incident_flux[index]) < 2e-5);
      assert.ok(Math.abs(spectrum.uncollected_fraction[index] - (1 - spectrum.reflectance[index] - spectrum.transmittance[index])) < 2e-5);
      for (const key of ['reflectance', 'transmittance', 'uncollected_fraction']) {
        const point = panel.locator(`[data-series="${key}"] circle[data-bin="${index}"]`);
        assert.equal(Number(await point.getAttribute('data-value')), spectrum[key][index], `${key} plotted sample must equal the actual API value.`);
      }
    }
    const rawValues = ['reflectance', 'transmittance', 'uncollected_fraction'].flatMap(key => spectrum[key]);
    const low = Math.min(0, ...rawValues), high = Math.max(1, ...rawValues), padding = Math.max(.03, (high - low) * .07), yMin = low - padding, yMax = high + padding;
    for (const key of ['reflectance', 'transmittance', 'uncollected_fraction']) {
      const index = Math.floor(spectrum.wavelength_um.length / 2), point = panel.locator(`[data-series="${key}"] circle[data-bin="${index}"]`);
      assert.ok(Math.abs(Number(await point.getAttribute('cy')) - (20 + (yMax - spectrum[key][index]) / (yMax - yMin) * 250)) < 1e-8, 'Graph positions retain signed numerical values.');
    }
    const slider = panel.getByLabel('Port spectral sample', { exact: true });
    for (const keypress of ['Home', 'End']) {
      await slider.focus(); await slider.press(keypress); const index = Number(await slider.inputValue());
      for (const key of ['reflectance', 'transmittance', 'uncollected_fraction']) assert.equal(await panel.locator(`.port-selected-sample [data-result="${key}"]`).innerText(), fmt(spectrum[key][index]));
    }
    assert.match(await panel.locator('.port-interpretation').innerText(), /not an absorption measurement/);
    await panel.locator('.port-samples > summary').click(); assert.equal(await panel.locator('.port-samples tbody tr').count(), spectrum.wavelength_um.length); await panel.locator('.port-samples > summary').click();
    const jsonFile = await download(panel.getByRole('button', { name: 'Export spectrum JSON', exact: true }));
    assert.deepEqual(JSON.parse(fs.readFileSync(jsonFile, 'utf8')), spectrum);
    const csvFile = await download(panel.getByRole('link', { name: 'Export spectrum CSV', exact: true }));
    const csvLines = fs.readFileSync(csvFile, 'utf8').trim().split(/\r?\n/).map(line => line.split(','));
    const columns = csvLines.shift(); assert.equal(csvLines.length, spectrum.wavelength_um.length);
    for (let index = 0; index < csvLines.length; index++) for (const key of ['wavelength_um', 'frequency_per_um', 'incident_flux', 'reflected_flux', 'transmitted_flux', 'reference_transmitted_flux', 'reflectance', 'transmittance', 'uncollected_fraction']) assert.equal(Number(csvLines[index][columns.indexOf(key)]), spectrum[key][index]);
    assert.ok(csvLines.every(row => row[columns.indexOf('valid_bin')] === 'true'));
    const zipFile = await download(panel.getByRole('link', { name: 'Export port results .zip', exact: true })); const archive = fs.readFileSync(zipFile); assert.equal(archive.readUInt32LE(0), 0x04034b50); assert.ok(archive.length > 2000);
    const configFile = await download(panel.getByRole('button', { name: 'Save run configuration', exact: true })); assert.deepEqual(JSON.parse(fs.readFileSync(configFile, 'utf8')), job.config);
    await panel.locator('.run-passport > summary').click(); await panel.locator('.passport-file > span').filter({ hasText: 'spectra.npz' }).waitFor(); await panel.locator('.passport-file > span').filter({ hasText: 'spectrum.csv' }).waitFor();
    const passportResponse = await page.request.get(`${base}/api/jobs/${id}/passport`); assert.ok(passportResponse.ok()); const passport = await passportResponse.json();
    const expectedSolver = job.config.solver === 'meep' ? 'Meep' : 'FDTDX';
    const expectedVersion = passport.versions[job.config.solver] ?? passport.versions[passport.solver] ?? '';
    assert.equal((await panel.locator('.passport-facts > div').first().locator('strong').innerText()).trim(), `${expectedSolver} ${expectedVersion}`.trim(), 'Passport must identify the actual solver and version, including legacy capitalized metadata.');
    await panel.locator('.run-passport > summary').click();
    if (job.config.device_kind === 'gap') {
      await panel.getByLabel('Port Gap length', { exact: true }).fill('0.375'); await panel.locator('.snapshot-note').waitFor();
      const unchanged = await page.request.get(`${base}/api/jobs/${id}`); assert.equal((await unchanged.json()).config.gap_um, job.config.gap_um);
      await panel.getByLabel('Port Gap length', { exact: true }).fill(String(job.config.gap_um)); await panel.getByText('Port study validated', { exact: true }).waitFor();
    }
    // Presentation-only masked-bin fixture: all other data remain the actual run.
    const middle = Math.floor(spectrum.wavelength_um.length / 2), masked = structuredClone(spectrum);
    masked.valid_bin[middle] = false; masked.metadata.spectra_valid = false;
    for (const key of ['reflectance', 'transmittance', 'uncollected_fraction']) masked[key][middle] = null;
    masked.bin_reasons[middle] = 'UI-state test: masked sample; not physical evidence.';
    await page.route(`**/api/jobs/${id}/spectrum`, route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(masked) }));
    await reopenSaved();
    assert.equal(await panel.locator('.port-selected-sample').getAttribute('data-valid'), 'false');
    assert.equal(await panel.locator('.port-selected-sample [data-result="transmittance"]').innerText(), 'Masked');
    assert.equal(await panel.locator(`.port-spectrum-plot circle[data-bin="${middle}"]`).count(), 0, 'Masked bins must not become zero-valued plotted points.');
    assert.equal((await panel.locator('[data-series="transmittance"] path').getAttribute('d')).match(/M/g).length, 2, 'The curve must break at the masked sample.');
    await page.unroute(`**/api/jobs/${id}/spectrum`);
    await page.route(`**/api/jobs/${id}/spectrum`, route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'UI-state test: spectrum transport failure.' }) }));
    await enter(); await panel.getByLabel('Recent port run', { exact: true }).selectOption(id);
    await panel.getByRole('button', { name: 'Retry spectrum', exact: true }).waitFor(); assert.equal(await panel.locator('.port-spectrum-plot').count(), 0);
    await page.unroute(`**/api/jobs/${id}/spectrum`); await panel.getByRole('button', { name: 'Retry spectrum', exact: true }).click();
    await panel.locator(`.port-spectrum[data-run-id="${id}"]`).waitFor();
    // Restore the saved study in this isolated browser for truthful final screenshots.
    await panel.getByLabel('Import port configuration', { exact: true }).setInputFiles({ name: 'saved-port-study.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(job.config)) });
    await panel.getByText('Port study validated', { exact: true }).waitFor();
    await panel.locator(`input[name="port-target"][value="${job.config.solver}:${job.config.device}"]`).check();
    await panel.getByText('Port study validated', { exact: true }).waitFor();
    await page.waitForTimeout(300);
    assert.equal(await panel.locator('.snapshot-note').count(), 0, 'Restored editor matches the immutable saved study.');
    await panel.locator('.port-output').screenshot({ path: path.join(output, 'port-measurement-result.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No horizontal page overflow at 390px.');
    await panel.locator('.port-output').screenshot({ path: path.join(output, 'port-measurement-mobile.png') });
    assert.deepEqual(errors, []); assert.equal(submissions, readOnly ? 0 : 1);
    const report = { passed: true, checked_at: new Date().toISOString(), browser: await browser.version(), run_id: id, solver: job.config.solver, requested_device: job.config.device, actual_device: spectrum.metadata.actual_device, device_name: spectrum.metadata.device_name, config: job.config, valid_bins: spectrum.valid_bin.filter(Boolean).length, observed_stages: [...observedStages], settlement: spectrum.metadata.settlement, checks: ['No automatic target selection; invalid parameters cannot submit', 'Real matched-reference and device study completed with recorded settlement', 'Every plotted spectral sample equals its actual saved value; signed coordinates retained', 'R/T and uncollected residual match the signed incident normalization', 'Keyboard sample inspection and full numerical table', 'CSV and JSON exports equal the actual API data; result ZIP downloaded; spectrum hashes visible in passport', ...(readOnly ? [] : ['Edited parameters were submitted exactly; later edits cannot change saved configuration']), 'Masked-bin UI fixture produces gaps and no fabricated zeros', 'HTTP-error UI fixture hides unavailable spectrum; retry restores actual data', '390px layout has no horizontal page overflow', 'No browser exceptions'], new_solver_jobs: submissions, ui_state_fixtures: ['One masked bin based on an actual spectrum, explicitly not physical evidence', 'HTTP 503 spectrum response'], browser_errors: errors };
    fs.writeFileSync(path.join(output, 'ports-browser.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
