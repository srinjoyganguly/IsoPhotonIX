/** Read-only browser verification of the actual local public-mode benchmark report.
 * Optional PHOTONICS_URL, PHOTONICS_BROWSER_CHANNEL, PHOTONICS_ARTIFACT_DIR.
 * Only missing-report and transport-failure states are mocked; numerical evidence
 * and its export come from the actual backend. This script starts no solver jobs.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
(async () => {
  const base = process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output = path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/public-benchmarks');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.PHOTONICS_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, acceptDownloads: true });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base);
    const panel = page.getByRole('region', { name: 'Public mode benchmark evidence' });
    await panel.locator('.benchmark-card').first().waitFor();
    const response = await page.request.get(`${base}/api/benchmarks`);
    assert.ok(response.ok()); const evidence = await response.json();
    assert.equal(evidence.status, 'available'); assert.equal(evidence.scope, 'mode-benchmarks');
    assert.ok(evidence.cases.length >= 3, 'Expected the three completed public mode benchmarks.');
    assert.equal(await panel.locator('.benchmark-card').count(), evidence.cases.length);
    for (const item of evidence.cases) {
      const card = panel.getByRole('article', { name: item.title, exact: true });
      assert.equal(await card.locator('h3').innerText(), item.title);
      const values = await card.locator('.benchmark-values dd').allTextContents();
      assert.deepEqual(values, [String(Number(item.computed_neff.toFixed(7))), String(Number(item.reference_neff.toFixed(7))), `${(100 * item.relative_error).toPrecision(4)}%`]);
      if (Number.isFinite(item.last_refinement_relative_change)) assert.ok((await card.locator('.benchmark-mesh-note').innerText()).includes(`${(100 * item.last_refinement_relative_change).toPrecision(4)}%`));
      const expectedKind = item.reference_kind === 'analytic_reference' ? 'Analytic reference' : 'Published solver value';
      assert.equal(await card.locator('.benchmark-reference-kind').innerText(), expectedKind);
      assert.match(await card.locator('.benchmark-reference-note').innerText(), /no commercial solver was run/);
      assert.match(await card.locator('.benchmark-card-scope').innerText(), /does not validate driven 3D fields, ports, transmission, or a complete device/);
      assert.equal(await card.locator('.benchmark-source').getAttribute('href'), new URL(item.source_url).href);
      await card.locator('.benchmark-case-details > summary').click();
      if (item.refinements?.length) assert.equal(await card.locator('tbody tr').count(), item.refinements.length);
      if (item.reproduction_command) assert.equal(await card.locator('.benchmark-command').innerText(), item.reproduction_command);
      await card.locator('.benchmark-case-details > summary').click();
    }
    const component = panel.locator('details[aria-label="Published component comparisons"]');
    await component.locator(':scope > summary').click();
    const comparisonResponse = await page.request.get(base+'/api/component-benchmarks');
    assert.equal(comparisonResponse.status(),200);
    const comparisons = await comparisonResponse.json();
    const format = n => n.toLocaleString('en-GB',{maximumSignificantDigits:8});
    assert.equal(await component.locator('article').count(),comparisons.cases.length);
    for(const [i,item] of comparisons.cases.entries()){
      const article=component.locator('article').nth(i);
      assert.equal(await article.locator('h3').innerText(),item.title);
      assert.equal(await article.locator('a').getAttribute('href'),item.source_url);
      for(const [j,metric] of item.metrics.entries()){
        const row=article.locator('tbody tr').nth(j);
        assert.deepEqual(await row.locator('td').allTextContents(),[format(metric.reference),format(metric.computed),format(100*metric.relative_error)+'%']);
      }
      await article.locator('details > summary').click();
      assert.deepEqual(await article.locator('details p').allTextContents(),item.limitations);
      await article.locator('details > summary').click();
    }
    const awaiting=page.waitForEvent('download');
    await component.getByRole('link',{name:'Download values, sources and refinement evidence'}).click();
    const comparisonFile=path.join(output,'component-comparisons.json');
    await (await awaiting).saveAs(comparisonFile);
    assert.deepEqual(JSON.parse(fs.readFileSync(comparisonFile,'utf8')),comparisons);
    await component.screenshot({path:path.join(output,'component-comparison-table.png')});
    await component.locator(':scope > summary').click();
    const downloading = page.waitForEvent('download');
    await panel.getByRole('button', { name: 'Download evidence JSON', exact: true }).click();
    const download = await downloading, file = path.join(output, download.suggestedFilename()); await download.saveAs(file);
    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), evidence);
    await panel.locator('.benchmark-report > summary').first().click();
    if (evidence.reproduction_command) assert.equal(await panel.locator('.benchmark-report .benchmark-command').innerText(), evidence.reproduction_command);
    if (evidence.report_sha256) assert.ok((await panel.locator('.benchmark-hash').innerText()).includes(evidence.report_sha256));
    await panel.locator('.benchmark-report > summary').first().click();
    await panel.screenshot({ path: path.join(output, 'public-mode-benchmarks-desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await panel.scrollIntoViewIfNeeded();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No horizontal page overflow at 390px.');
    await panel.screenshot({ path: path.join(output, 'public-mode-benchmarks-mobile.png') });
    await page.route('**/api/benchmarks', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ schema_version: 1, scope: 'mode-benchmarks', status: 'not_run', cases: [], notes: [], reproduction_command: evidence.reproduction_command }) }));
    await panel.getByRole('button', { name: 'Refresh evidence', exact: true }).click();
    await panel.getByText('Benchmarks not run', { exact: true }).waitFor();
    assert.equal(await panel.locator('.benchmark-card').count(), 0);
    await page.unroute('**/api/benchmarks');
    await page.route('**/api/benchmarks', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Unavailable for transport-state test' }) }));
    await panel.getByRole('button', { name: 'Refresh evidence', exact: true }).click();
    await panel.getByRole('alert').waitFor();
    assert.match(await panel.getByRole('alert').innerText(), /HTTP 503/);
    assert.equal(await panel.locator('.benchmark-card').count(), 0);
    await page.unroute('**/api/benchmarks');
    await panel.getByRole('button', { name: 'Refresh evidence', exact: true }).click();
    await panel.locator('.benchmark-card').first().waitFor();
    assert.deepEqual(errors, []);
    const report = { passed: true, checked_at: new Date().toISOString(), browser: await browser.version(), component_metrics: comparisons.cases.map(c=>({id:c.id,vendor:c.vendor,metrics:c.metrics.length})), cases: evidence.cases.map(({ id, reference_kind, computed_neff, reference_neff, relative_error }) => ({ id, reference_kind, computed_neff, reference_neff, relative_error })), checks: ['Every published component metric, local computation, source, displayed percentage and limitation matches the comparison API; the downloaded JSON is exact', 'All displayed neff values and relative differences equal the actual MPB benchmark API report', 'Each card identifies analytic versus published solver references and the mode-only scope', 'Official source links, numerical refinement details and reproduction commands preserved', 'Downloaded evidence JSON equals the complete API response', 'Report provenance hash retained', '390px mobile layout has no horizontal overflow', 'Missing report and HTTP failure show no numerical result cards', 'Refresh recovers the actual evidence report', 'No browser exceptions'], new_solver_run: false, transport: 'Actual benchmark data; only empty-report and HTTP-failure UI states intercepted', browser_errors: errors };
    fs.writeFileSync(path.join(output, 'public-benchmarks-browser.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
