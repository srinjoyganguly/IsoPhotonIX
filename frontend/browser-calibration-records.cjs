/** Compare every bundled matrix cell to its browser display, and inspect each spectrum. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

async function compareMatrix(view, report) {
  const labels=report.ports || report.port_order || report.channels;
  let count=0;
  for(let input=0; input<labels.length; input++){
    await view.getByLabel('Input channel').selectOption(String(input));
    for(let f=0; f<report.wavelength_um.length; f++){
      await view.getByLabel('Sample wavelength').selectOption(String(f));
      const displayed=await view.locator('tbody tr').evaluateAll(rows=>
        rows.map(row=>Array.from(row.querySelectorAll('td'),cell=>cell.textContent)));
      const expected=labels.map((_,out)=>{
        const z=report.s?report.s[f][out][input]:[report.s_real[f][out][input],report.s_imag[f][out][input]];
        return [z[0],z[1],z[0]*z[0]+z[1]*z[1]].map(x=>x.toPrecision(5));
      });
      assert.deepEqual(displayed,expected);
      count+=labels.length;
    }
  }
  return count;
}

const spectrumKeys = {
  'Through': ['through', 'valid_bins'],
  'Drop': ['drop', 'valid_bins'],
  'Reflection': ['reflection', 'valid_bins'],
  'Add-side outgoing': ['add_side', 'valid_bins'],
  'Fiber to TE guide': ['beam_to_guide','valid_beam_bins'],
  'TE guide to Gaussian': ['guide_to_beam','valid_energy_bins'],
  'Gaussian to TE guide': ['beam_to_guide', 'valid_beam_bins'],
  'Total output aperture flux': ['direct_output_fraction', 'valid_direct_bins'],
  'Upward radiation fraction': ['guide_radiation_up', 'valid_energy_bins'],
  'Downward radiation fraction': ['guide_radiation_down', 'valid_energy_bins'],
};

async function checkSpectra(view, report, entry) {
  const geometry = view.getByLabel('Reference geometry');
  const options = await geometry.locator('option').allTextContents();
  for (const name of options) {
    await geometry.selectOption(name);
    const row = report.series?.[name] || report.studies?.[name]?.fine || report.studies?.[name]?.guided
      || (name === 'Tidy3D compact aperture' ? report.compact_tidy_domain : report.result);
    assert.ok(row, name);
    const wavelength = view.getByLabel('Wavelength sample');
    const readout = view.getByLabel('Measured power fraction');
    const wavelengths = row.wavelength_um;
    const options = await wavelength.locator('option').allTextContents();
    assert.deepEqual(options, wavelengths.map(w => w.toFixed(5) + ' µm'));
    const observable = view.getByLabel('Observable');
    const series = await observable.locator('option').allTextContents();
    for (const [j, label] of series.entries()) {
      await observable.selectOption(String(j));
      const [key, maskKey] = spectrumKeys[label];
      const values = row[key], mask = row[maskKey] || values.map(() => true);
      assert.equal(values.length, wavelengths.length);
      assert.equal(values.length, mask.length);

      // Check every plotted point against the native measurements. Invalid samples break lines.
      const samples = values.filter((_, i) => mask[i]);
      const top = Math.max(.01, ...samples), bottom = Math.min(0, ...samples);
      const lo = Math.min(...wavelengths), hi = Math.max(...wavelengths);
      const segments = [];
      for (let i = 0; i < wavelengths.length; i++) {
        if (!mask[i]) continue;
        if (i === 0 || !mask[i - 1]) segments.push([]);
        segments.at(-1).push([36 + 466 * (wavelengths[i] - lo) / (hi - lo || 1),
          125 - 110 * (values[i] - bottom) / (top - bottom)]);
      }
      const plotted = await view.locator('polyline').evaluateAll(lines =>
        lines.map(line => line.getAttribute('points').trim().split(/\s+/).map(pair => pair.split(',').map(Number))));
      assert.deepEqual(plotted, segments);
      entry.plotted_samples += samples.length;

      // Exercise both boundary samples, the nominal band center and at least one rejected sample.
      const indices = new Set([0, Math.floor(values.length / 2), values.length - 1, mask.indexOf(false)]);
      for (const i of indices) {
        if (i < 0) continue;
        await wavelength.selectOption(String(i));
        assert.equal(await readout.textContent(), mask[i] ? values[i].toPrecision(7) : 'Unqualified');
        entry.sample_readouts++;
      }
      if (samples.length) {
        // Click at the first wavelength in the plot's own coordinates, including descending grids.
        const svg = view.locator('svg');
        const point = await svg.evaluate((el, x) => {
          const p = new DOMPoint(x, 55).matrixTransform(el.getScreenCTM());
          return {clientX:p.x,clientY:p.y};
        }, 36 + 466 * (wavelengths[0] - lo) / (hi - lo || 1));
        await svg.dispatchEvent('click', point);
        assert.equal(await wavelength.inputValue(), '0');
      }
      entry.spectra++;
    }
  }
}

(async()=>{
  const origin=process.env.PHOTONICS_URL || 'http://127.0.0.1:8000';
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR || 'test-results/calibrated-records');
  fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:960}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(120000);
    const catalog=await (await page.request.get(origin+'/api/calibrations')).json();
    await page.goto(origin);
    await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const panel=page.getByRole('region',{name:'Calibrated components',exact:true});
    const cases={};
    for(const preset of catalog){
      await panel.getByRole('button',{name:preset.name,exact:true}).click();
      const records=panel.getByRole('region',{name:'Recorded evidence',exact:true});
      const entry={records:0,matrix_entries:0,spectra:0,plotted_samples:0,sample_readouts:0,bare_ring_pairs:0};
      for(const [i,record] of preset.evidence.recorded.entries()){
        await records.getByLabel('Recorded result').selectOption(String(i));
        const r=record.report;
        if(r.s || r.s_real){
          entry.matrix_entries+=await compareMatrix(records.getByRole('region',{name:'Measured scattering'}),r);
        }else if(r.series || r.studies || r.compact_tidy_domain || r.result){
          await checkSpectra(records.getByRole('region',{name:'Measured spectrum'}),r,entry);
        }else{
          const table=records.getByRole('region',{name:'Isolated ring spacing'});
          const pairs=await table.locator('tbody tr').allTextContents();
          assert.deepEqual(pairs,Object.entries(r.isolated_finite_height_ring).map(([name,row])=>
            name + row.measured_bare_ring_fsr_nm.toFixed(4) + row.published_loaded_ring_fsr_nm.toFixed(4)));
          entry.bare_ring_pairs+=pairs.length;
        }
        entry.records++;
      }
      assert(entry.records>0);cases[preset.case]=entry;
    }
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({passed:true,checked_at:new Date().toISOString(),cases,
      no_browser_errors:true,mobile_width:390},null,2)+'\n');
    console.log(JSON.stringify(cases));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
