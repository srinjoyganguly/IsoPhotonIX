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

async function checkSpectra(view) {
  const geometry=view.getByLabel('Reference geometry');
  const options=await geometry.locator('option').allTextContents();
  let count=0;
  for(const name of options){
    await geometry.selectOption(name);
    const observable=view.getByLabel('Observable');
    const series=await observable.locator('option').allTextContents();
    for(let j=0;j<series.length;j++){
      await observable.selectOption(String(j));
      const points=await view.locator('polyline').evaluateAll(lines=>lines.flatMap(line=>
        line.getAttribute('points').trim().split(/\s+/).filter(Boolean)));
      assert.ok(points.length>0,series[j]);
      assert(points.every(pair=>pair.split(',').length===2 && pair.split(',').every(n=>Number.isFinite(Number(n)))));
      count++;
    }
  }
  return count;
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
      const entry={records:0,matrix_entries:0,spectra:0,bare_ring_pairs:0};
      for(const [i,record] of preset.evidence.recorded.entries()){
        await records.getByLabel('Recorded result').selectOption(String(i));
        const r=record.report;
        if(r.s || r.s_real){
          entry.matrix_entries+=await compareMatrix(records.getByRole('region',{name:'Measured scattering'}),r);
        }else if(r.series || r.studies || r.compact_tidy_domain || r.result){
          entry.spectra+=await checkSpectra(records.getByRole('region',{name:'Measured spectrum'}));
        }else{
          const table=records.getByRole('region',{name:'Isolated ring spacing'});
          const pairs=await table.locator('tbody tr').count();
          assert.equal(pairs,Object.keys(r.isolated_finite_height_ring).length);entry.bare_ring_pairs+=pairs;
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
