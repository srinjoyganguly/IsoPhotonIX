/** Check vendor data, Q labels, measured field samples and custom port exports in Chrome. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
(async()=>{
  const origin=process.env.PHOTONICS_URL||'http://127.0.0.1:8000';
  const out=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR||'test-results/spatial-reference');
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:960}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin);await page.getByRole('button',{name:'Calibrated components',exact:true}).click();
    const panel=page.getByRole('region',{name:'Calibrated components',exact:true});
    const api=await (await page.request.get(origin+'/api/calibrations')).json();
    let pairs=0, fields=0, qRows=0, powerDiagnostics=0;
    const modelExports=[];
    for(const key of ['ring-3d','grating-3d','multimode-tm','layout-ports-2d','layout-ports-3d']){
      const preset=api.find(p=>p.case===key);
      if(!preset)continue;
      await panel.getByRole('button',{name:preset.name,exact:true}).click();
      const records=panel.getByRole('region',{name:'Recorded evidence'});
      for(const [i,r] of (preset.evidence?.recorded||[]).entries()){
        await records.getByLabel('Recorded result').selectOption(String(i));
        if(r.report.power_diagnostic){
          const detail=r.report.power_diagnostic;
          if(detail.observable==='ring'){
            const note=await records.getByRole('region',{name:'Ring detuning'}).textContent();
            assert(note.includes((1000*detail.nearest_published_drop_line.published_sampled_center_um).toFixed(2)));
          }else{
            const note=await records.getByRole('region',{name:'Grating power difference'}).textContent();
            assert(note.includes((100*detail.mean_absolute_difference).toFixed(2)));
          }
          powerDiagnostics++;
        }
        const view=records.getByRole('region',{name:'Commercial spectrum comparison'});
        const series={};
        for(const [name,value] of Object.entries(r.report.vendor_comparison?.channels||{})){
          series[name]=value.paired_spectrum;
        }
        const ref=r.report.comparisons?.find(c=>c.geometry_matched);
        if(ref && (ref.published_power || ref.published_s21_power)){
          series['Guided output']={wavelength_um:r.report.result.wavelength_um,local_power:r.report.result.beam_to_guide,
            published_power:ref.published_power||ref.published_s21_power};
        }
        for(const [name,data] of Object.entries(series)){
          await view.getByLabel('Compared output').selectOption(name);
          for(const j of [0,Math.floor(data.wavelength_um.length/2),data.wavelength_um.length-1]){
            await view.getByLabel('Compared wavelength').selectOption(String(j));
            const label=await view.getByLabel('Compared powers').textContent();
            assert(label.includes((100*data.local_power[j]).toFixed(3)+'%'));
            assert(label.includes((100*data.published_power[j]).toFixed(3)+'%'));
            assert(label.endsWith((100*(data.local_power[j]-data.published_power[j])).toFixed(3)+' percentage points'));
            pairs++;
          }
        }
        if(r.report.loaded_loss){
          const q=records.getByRole('region',{name:'Loaded resonance measurements'});
          const qualified=r.report.loaded_loss.peaks.filter(p=>p.qualified && p.qualified_equal_bus_loss);
          assert.equal(await q.locator('tbody tr').count(),qualified.length);qRows+=qualified.length;
        }
        if(r.field){
          const field=records.getByRole('region',{name:'Measured field plane'});
          const x=Math.floor(r.field.horizontal_um.length/2),y=Math.floor(r.field.vertical_um.length/2);
          assert((await field.getByLabel('Field sample').textContent()).endsWith(r.field.values[y][x].toPrecision(7)));
          fields++;
        }
      }
      if(key==='ring-3d'||key==='grating-3d'){
        await panel.getByLabel('3D geometry').selectOption('lumerical');
        const ring=key==='ring-3d';
        const model=ring?'exact-area':'direct-yee';
        await panel.getByLabel(ring?'Ring boundary sampling':'Grating material sampling').selectOption(model);
        if(ring)await panel.getByLabel('Ring material response').selectOption('palik-ade');
        else await panel.getByLabel('Grating wavelengths').selectOption('dense');
        await panel.getByRole('radio',{name:/FDTDX · GPU/i}).check();
        const post=page.waitForRequest(request=>request.url().endsWith('/api/reproduce/calibration')&&request.method()==='POST');
        const download=page.waitForEvent('download');
        await panel.getByRole('button',{name:'Export Python + notebook',exact:true}).click();
        const config=(await post).postDataJSON();
        assert.equal(config[ring?'ring_interface':'grating_material_model'],model);
        if(ring)assert.equal(config.ring_material,'palik-ade');
        else assert.equal(config.grating_spectrum,'dense');
        await (await download).saveAs(path.join(out,key+'-program.zip'));
        modelExports.push({case:key,model,ring_material:config.ring_material??null,wavelength_grid:config.grating_spectrum??null});
      }
      if(key==='layout-ports-2d'){
        const input=panel.getByLabel('west x',{exact:true});
        await input.fill('-2');await input.press('Tab');
        const json=JSON.parse(await panel.getByLabel('Planar layout JSON').inputValue());
        assert.equal(json.ports[0].center_um[0],-2);
      }
    }
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({passed:true,checked_at:new Date().toISOString(),
      matched_vendor_readouts:pairs,native_field_readouts:fields,qualified_loaded_q_rows:qRows,power_diagnostics:powerDiagnostics,
      model_program_exports:modelExports,
      editable_port_position:true,mobile_width_px:390,browser_errors:errors},null,2)+'\n');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
