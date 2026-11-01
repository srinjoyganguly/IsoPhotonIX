/** Focused real-data UI verification. Optional Playwright setup: docs/browser-tests.md. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');

(async()=>{
  const output=path.resolve(process.env.PHOTONICS_ARTIFACT_DIR||'test-results/upgrades');fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.PHOTONICS_BROWSER_CHANNEL?{channel:process.env.PHOTONICS_BROWSER_CHANNEL}:{})});
  const context=await browser.newContext({viewport:{width:1280,height:900}});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.PHOTONICS_URL||'http://127.0.0.1:8000';
  async function json(url){const response=await page.request.get(base+url);assert.equal(response.status(),200,url);return response.json();}
  async function waitFrame(index,count){await page.waitForFunction(({index,count})=>document.querySelector('.frame-scrubber strong')?.textContent===`${index+1} / ${count}`&&document.querySelector('.frame-time small')?.textContent==='Simulated time',{index,count});}
  async function comparePixel(values,peak){
    let sample={x:0,y:0,value:0};values.forEach((row,y)=>row.forEach((value,x)=>{if(Math.abs(value)>Math.abs(sample.value))sample={x,y,value};}));
    const actual=await page.locator('.field-canvas').evaluate((canvas,s)=>Array.from(canvas.getContext('2d').getImageData(s.x,canvas.height-s.y-1,1,1).data),sample);
    const strength=Math.abs(sample.value)/peak;const color=sample.value<0?[73,141,249]:[255,157,104];const expected=[19,28,41].map((value,i)=>Math.round(value+(color[i]-value)*strength));
    assert.deepEqual(actual,[...expected,255],'Canvas sample must equal the real numerical array using its declared scale.');
  }
  try{
    await page.goto(base);await page.getByRole('button',{name:'2D simulation',exact:true}).click();await page.getByText('Parameters validated',{exact:false}).waitFor();
    await page.getByLabel('Core width',{exact:true}).fill('0.65');await page.getByText('Parameters validated',{exact:false}).waitFor();
    const target=process.env.PHOTONICS_TEST_TARGET||'meep:cpu';await page.locator(`input[name="execution-target"][value="${target}"]`).check({force:true});
    await page.getByRole('button',{name:'Run simulation'}).click();await page.locator('.status-tag.completed').waitFor({timeout:180000});await page.locator('.field-canvas').waitFor();
    const exportPath=await page.getByRole('link',{name:'Export results .zip'}).getAttribute('href');const id=exportPath.split('/')[3];
    const info=await json(`/api/jobs/${id}/frames`);assert.ok(info.count>1&&info.global_peak_abs>0);
    await page.getByRole('tab',{name:'Time evolution'}).click();await waitFrame(0,info.count);
    const firstScale=await page.locator('.colorbar-labels').textContent();
    await page.getByLabel('Recorded frame',{exact:true}).focus();await page.getByLabel('Recorded frame',{exact:true}).press('End');await waitFrame(info.count-1,info.count);
    assert.equal(await page.locator('.colorbar-labels').textContent(),firstScale,'Playback scale must remain fixed.');
    const last=await json(`/api/jobs/${id}/frames/${info.count-1}`);await comparePixel(last.field,info.global_peak_abs);
    assert.equal(await page.locator('.frame-time strong').textContent(),`${last.time_fs.toFixed(2)} fs`);
    await page.getByRole('button',{name:'Play recorded frames'}).click();await page.waitForFunction(()=>Number(document.querySelector('.frame-scrubber strong')?.textContent?.split('/')[0])>=3);await page.getByRole('button',{name:'Pause playback'}).click();
    await page.waitForFunction(()=>document.querySelector('.frame-time small')?.textContent==='Simulated time');
    await page.getByLabel('Recorded frame',{exact:true}).focus();await page.getByLabel('Recorded frame',{exact:true}).press('Home');await waitFrame(0,info.count);
    await page.getByRole('button',{name:'Play recorded frames'}).click();
    await page.waitForFunction(()=>Number(document.querySelector('.frame-scrubber strong')?.textContent?.split('/')[0])>=3);
    await page.getByRole('button',{name:'3D design',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.frame-play')?.getAttribute('aria-label')==='Play recorded frames'&&document.querySelector('.frame-time small')?.textContent==='Simulated time');
    const hiddenFrame=await page.locator('.frame-scrubber strong').textContent();
    const hiddenTime=await page.locator('.frame-time strong').textContent();
    await page.waitForTimeout(1200);
    assert.equal(await page.locator('.frame-scrubber strong').textContent(),hiddenFrame,'Hidden simulation playback must not advance.');
    assert.equal(await page.locator('.frame-time strong').textContent(),hiddenTime,'Hidden simulation must retain its displayed time.');
    await page.getByRole('button',{name:'2D simulation',exact:true}).click();
    assert.equal(await page.getByRole('button',{name:'Play recorded frames'}).isVisible(),true,'Returning to simulation must leave playback paused.');
    assert.equal(await page.locator('.frame-scrubber strong').textContent(),hiddenFrame,'Returning to simulation must retain its frame.');
    assert.equal(await page.locator('.frame-time strong').textContent(),hiddenTime,'Returning to simulation must retain its time.');
    await page.screenshot({path:path.join(output,'time-evolution.png'),fullPage:true});
    await page.getByRole('tab',{name:'Ez field',exact:true}).click();const final=await json(`/api/jobs/${id}/field`);const finalPeak=Math.max(...final.field.flat().map(Math.abs));await comparePixel(final.field,finalPeak);
    await page.locator('.run-passport>summary').click();await page.locator('.passport-file code').first().waitFor();const passport=await json(`/api/jobs/${id}/passport`);
    const expectedHashes=passport.files.filter(file=>['config.json','arrays.npz'].includes(file.name)).map(file=>file.sha256);assert.deepEqual(await page.locator('.passport-file code').allTextContents(),expectedHashes);
    assert.equal(await page.locator('.reproduce-command').textContent(),passport.reproduce_command);assert.equal(passport.config.core_width_um,0.65);
    await page.locator('.run-passport').screenshot({path:path.join(output,'run-passport.png')});
    await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.setViewportSize({width:1280,height:900});
    await page.getByLabel('Core width',{exact:true}).fill('0.7');await page.getByText('The editor has changed.',{exact:false}).waitFor();assert.equal((await json(`/api/jobs/${id}/passport`)).config.core_width_um,0.65);
    await page.getByRole('button',{name:'Run simulation'}).click();assert.equal(await page.locator('.evolution-player').count(),0);await page.locator('.status-tag.completed').waitFor({timeout:180000});await page.locator('.field-canvas').waitFor();
    const second=(await page.getByRole('link',{name:'Export results .zip'}).getAttribute('href')).split('/')[3];assert.notEqual(second,id);
    const secondInfo=await json(`/api/jobs/${second}/frames`);await page.getByRole('tab',{name:'Time evolution'}).click();await waitFrame(0,secondInfo.count);
    const failing=`**/api/jobs/${second}/frames/1`;await page.route(failing,route=>route.abort('failed'));
    await page.getByLabel('Recorded frame',{exact:true}).focus();await page.getByLabel('Recorded frame',{exact:true}).press('ArrowRight');await page.getByRole('alert').filter({hasText:'Time evolution unavailable'}).waitFor();
    await page.unroute(failing);await page.getByRole('button',{name:'Retry',exact:true}).click();await waitFrame(1,secondInfo.count);
    assert.deepEqual(errors,[]);
    const report={passed:true,target,run_id:id,second_run_id:second,frame_count:info.count,global_peak_abs:info.global_peak_abs,checks:['real array-to-canvas equality','fixed amplitude scale','exact fs label','play and pause recorded frames','workspace switch pauses playback and preserves frame/time','final snapshot preserved','passport hashes and reproduction command','immutable passport config','new run resets frames','visible injected transport failure and real retry','390px no horizontal overflow'],browser_errors:errors};
    fs.writeFileSync(path.join(output,'upgrade-smoke.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
