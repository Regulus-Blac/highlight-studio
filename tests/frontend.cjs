/* Run against a local server. Media responses are contract fixtures, not real inference. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.FRONTEND_URL || 'http://127.0.0.1:8000';
const { expect } = require(process.env.PLAYWRIGHT_MODULE ? `${process.env.PLAYWRIGHT_MODULE}/test` : '@playwright/test');
const taskId = '2'.repeat(32), uploadId = '1'.repeat(32), exportId = '3'.repeat(32);
const planTemplate = JSON.parse(fs.readFileSync('contracts/examples/preview.json', 'utf8'));

(async () => {
  const browser = await chromium.launch({channel: process.env.BROWSER_CHANNEL || 'msedge', headless: true});
  const context = await browser.newContext({viewport:{width:1440,height:1050}});
  const page = await context.newPage();
  const errors = [];
  let fixtureServer;
  page.on('pageerror', e => errors.push(e.message));
  try {
    let basicPreviewRequests=0,processingRequests=0;page.on('request',r=>{if(r.url().endsWith('/preview'))basicPreviewRequests++;if(r.method()==='POST'&&/\/api\/v1\/(uploads|tasks)/.test(r.url()))processingRequests++;});
    await page.goto(base);
    await expect(page.locator('#service-label')).toHaveText('服务已连接');
    await expect(page.locator('#start-analysis')).toBeDisabled();
    await expect(page.locator('[data-execution=cloud]')).toHaveAttribute('aria-pressed','true');
    await expect(page.locator('#start-analysis')).toHaveText('服务暂不可用');
    assert.equal(await page.evaluate(()=>time(59.999)),'01:00.00');
    await page.locator('#mode-help').click();
    await expect(page.locator('#mode-explanation')).toContainText('不会调用 AI');
    await page.locator('[data-view=settings]').click();
    await expect(page.locator('#settings-engine')).toHaveText('演示分析');
    await page.locator('[data-view=editor]').click();
    await page.locator('#file-input').setInputFiles({name:'bad.txt',mimeType:'text/plain',buffer:Buffer.from('bad')});
    await expect(page.locator('#message-text')).toContainText('请选择 MP4');
    // Generate a video entirely in-browser; no external fixtures or user files.
    const bytes = await page.evaluate(async () => {
      const canvas = document.createElement('canvas');canvas.width=320;canvas.height=180;
      const ctx=canvas.getContext('2d'),stream=canvas.captureStream(10),chunks=[];
      const recorder=new MediaRecorder(stream,{mimeType:'video/webm'});
      recorder.ondataavailable=e=>chunks.push(e.data);
      const done=new Promise(resolve=>recorder.onstop=resolve);
      recorder.start();
      let frame=0;const draw=setInterval(()=>{ctx.fillStyle=`hsl(${frame++*10},50%,35%)`;ctx.fillRect(0,0,320,180);ctx.fillStyle='white';ctx.font='20px sans-serif';ctx.fillText('LOCAL TEST VIDEO',60,100);},100);
      await new Promise(r=>setTimeout(r,3100));recorder.stop();await done;clearInterval(draw);stream.getTracks().forEach(t=>t.stop());
      return Array.from(new Uint8Array(await new Blob(chunks,{type:'video/webm'}).arrayBuffer()));
    });
    const video = {name:'local-test.webm',mimeType:'video/webm',buffer:Buffer.from(bytes)};
    await page.locator('#file-input').setInputFiles(video);
    await expect(page.locator('#replace-file')).toBeVisible();
    await expect(page.locator('#replace-file')).toBeEnabled();
    await page.locator('[data-ratio="16,9"]').click();
    await expect(page.locator('#start-analysis')).toBeDisabled();
    await page.locator('[data-execution=local]').click();
    await expect(page.locator('#start-analysis')).toHaveText('服务暂不可用');
    await expect(page.locator('#start-analysis')).toBeDisabled();
    await page.evaluate(()=>start());assert.equal(processingRequests,0,'Unready cloud/local modes must not transmit video');
    await page.locator('#debug-mode').click();
    await page.locator('#start-analysis').click();
    await expect(page.locator('#task-state')).toContainText('演示已完成',{timeout:15000});
    await expect(page.locator('#mode-notice')).toContainText('演示模式');
    await expect(page.locator('[data-execution=cloud]')).toBeDisabled();
    await expect(page.locator('#debug-mode')).toBeDisabled();
    await page.reload();
    await expect(page.locator('#task-state')).toContainText('演示已完成',{timeout:15000});
    assert.equal(basicPreviewRequests,0,'Basic mock must not request unavailable media preview');
    await expect(page.locator('#global-message')).toBeHidden();
    await expect(page.locator('#timeline-summary')).toHaveText('演示已完成');
    await page.locator('[data-view=history]').click();
    await expect(page.locator('.history-card')).toHaveCount(1);
    console.log('PASS actual HTTP: import, task, mock label, refresh recovery, history');
    await context.close();

    const mediaContext = await browser.newContext({viewport:{width:1440,height:1050}});
    const p = await mediaContext.newPage();p.on('pageerror',e=>errors.push(e.message));
    let taskReads=0,exportReads=0,failCreateOnce=true,created=0,exportCreated=0;
    const keys=[],exportRequests=[];
    let currentPlan=structuredClone(planTemplate), latestSegments=null;
    const result={api_version:'1',video_id:uploadId,target_ratio:[16,9],engine:'synthetic-media-v1',mock:true,predictions:planTemplate.frame_crops.map(f=>({frame:f.frame,bboxes:f.crop.slice(0,3)}))};
    const task=(status)=>({api_version:'1',task_id:taskId,upload_id:uploadId,video_id:uploadId,target_ratio:[16,9],engine:'synthetic-media-v1',mock:true,status,progress:{stage:status==='running'?'predicting':status==='succeeded'?'completed':'queued',percent:status==='running'?null:status==='succeeded'?100:0},result:status==='succeeded'?result:null,error:null,created_at:new Date().toISOString()});
    // Chromium aborts downloads fulfilled by route interception. Serve download bytes
    // over a real local socket; all other media payloads remain explicit fixtures.
    fixtureServer=http.createServer(async(req,res)=>{
      if(req.url===`/api/v1/exports/${exportId}/file`){
        const body=JSON.stringify(result)+'\n';res.writeHead(200,{'Content-Type':'application/x-ndjson','Content-Length':Buffer.byteLength(body)});res.end(body);return;
      }
      try{const upstream=await fetch(base+req.url);res.writeHead(upstream.status,{'Content-Type':upstream.headers.get('content-type')||'application/octet-stream'});res.end(Buffer.from(await upstream.arrayBuffer()));}
      catch{res.writeHead(502);res.end();}
    });
    await new Promise(resolve=>fixtureServer.listen(0,'127.0.0.1',resolve));
    const mediaBase=`http://127.0.0.1:${fixtureServer.address().port}`;
    await mediaContext.route('**/*',async route=>{
      const req=route.request(),url=new URL(req.url()),path=url.pathname;
      assert.equal(url.origin,mediaBase,'No external network requests permitted');
      const reply=(body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
      if(path==='/ready')return reply({ready:true,mock:true,engine:'synthetic-media-v1'});
      if(path==='/api/v1/model')return reply({api_version:'1',available:true,mock:true,engine:'synthetic-media-v1',prediction_api_version:'1'});
      if(path==='/api/v1/uploads')return reply({api_version:'1',upload_id:uploadId,filename:'local-test.webm',size_bytes:bytes.length,sha256:'0'.repeat(64),created_at:new Date().toISOString()},201);
      if(path==='/api/v1/tasks'){
        assert.deepEqual(req.postDataJSON(),{api_version:'1',upload_id:uploadId,target_ratio:[16,9]});keys.push(req.headers()['idempotency-key']);if(!created)created++;
        if(failCreateOnce){failCreateOnce=false;return route.abort();}return reply(task('queued'),200);
      }
      if(path===`/api/v1/tasks/${taskId}`)return reply(task(++taskReads===1?'running':'succeeded'));
      if(path===`/api/v1/tasks/${taskId}/preview`){
        if(req.method()==='POST'){
          latestSegments=req.postDataJSON().segments;currentPlan=structuredClone(planTemplate);currentPlan.segments=latestSegments;
          currentPlan.frame_crops=planTemplate.frame_crops.filter(f=>latestSegments.some(s=>s.enabled&&f.frame>=s.start_frame&&f.frame<s.end_frame_exclusive));
          currentPlan.duration_seconds=currentPlan.frame_crops.length/10;currentPlan.output=currentPlan.frame_crops.length?planTemplate.output:null;
        }
        return reply(currentPlan);
      }
      if(path===`/api/v1/tasks/${taskId}/source`)return route.fulfill({contentType:'video/webm',body:Buffer.from(bytes)});
      if(path==='/api/v1/exports'){exportCreated++;exportReads=0;exportRequests.push(req.postDataJSON());return reply({api_version:'1',export_id:exportId,task_id:taskId,format:req.postDataJSON().format,status:'queued',mock:true,artifact:null,error:null},202);}
      if(path===`/api/v1/exports/${exportId}`)return reply({api_version:'1',export_id:exportId,task_id:taskId,format:exportRequests.at(-1).format,status:++exportReads===1?'running':'succeeded',mock:true,error:null,artifact:exportReads===1?null:{url:`/api/v1/exports/${exportId}/file`,size_bytes:200,sha256:'0'.repeat(64),duration_seconds:currentPlan.duration_seconds,output:currentPlan.output}});
      if(path===`/api/v1/exports/${exportId}/file`)return route.continue();
      return route.continue();
    });
    await p.goto(mediaBase);await expect(p.locator('#service-label')).toHaveText('服务已连接');
    await p.locator('#file-input').setInputFiles(video);await p.locator('[data-ratio="16,9"]').click();await p.locator('#debug-mode').click();await p.locator('#start-analysis').click();
    await expect(p.locator('#task-state')).toContainText('任务创建未确认');
    await p.reload();await expect(p.locator('#task-state')).toContainText('有待确认');await p.locator('#retry-task').click();
    await expect(p.locator('#task-state')).toContainText('正在分析视频',{timeout:10000});
    assert.equal(await p.locator('#task-state').textContent(),'正在分析视频，请稍候…');
    await expect(p.locator('.segment-row')).toHaveCount(2,{timeout:15000});
    assert.equal(keys.length,2);assert.equal(keys[0],keys[1]);assert.equal(created,1);
    await p.locator('#crop-view').click();await expect(p.locator('#crop-canvas')).toBeVisible();
    await p.locator('[aria-label="片段 1 起点 / 秒"]').fill('2.5');await p.locator('#apply-edits').click();await expect(p.locator('#message-text')).toContainText('不能重叠');
    await p.locator('[aria-label="片段 1 起点 / 秒"]').fill('0');
    // A failed connection can expose resume while an unapplied draft exists.
    await p.evaluate(()=>document.getElementById('resume-task').hidden=false);
    p.once('dialog',dialog=>dialog.dismiss());await p.locator('#resume-task').click();
    await expect(p.locator('#apply-edits')).toBeEnabled();
    await expect(p.locator('[aria-label="片段 1 起点 / 秒"]')).toHaveValue('0');
    await p.locator('[aria-label="选择片段 2"]').uncheck();
    await expect(p.locator('#export-button')).toBeDisabled();await p.locator('#apply-edits').click();await expect(p.locator('#export-count')).toHaveText('1 段');
    assert.equal(latestSegments[1].enabled,false);assert.equal(latestSegments[0].start_frame,0);
    await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:'runtime/frontend-media-desktop.png',fullPage:true});
    await p.locator('#export-format').selectOption('jsonl');await p.locator('#export-button').click();
    await expect(p.locator('#download')).toBeVisible({timeout:15000});assert.equal(exportCreated,1);assert.equal(exportRequests[0].segments[1].enabled,false);
    const downloadEvent=p.waitForEvent('download');await p.locator('#download').click();const download=await downloadEvent;await download.saveAs('runtime/frontend-test-result.jsonl');assert.equal(JSON.parse(fs.readFileSync('runtime/frontend-test-result.jsonl','utf8')).mock,true);
    await p.reload();await expect(p.locator('#download')).toBeVisible({timeout:15000});await expect(p.locator('#export-count')).toHaveText('1 段');
    await p.locator('[aria-label="选择片段 1"]').uncheck();await p.locator('#apply-edits').click();await expect(p.locator('#export-count')).toHaveText('0 段');await p.locator('#export-format').selectOption('mp4');await expect(p.locator('#export-button')).toBeDisabled();await p.locator('#export-format').selectOption('jsonl');await expect(p.locator('#export-button')).toBeEnabled();
    await p.setViewportSize({width:390,height:844});await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:'runtime/frontend-mobile.png',fullPage:true});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'Mobile must not overflow horizontally');
    await p.locator('#new-project').click();await p.locator('#new-dialog [value=confirm]').click();await expect(p.locator('#empty-state')).toBeVisible();
    await p.locator('[data-view=history]').click();await expect(p.locator('.history-card')).toHaveCount(1);
    await p.locator('[data-view=editor]').click();
    await p.route('**/api/v1/model',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({api_version:'1',engine:'configured-local-test',mock:false,available:true,prediction_api_version:'1'})}));
    await p.locator('#refresh-service').click();await expect(p.locator('#local-availability')).toHaveText('可以使用');
    await p.locator('[data-execution=local]').click();await p.locator('#file-input').setInputFiles(video);
    await expect(p.locator('#start-analysis')).toBeEnabled();await expect(p.locator('#start-analysis')).toHaveText('✦ 开始分析');
    await p.route('**/api/v1/deployment',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({location:'remote',mode:'real',transport:'ssh'})}));
    await p.route('**/api/v1/model',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({api_version:'1',engine:'f1-derived-video-v0',mock:false,available:true,prediction_api_version:'1'})}));
    await p.locator('#refresh-service').click();
    await expect(p.locator('#start-analysis')).toHaveText('✦ 开始分析');
    await expect(p.locator('#debug-mode')).toBeHidden();
    const internal=/\bF1\b|f1-derived|SSH|\bGPU\b|mock-v1|configured-local-test|127\.0\.0\.1/;
    assert(!internal.test(await p.locator('body').innerText()),'The editor must not show internal engine or deployment details');
    await p.locator('[data-view=settings]').click();
    await expect(p.locator('#settings-engine')).toHaveText('智能分析');
    assert(!internal.test(await p.locator('body').innerText()),'Settings must use user-facing service names');
    await p.locator('[data-view=editor]').click();
    console.log('PASS user-facing copy: clean editor and settings, explicit demo labels');
    console.log('PASS execution modes: default cloud, unready upload guard, mode lock, configured local service');
    console.log('PASS contract-fixture browser: ambiguous create recovery, idempotency, running state, crop, invalid edits, selection, export/download, reload, empty selection, mobile, new task');
    assert.deepEqual(errors,[]);console.log('PASS no browser JavaScript errors');await mediaContext.close();
  } finally {await browser.close();if(fixtureServer)await new Promise(resolve=>fixtureServer.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
