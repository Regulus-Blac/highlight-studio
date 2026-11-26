/* Failed/cancelled tasks must never trap the user in a locked editor. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {expect}=require(process.env.PLAYWRIGHT_MODULE?`${process.env.PLAYWRIGHT_MODULE}/test`:'@playwright/test');
const base=process.env.FRONTEND_URL||'http://127.0.0.1:8000';
const id='a'.repeat(32),upload='b'.repeat(32);

(async()=>{
 const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'msedge',headless:true});
 try{
  for(const status of ['failed','cancelled']){
   const context=await browser.newContext({viewport:{width:1440,height:1050}});
   await context.addInitScript(({id,upload,status})=>{
    const record={execution:'cloud',id,uploadId:upload,fileName:'上次的视频.mp4',ratio:[9,16],created:new Date().toISOString(),status};
    localStorage.setItem('highlight.studio.workspace.v1.remote.real',JSON.stringify({records:[record],activeId:id,pending:null}));
   },{id,upload,status});
   await context.route('**/api/v1/deployment',route=>route.fulfill({json:{location:'remote',mode:'real',transport:'ssh'}}));
   await context.route('**/api/v1/model',route=>route.fulfill({json:{api_version:'1',engine:'fixture-real',mock:false,available:true,prediction_api_version:'1'}}));
   await context.route('**/ready',route=>route.fulfill({json:{ready:true,mock:false,engine:'fixture-real'}}));
   await context.route(`**/api/v1/tasks/${id}`,route=>route.fulfill({json:{api_version:'1',task_id:id,upload_id:upload,video_id:upload,target_ratio:[9,16],engine:'fixture-real',mock:false,status,progress:{stage:status,percent:null},result:null,error:status==='failed'?{code:'unsupported_media_profile',message:'Unsupported frame timestamps',retryable:false}:null,created_at:new Date().toISOString()}}));
   const page=await context.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
   await page.goto(base);
   await page.waitForFunction(status=>state.task?.status===status&&!state.busy,status);
   await expect(page.locator('#choose-file')).toBeEnabled();
   await expect(page.locator('#replace-file')).toBeVisible();
   await expect(page.locator('#replace-file')).toBeEnabled();
   await expect(page.locator('[data-ratio="16,9"]')).toBeEnabled();
   await expect(page.locator('#new-project')).toBeEnabled();
   if(status==='failed'){
    await expect(page.locator('#task-state')).toContainText('重新选择视频');
    await expect(page.locator('#retry-task')).toBeHidden();
    await expect(page.locator('#start-analysis')).toBeDisabled();
   }else await expect(page.locator('#retry-task')).toBeEnabled();
   await page.reload();await page.waitForFunction(status=>state.task?.status===status&&!state.busy,status);
   await expect(page.locator('#choose-file')).toBeEnabled();
   const bytes=await page.evaluate(async()=>{
    const c=document.createElement('canvas');c.width=320;c.height=180;const ctx=c.getContext('2d');
    const stream=c.captureStream(5),chunks=[],rec=new MediaRecorder(stream,{mimeType:'video/webm'});rec.ondataavailable=e=>chunks.push(e.data);const done=new Promise(r=>rec.onstop=r);
    rec.start();const timer=setInterval(()=>{ctx.fillStyle='#446688';ctx.fillRect(0,0,320,180);},60);await new Promise(r=>setTimeout(r,500));rec.stop();await done;clearInterval(timer);stream.getTracks().forEach(t=>t.stop());return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
   });
   await page.locator('#file-input').setInputFiles({name:'new-video.webm',mimeType:'video/webm',buffer:Buffer.from(bytes)});
   await expect(page.locator('#start-analysis')).toBeEnabled();
   assert.equal(await page.evaluate(()=>state.task),null,'A new source must not reuse a failed task or upload');
   assert.equal(await page.evaluate(()=>state.record),null);
   assert.equal(await page.evaluate(()=>state.records.length),1,'The old task history must remain');
   assert.deepEqual(errors,[]);
   await context.close();console.log(`PASS recover ${status} task, refresh, replace source, preserve history`);
  }
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
