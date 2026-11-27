/* Exercise the real saved-launcher gateway. No SSH login, GPU or model assets. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {spawn}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {expect}=require(process.env.PLAYWRIGHT_MODULE?`${process.env.PLAYWRIGHT_MODULE}/test`:'@playwright/test');
const fixture=`
import tempfile, threading, sys
from scripts.start_demo import RemoteConnection, load_profile, make_server
from services.model_adapter.adapter import MockAdapter
def offline():
    raise RuntimeError('Intentionally offline browser fixture')
with tempfile.TemporaryDirectory() as directory:
    connection=RemoteConnection(load_profile())
    connection._start_process=offline
    server=make_server({**load_profile(), 'local_port':0}, connection, runtime=directory, adapter=MockAdapter())
    connection.connect()
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    print(server.server_port,flush=True)
    try:
        sys.stdin.readline()
    finally:
        connection.close()
        server.shutdown(); server.server_close(); worker.join(5)
`;

(async()=>{
 const server=spawn(process.env.PYTHON_BIN||'python',['-u','-c',fixture],{stdio:['pipe','pipe','pipe']});
 let stderr='';server.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-12000);});
 let browser;
 try{
  const port=await new Promise((resolve,reject)=>{
   let output='';const timeout=setTimeout(()=>reject(new Error(`Fixture did not start: ${stderr}`)),15000);
   server.once('error',error=>{clearTimeout(timeout);reject(error);});
   server.once('exit',code=>{clearTimeout(timeout);reject(new Error(`Fixture exited (${code}): ${stderr}`));});
   server.stdout.on('data',chunk=>{output+=chunk;if(output.includes('\n')){clearTimeout(timeout);resolve(Number(output.trim()));}});
  });
  assert(port>0);const base=`http://127.0.0.1:${port}`;
  browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'msedge',headless:true});
  const context=await browser.newContext({viewport:{width:1440,height:1050}}),page=await context.newPage(),errors=[],requests=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>requests.push({method:request.method(),path:new URL(request.url()).pathname}));
  await page.goto(base);
  await expect(page.locator('#service-label')).toHaveText('本机测试已就绪');
  await expect(page.locator('#cloud-availability')).toHaveText('未连接');
  await expect(page.locator('[data-execution=cloud]')).toBeDisabled();
  await expect(page.locator('#debug-mode')).toBeEnabled();
  await expect(page.locator('#debug-mode')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#choose-file')).toBeEnabled();
  console.log('PASS offline page loads with remote disabled and local demo available');

  const bytes=Buffer.from(await page.evaluate(async()=>{
   const canvas=document.createElement('canvas');canvas.width=320;canvas.height=180;
   const ctx=canvas.getContext('2d'),stream=canvas.captureStream(10),chunks=[];
   ctx.fillStyle='#244879';ctx.fillRect(0,0,320,180);
   const recorder=new MediaRecorder(stream,{mimeType:'video/webm'});
   recorder.ondataavailable=event=>chunks.push(event.data);const done=new Promise(resolve=>recorder.onstop=resolve);
   recorder.start();const timer=setInterval(()=>ctx.fillRect(0,0,320,180),100);
   await new Promise(resolve=>setTimeout(resolve,1100));recorder.stop();await done;clearInterval(timer);
   stream.getTracks().forEach(track=>track.stop());return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
  }));
  await page.locator('#file-input').setInputFiles({name:'offline-test.webm',mimeType:'video/webm',buffer:bytes});
  await expect(page.locator('#start-analysis')).toBeEnabled();
  await page.locator('#start-analysis').click();
  await expect(page.locator('#task-state')).toContainText('演示已完成',{timeout:15000});
  await expect(page.locator('#mode-notice')).toContainText('演示模式');
  assert(requests.some(r=>r.method==='POST'&&r.path==='/demo/api/v1/uploads'));
  assert(requests.some(r=>r.method==='POST'&&r.path==='/demo/api/v1/tasks'));
  assert(!requests.some(r=>r.method==='POST'&&/^\/api\/v1\/(uploads|tasks)/.test(r.path)),'Local demo must never transmit a task to the remote');
  await page.reload();
  await expect(page.locator('#task-state')).toContainText('演示已完成',{timeout:15000});
  await expect(page.locator('#video')).toHaveAttribute('src',new RegExp(`^${base}/demo/api/v1/tasks/[a-f0-9]{32}/source$`));
  await page.waitForFunction(()=>document.getElementById('video').readyState>=1);
  assert(requests.some(r=>/^\/demo\/api\/v1\/tasks\/[a-f0-9]{32}\/source$/.test(r.path)),'Native media requests must stay local after reload');
  console.log('PASS real local upload, empty mock task, persisted recovery and native video source');

  // Advertise a ready remote without running inference. Local records must stay local.
  const metadata={
   '/api/v1/deployment':{location:'remote',mode:'real',transport:'ssh',connection:'connected',local_demo:true},
   '/api/v1/model':{api_version:'1',engine:'f1-derived-video-v0',mock:false,available:true,prediction_api_version:'1'},
   '/ready':{ready:true,engine:'f1-derived-video-v0',mock:false}
  };
  const routes=[];
  for(const [path,value] of Object.entries(metadata)){
   const url=base+path,handler=route=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
   routes.push([url,handler]);await page.route(url,handler);
  }
  await page.locator('#refresh-service').click();
  await expect(page.locator('#cloud-availability')).toHaveText('可以使用');
  await expect(page.locator('#debug-mode')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#service-label')).toHaveText('本机测试已就绪');
  await page.reload();await expect(page.locator('#task-state')).toContainText('演示已完成');
  assert((await page.locator('#video').getAttribute('src')).includes('/demo/api/v1/tasks/'));
  await page.locator('#new-project').click();await page.locator('#new-dialog button[value=confirm]').click();
  await expect(page.locator('[data-execution=cloud]')).toBeEnabled();
  await page.locator('[data-execution=cloud]').click();
  await expect(page.locator('#debug-mode')).toBeEnabled();
  await expect(page.locator('#debug-mode')).toBeVisible();
  for(const [url,handler] of routes)await page.unroute(url,handler);
  await page.locator('#refresh-service').click();
  await expect(page.locator('#cloud-availability')).toHaveText('未连接');
  await expect(page.locator('[data-execution=cloud]')).toBeDisabled();
  console.log('PASS reconnect enables remote, preserves local records and keeps local demo selectable');

  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1050});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Offline layout fits ${width}px`);
   fs.mkdirSync('runtime/offline-startup',{recursive:true});
   await page.screenshot({path:`runtime/offline-startup/${width}.png`,fullPage:true});
  }
  await context.close();

  // A saved real task fails explicitly; it must never be queried as a local demo.
  const oldContext=await browser.newContext(),old=await oldContext.newPage(),oldRequests=[],taskId='a'.repeat(32);
  old.on('pageerror',error=>errors.push(error.message));
  old.on('request',request=>oldRequests.push(new URL(request.url()).pathname));
  await oldContext.addInitScript(({taskId})=>{
   const record={execution:'cloud',id:taskId,uploadId:'b'.repeat(32),fileName:'real-task.mp4',ratio:[16,9],status:'running'};
   localStorage.setItem('highlight.studio.workspace.v1.remote.real',JSON.stringify({records:[record],activeId:taskId}));
  },{taskId});
  await old.goto(base);
  await expect(old.locator('#message-text')).toContainText('远程服务器未连接');
  await expect(old.locator('#task-state')).toContainText('任务恢复失败');
  assert(oldRequests.includes(`/api/v1/tasks/${taskId}`));
  assert(!oldRequests.some(path=>path.includes(`/demo/api/v1/tasks/${taskId}`)));
  await old.locator('#debug-mode').click();
  await expect(old.locator('#file-summary')).toHaveText('选择视频后即可开始');
  assert.equal(await old.evaluate(()=>state.record),null,'Switching mode starts a clean workspace and preserves old records only in history');
  assert.equal(await old.evaluate(()=>state.records.length),1);
  await oldContext.close();assert.deepEqual(errors,[]);
  console.log('PASS old real task fails honestly, local mode clears its upload reference, no browser errors');
 }finally{
  if(browser)await browser.close();
  server.stdin.end('\n');
  if(server.exitCode===null)await new Promise(resolve=>{
   const timeout=setTimeout(()=>{server.kill();resolve();},7000);
   server.once('exit',()=>{clearTimeout(timeout);resolve();});
  });
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
