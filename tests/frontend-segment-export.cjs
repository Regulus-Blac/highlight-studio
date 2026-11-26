/* Browser export workflow against explicit API fixtures; downloaded payloads are JSON receipts, not encoded video. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {expect}=require(process.env.PLAYWRIGHT_MODULE?`${process.env.PLAYWRIGHT_MODULE}/test`:'@playwright/test');
const taskId='7'.repeat(32),uploadId='8'.repeat(32),exportsByKey=new Map(),requests=[];
const segments=[{suggestion_id:'0',start_frame:1,end_frame_exclusive:6,enabled:true},{suggestion_id:'1',start_frame:10,end_frame_exclusive:15,enabled:true},{suggestion_id:'2',start_frame:20,end_frame_exclusive:26,enabled:true}];
let video=Buffer.alloc(0),hold=false,failNext=false,previewPosts=0;
function plan(edits=segments){
 const frames=edits.filter(s=>s.enabled).flatMap(s=>Array.from({length:s.end_frame_exclusive-s.start_frame},(_,i)=>({frame:s.start_frame+i,crop:[0,0,160,90]})));
 return {api_version:'1',video_id:uploadId,target_ratio:[16,9],engine:'synthetic-media-v1',mock:true,rules_version:'consecutive-inward-v1',plan_id:'f'.repeat(64),source_sha256:'0'.repeat(64),source_url:`/api/v1/tasks/${taskId}/source`,metadata:{width:320,height:180,fps_num:10,fps_den:1,frame_count:30,duration_seconds:3,has_audio:false,rotation:0,validation_profile:'cpu-cfr-v1'},suggestions:segments,segments:edits,frame_crops:frames,duration_seconds:frames.length/10,output:frames.length?{width:160,height:90}:null};
}
function item(entry){
 const duration=entry.request.segments.filter(s=>s.enabled).reduce((sum,s)=>sum+s.end_frame_exclusive-s.start_frame,0)/10;
 return {api_version:'1',export_id:entry.id,task_id:taskId,format:entry.request.format,status:entry.status,mock:true,error:entry.status==='failed'?{code:'render_failed',message:'fixture failure',retryable:false}:null,artifact:entry.status==='succeeded'?{url:`/api/v1/exports/${entry.id}/file`,sha256:'0'.repeat(64),size_bytes:100,duration_seconds:duration,output:{width:160,height:90}}:null};
}
const server=http.createServer(async(req,res)=>{
 const path=new URL(req.url,'http://127.0.0.1').pathname;
 const reply=(data,status=200)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(data));};
 const assets={'/':['apps/frontend/index.html','text/html'],'/assets/app.js':['apps/frontend/app.js','text/javascript'],'/assets/styles.css':['apps/frontend/styles.css','text/css']};
 if(assets[path]){const [file,type]=assets[path];res.writeHead(200,{'Content-Type':type});return res.end(fs.readFileSync(file));}
 if(path==='/api/v1/deployment')return reply({location:'remote',mode:'mock',transport:'ssh'});
 if(path==='/api/v1/model')return reply({api_version:'1',engine:'synthetic-media-v1',available:true,mock:true,prediction_api_version:'1'});
 if(path==='/ready')return reply({ready:true,engine:'synthetic-media-v1',mock:true});
 if(path===`/api/v1/tasks/${taskId}`)return reply({api_version:'1',task_id:taskId,video_id:uploadId,upload_id:uploadId,target_ratio:[16,9],engine:'synthetic-media-v1',mock:true,status:'succeeded',progress:{stage:'completed',percent:100},error:null,result:{predictions:[]}});
 if(path===`/api/v1/tasks/${taskId}/source`){res.writeHead(200,{'Content-Type':'video/webm','Content-Length':video.length});return res.end(video);}
 let body;
 if(req.method==='POST'){const chunks=[];for await(const chunk of req)chunks.push(chunk);body=JSON.parse(Buffer.concat(chunks));}
 if(path===`/api/v1/tasks/${taskId}/preview`){if(body)previewPosts++;return reply(plan(body?.segments));}
 if(path==='/api/v1/exports'&&body){
  const key=req.headers['idempotency-key'];requests.push({key,request:structuredClone(body)});
  let entry=exportsByKey.get(key);
  if(!entry){entry={id:String(exportsByKey.size+1).padStart(32,'0'),request:structuredClone(body),status:'queued',fail:failNext};failNext=false;exportsByKey.set(key,entry);}
  return reply(item(entry),202);
 }
 const entry=[...exportsByKey.values()].find(e=>path.startsWith(`/api/v1/exports/${e.id}`));
 if(entry){
  if(path.endsWith('/file')){const receipt=JSON.stringify(entry.request);res.writeHead(200,{'Content-Type':'application/octet-stream','Content-Length':Buffer.byteLength(receipt)});return res.end(receipt);}
  entry.status=hold?'running':entry.fail?'failed':'succeeded';return reply(item(entry));
 }
 return reply({error:{code:'not_found',message:'Unknown fixture route'}},404);
});

(async()=>{
 const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1050},acceptDownloads:true}),page=await context.newPage(),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const base=`http://127.0.0.1:${server.address().port}`;
 let downloads=0;page.on('download',()=>downloads++);
 try{
  video=Buffer.from(await page.evaluate(async()=>{
   const canvas=document.createElement('canvas');canvas.width=320;canvas.height=180;const ctx=canvas.getContext('2d'),stream=canvas.captureStream(10),chunks=[];
   const recorder=new MediaRecorder(stream,{mimeType:'video/webm'});recorder.ondataavailable=e=>chunks.push(e.data);const done=new Promise(resolve=>recorder.onstop=resolve);
   let frame=0;const draw=()=>{ctx.fillStyle=`hsl(${frame++*12},60%,45%)`;ctx.fillRect(0,0,320,180);};draw();recorder.start();const timer=setInterval(draw,100);await new Promise(resolve=>setTimeout(resolve,3100));recorder.stop();await done;clearInterval(timer);stream.getTracks().forEach(t=>t.stop());return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
  }));
  await context.addInitScript(({taskId,uploadId})=>{
   if(localStorage.getItem('highlight.studio.workspace.v1.remote.mock'))return;
   const record={execution:'debug',id:taskId,uploadId,fileName:'demo.one.mp4',ratio:[16,9],status:'succeeded',created:new Date().toISOString()};
   localStorage.setItem('highlight.studio.workspace.v1.remote.mock',JSON.stringify({records:[record],activeId:taskId,pending:null}));
  },{taskId,uploadId});
  await page.goto(base);await expect(page.locator('.segment-export')).toHaveCount(3);await page.waitForFunction(()=>!state.busy);
  for(let i=1;i<=3;i++)await page.getByRole('checkbox',{name:`选择片段 ${i}`,exact:true}).uncheck();
  await expect(page.locator('#export-button')).toBeDisabled();await expect(page.locator('#export-count')).toHaveText('0 段');
  hold=true;const firstDownload=page.waitForEvent('download');await page.getByRole('button',{name:'单独导出高光 2 为 MP4',exact:true}).click();
  await expect(page.locator('#export-state')).toContainText('高光 02');
  await expect(page.locator('.segment-export').first()).toBeDisabled();await expect(page.locator('#export-button')).toBeDisabled();
  await expect(page.locator('#export-format')).toBeDisabled();
  assert.deepEqual(requests[0].request.segments,[{...segments[1],enabled:true}]);assert.equal(requests[0].request.format,'mp4');
  hold=false;const single=await firstDownload;assert(single.suggestedFilename().includes('高光 02'));assert(single.suggestedFilename().endsWith('.mp4'));
  await expect(page.locator('#download')).toContainText('高光 02');
  assert.equal(await page.locator('.segment-enabled:checked').count(),0,'Single export must preserve the multi-selection');
  assert.equal(downloads,1);await page.reload();await expect(page.locator('#download')).toContainText('高光 02');
  assert.equal(downloads,1,'Restoring an export must not start an unsolicited download');
  console.log('PASS independent MP4 export, unchecked segment, auto download, serial controls and refresh');

  await page.getByRole('checkbox',{name:'选择片段 1',exact:true}).check();await page.getByRole('checkbox',{name:'选择片段 3',exact:true}).check();
  await page.getByRole('spinbutton',{name:'片段 1 终点 / 秒',exact:true}).fill('0.5');
  const previewsBefore=previewPosts,combinedDownload=page.waitForEvent('download');await page.locator('#export-button').click();
  const combined=await combinedDownload;assert(combined.suggestedFilename().includes('高光拼接-2段'));
  assert.equal(previewPosts,previewsBefore+1,'One-click export applies edits before creating the snapshot');
  assert.deepEqual(requests[1].request.segments,[{...segments[0],end_frame_exclusive:5}, {...segments[1],enabled:false},segments[2]]);
  await expect(page.locator('#export-duration')).toHaveText('1.00 秒');await expect(page.locator('#export-count')).toHaveText('2 段');
  assert.equal(await page.locator('#apply-edits').isDisabled(),true);console.log('PASS selected non-adjacent clips concatenate in source order with edited bounds');

  const beforeInvalid=requests.length;await page.getByRole('spinbutton',{name:'片段 1 起点 / 秒',exact:true}).fill('3');await page.locator('#export-button').click();
  await expect(page.locator('#global-message')).toBeVisible();assert.equal(requests.length,beforeInvalid,'Invalid edits must not create an export');
  await page.getByRole('spinbutton',{name:'片段 1 起点 / 秒',exact:true}).fill('0.1');

  // Commit the server-side job, then drop the response at the browser boundary.
  await page.locator('#export-format').selectOption('jsonl');
  await page.route('**/api/v1/exports',async route=>{await route.fetch();await route.abort();},{times:1});
  await page.getByRole('button',{name:'单独导出高光 3 为 MP4',exact:true}).click();
  await expect(page.locator('#export-button')).toHaveText('↻ 重试上次导出');await expect(page.locator('#export-button')).toBeEnabled();
  await expect(page.locator('.segment-export').first()).toBeDisabled();const ambiguous=requests.at(-1);assert.equal(ambiguous.request.format,'mp4','Individual exports always produce video');
  await page.reload();await expect(page.locator('#export-button')).toHaveText('↻ 重试上次导出');await page.waitForFunction(()=>!state.busy);
  const retryDownload=page.waitForEvent('download');await page.locator('#export-button').click();const retried=await retryDownload;
  assert(retried.suggestedFilename().includes('高光 03'));assert.deepEqual(requests.at(-1),ambiguous);assert.equal(exportsByKey.size,3);
  console.log('PASS ambiguous export retry after reload preserves scope, request and idempotency key');

  failNext=true;const beforeFailure=downloads;await page.getByRole('button',{name:'单独导出高光 1 为 MP4',exact:true}).click();
  await page.waitForFunction(()=>state.exporting?.status==='failed');await expect(page.locator('#download')).toBeHidden();
  await expect(page.locator('.segment-export').first()).toBeEnabled();assert.equal(downloads,beforeFailure);
  const failedKey=requests.at(-1).key,afterFailure=page.waitForEvent('download');await page.getByRole('button',{name:'单独导出高光 1 为 MP4',exact:true}).click();await afterFailure;
  assert.notEqual(requests.at(-1).key,failedKey);console.log('PASS failed export unlocks and a retry creates a new snapshot');

  await page.locator('#export-format').selectOption('mp4');
  for(const width of [1440,1024,900,740,390]){
   await page.setViewportSize({width,height:1050});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No horizontal overflow at ${width}px`);
   assert(await page.locator('.segment-row').evaluateAll(rows=>rows.every(row=>row.querySelector('.segment-actions').getBoundingClientRect().right<=row.getBoundingClientRect().right+1)),`Export actions stay inside the row at ${width}px`);
  }
  await page.setViewportSize({width:390,height:844});await page.locator('.segments-panel').scrollIntoViewIfNeeded();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Segment actions must fit a narrow screen');
  fs.mkdirSync('runtime/segment-export',{recursive:true});await page.locator('.segments-panel').screenshot({path:'runtime/segment-export/mobile.png'});
  await page.setViewportSize({width:1440,height:1050});await page.locator('.segments-panel').screenshot({path:'runtime/segment-export/desktop.png'});
  assert.deepEqual(errors,[]);console.log('PASS narrow layout and no browser errors');
 }finally{await context.close();await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
