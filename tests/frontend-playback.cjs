/* Real browser playback; generated video and preview fixtures never invoke a model. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {expect}=require(process.env.PLAYWRIGHT_MODULE?`${process.env.PLAYWRIGHT_MODULE}/test`:'@playwright/test');
const base=process.env.FRONTEND_URL||'http://127.0.0.1:8000';

async function installPlan(page,segments){
  await page.evaluate(segments=>{
    const frames=segments.filter(s=>s.enabled).flatMap(s=>Array.from({length:s.end_frame_exclusive-s.start_frame},(_,i)=>({frame:s.start_frame+i,crop:[0,0,160,90]})));
    state.plan={metadata:{width:320,height:180,fps_num:20,fps_den:1,frame_count:60,duration_seconds:3,has_audio:false},segments,suggestions:segments,frame_crops:frames,output:{width:160,height:90},duration_seconds:frames.length/20};
    state.cropMap=new Map(frames.map(f=>[f.frame,f.crop]));renderPlan();renderControls();
  },segments);
}
const segment=(id,start,end,enabled=true)=>({suggestion_id:String(id),start_frame:start,end_frame_exclusive:end,enabled});
async function stoppedOn(page,frame){
  await page.waitForFunction(frame=>{
    const v=document.getElementById('video');
    return v.paused&&!v.seeking&&Math.abs(v.currentTime-frame/20)<.025;
  },frame,{timeout:4000});
  await expect(page.locator('#crop-canvas')).toBeVisible();
  await expect(page.locator('#crop-placeholder')).toBeHidden();
}

(async()=>{
 const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'msedge',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1050}});
 if(process.env.NO_VIDEO_FRAME_CALLBACK)await context.addInitScript(()=>{HTMLVideoElement.prototype.requestVideoFrameCallback=undefined;delete HTMLVideoElement.prototype.requestVideoFrameCallback;});
 const page=await context.newPage(),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 try{
  await page.goto(base);await expect(page.locator('#service-label')).toHaveText('服务已连接');
  const bytes=await page.evaluate(async()=>{
    const canvas=document.createElement('canvas');canvas.width=320;canvas.height=180;
    const ctx=canvas.getContext('2d'),stream=canvas.captureStream(20),chunks=[];
    const recorder=new MediaRecorder(stream,{mimeType:'video/webm'});
    recorder.ondataavailable=e=>chunks.push(e.data);const done=new Promise(resolve=>recorder.onstop=resolve);
    let frame=0;const draw=()=>{ctx.fillStyle=`hsl(${frame++*7},60%,40%)`;ctx.fillRect(0,0,320,180);ctx.fillStyle='white';ctx.fillText(String(frame),40,40);};
    draw();recorder.start();const timer=setInterval(draw,50);await new Promise(resolve=>setTimeout(resolve,3200));recorder.stop();await done;clearInterval(timer);stream.getTracks().forEach(t=>t.stop());
    return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
  });
  await page.locator('#file-input').setInputFiles({name:'playback.webm',mimeType:'video/webm',buffer:Buffer.from(bytes)});
  await page.waitForFunction(()=>document.getElementById('video').readyState>=2);
  await installPlan(page,[segment(0,2,8),segment(1,22,30)]);
  await page.locator('[aria-label="预览片段 1"]').click();await page.locator('#crop-view').click();
  await page.locator('#play-toggle').click();await stoppedOn(page,7);
  console.log('PASS clip boundary: pause on final cropped frame, without black gap');
  await expect(page.locator('#timeline-playhead')).toBeVisible();
  let position=await page.locator('#timeline-playhead').getAttribute('aria-valuenow');
  assert(Math.abs(Number(position)-.35)<.03,'Timeline pointer follows actual playback time');
  await page.locator('#play-toggle').click();await stoppedOn(page,29);
  console.log('PASS resume: play the next highlight and pause again');

  // Pointer dragging pauses playback and seeks; keyboard is an equivalent accessible control.
  const track=await page.locator('#timeline').boundingBox(),pointer=await page.locator('#timeline-playhead').boundingBox();
  await page.mouse.move(pointer.x+pointer.width/2,pointer.y+pointer.height/2);await page.mouse.down();
  await page.mouse.move(track.x+track.width*.4,track.y+track.height/2,{steps:8});await page.mouse.up();
  await page.waitForFunction(()=>!document.getElementById('video').seeking);
  assert(Math.abs(await page.locator('#video').evaluate(v=>v.currentTime)-1.2)<.06,'Dragging seeks to the requested source time');
  assert(await page.locator('#video').evaluate(v=>v.paused));
  await page.locator('#timeline-playhead').focus();await page.keyboard.press('ArrowRight');
  await page.waitForFunction(()=>!document.getElementById('video').seeking);
  assert(Math.abs(await page.locator('#video').evaluate(v=>v.currentTime)-1.25)<.06);
  console.log('PASS draggable and keyboard timeline pointer');

  await installPlan(page,[segment(0,2,8),segment(1,8,14)]);
  await page.locator('[aria-label="预览片段 1"]').click();await page.locator('#play-toggle').click();await stoppedOn(page,7);
  await page.locator('#play-toggle').click();await stoppedOn(page,13);
  console.log('PASS adjacent highlights pause separately');
  await installPlan(page,[segment(0,10,11)]);
  await page.locator('[aria-label="预览片段 1"]').click();await page.locator('#play-toggle').click();await stoppedOn(page,10);
  console.log('PASS single-frame highlight stays visible');

  await installPlan(page,[segment(0,2,8),segment(1,22,30,false)]);
  await page.locator('#source-view').click();await page.locator('[aria-label="预览片段 1"]').click();
  await page.locator('#play-toggle').click();await page.waitForFunction(()=>document.getElementById('video').currentTime>.6);
  assert.equal(await page.locator('#video').evaluate(v=>v.paused),false,'Source preview remains continuous');
  await page.locator('#play-toggle').click();await page.locator('#crop-view').click();await page.locator('#play-toggle').click();await stoppedOn(page,7);
  await page.locator('#play-toggle').click();await stoppedOn(page,7);
  console.log('PASS original playback and disabled highlight handling');

  await installPlan(page,[segment(0,2,8),segment(1,22,30)]);
  await page.setViewportSize({width:390,height:844});await page.locator('#timeline-playhead').scrollIntoViewIfNeeded();
  await expect(page.locator('#timeline-playhead')).toBeVisible();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  fs.mkdirSync('runtime/preview-playback',{recursive:true});
  await page.locator('.timeline-panel').screenshot({path:'runtime/preview-playback/mobile.png'});
  await page.setViewportSize({width:1440,height:1050});await page.locator('.timeline-panel').screenshot({path:'runtime/preview-playback/desktop.png'});
  assert.deepEqual(errors,[]);console.log('PASS mobile layout and browser errors');
 }finally{await context.close();await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
