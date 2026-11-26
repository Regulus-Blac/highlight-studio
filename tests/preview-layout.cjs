/* Exercise actual large video metadata without uploading or invoking a model. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.FRONTEND_URL || 'http://127.0.0.1:8000';

async function createVideo(page,width,height){
  return Buffer.from(await page.evaluate(async({width,height})=>{
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
    const ctx=canvas.getContext('2d');ctx.fillStyle='#244879';ctx.fillRect(0,0,width,height);
    const colors=['#ff4444','#33dd88','#4488ff','#ffdd33'];
    for(let i=0;i<4;i++){ctx.fillStyle=colors[i];ctx.fillRect(i%2?(width-80):0,i>1?(height-80):0,80,80);}
    const stream=canvas.captureStream(5),chunks=[],recorder=new MediaRecorder(stream,{mimeType:'video/webm'});
    recorder.ondataavailable=event=>chunks.push(event.data);
    const done=new Promise(resolve=>recorder.onstop=resolve);
    recorder.start();
    const redraw=setInterval(()=>{ctx.fillStyle='#244879';ctx.fillRect(width/2,height/2,12,12);},60);
    await new Promise(resolve=>setTimeout(resolve,600));recorder.stop();await done;clearInterval(redraw);
    stream.getTracks().forEach(track=>track.stop());
    return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
  },{width,height}));
}

async function assertContained(page,selector){
  const layout=await page.evaluate(selector=>{
    const parent=document.querySelector('.stage'),media=document.querySelector(selector);
    const rect=element=>{const r=element.getBoundingClientRect();return{left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height};};
    return{parent:rect(parent),media:rect(media),fit:getComputedStyle(media).objectFit};
  },selector);
  assert.equal(layout.fit,'contain','Previews must show the full image');
  for(const edge of ['left','top'])assert(layout.media[edge]>=layout.parent[edge]-1,`${selector} ${edge} escapes its viewer: ${JSON.stringify(layout)}`);
  for(const edge of ['right','bottom'])assert(layout.media[edge]<=layout.parent[edge]+1,`${selector} ${edge} is clipped by its viewer: ${JSON.stringify(layout)}`);
  assert(layout.media.width>0&&layout.media.height>0,'Preview must occupy a visible area');
}

(async()=>{
  const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'msedge',headless:true});
  fs.mkdirSync('runtime/preview-layout',{recursive:true});
  try{
    for(const [width,height] of [[1920,1080],[1080,1920],[1080,1080]]){
      const context=await browser.newContext({viewport:{width:1440,height:1050}});
      const page=await context.newPage();await page.goto(base);
      const bytes=await createVideo(page,width,height);
      await page.locator('#file-input').setInputFiles({name:`${width}x${height}.webm`,mimeType:'video/webm',buffer:bytes});
      await page.waitForFunction(width=>document.getElementById('video').videoWidth===width,width,{timeout:10000});
      for(const viewport of [{width:1440,height:1050},{width:390,height:844}]){
        await page.setViewportSize(viewport);
        await assertContained(page,'#video');
        await page.evaluate(()=>{
          const video=document.getElementById('video');
          state.plan={metadata:{fps_num:5,fps_den:1,frame_count:5}};
          state.cropMap=new Map([[0,[0,0,video.videoWidth,video.videoHeight]]]);state.crop=true;video.currentTime=0;drawCrop();
        });
        await assertContained(page,'#crop-canvas');
        await page.locator('.viewer-panel').screenshot({path:`runtime/preview-layout/${width}x${height}-${viewport.width}.png`});
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Layout must fit the screen');
        await page.evaluate(()=>{state.crop=false;document.getElementById('crop-canvas').hidden=true;});
      }
      console.log(`PASS full-frame source and crop preview: ${width}x${height}, desktop and mobile`);
      await context.close();
    }
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
