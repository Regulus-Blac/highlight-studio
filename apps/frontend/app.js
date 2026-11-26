'use strict';
const $ = (id) => document.getElementById(id);
let STORE = 'highlight.studio.workspace.v1';
const EXECUTION_STORE='highlight.studio.execution.v1';
let execution='cloud';
try{const saved=localStorage.getItem(EXECUTION_STORE);if(['cloud','local','debug'].includes(saved))execution=saved;}catch{}
const ACTIVE = new Set(['queued', 'running']);
const names = {queued:'排队等待',running:'正在分析',succeeded:'分析完成',failed:'分析失败',cancelled:'已取消'};
const state = {file:null, url:null, ratio:[9,16], model:null, ready:false, task:null, plan:null, record:null, exporting:null, autoDownloadId:null, dirty:false, busy:false, epoch:0, crop:false, cropMap:new Map(), previewSegment:null, previewEnded:false, records:[], pending:null};
let taskTimer, exportTimer, playbackAnimation, timelinePointerId=null;
const localHost = ['127.0.0.1','localhost','[::1]'].includes(location.hostname);
let deployment = {location:'local'};
const remoteService = () => deployment.location === 'remote';
const canRestartTask = () => ['failed','cancelled'].includes(state.task?.status);
const needsNewSource = () => state.task?.status==='failed'&&['invalid_media','unsupported_media','unsupported_media_profile','source_changed'].includes(state.task.error?.code);
const inputsLocked = () => state.busy||!!state.pending||!!state.task&&!canRestartTask();
function canProcess(){
  if(!localHost||!state.ready||!state.model)return false;
  if(remoteService())return execution==='debug'?state.model.mock===true:execution==='cloud'&&state.model.mock===false;
  return execution==='debug'?state.model.mock===true:execution==='local'&&state.model.mock===false;
}
function renderExecution(){
  const locked=inputsLocked();
  document.querySelectorAll('[data-execution]').forEach(button=>{
    const selected=button.dataset.execution===execution;
    button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));button.disabled=locked;
  });
  $('debug-mode').disabled=locked||!state.model?.mock;
  $('debug-mode').hidden=!state.model?.mock;
  $('debug-mode').setAttribute('aria-pressed',String(execution==='debug'));
  $('debug-mode').textContent=execution==='debug'?'正在体验演示模式':'体验演示模式';
  $('execution-badge').textContent={cloud:'云端',local:'本机',debug:'演示'}[execution];
  const configured=!remoteService()&&localHost&&state.model?.mock===false&&state.ready;
  const cloudReady=remoteService()&&state.ready;
  $('cloud-availability').textContent=cloudReady?'可以使用':'暂不可用';
  $('cloud-status').textContent=cloudReady?'可以使用':'暂不可用';
  $('local-availability').textContent=configured?'可以使用':'暂未开放';
  $('local-service-status').textContent=configured?'可以使用':'暂未开放';
  $('execution-detail').textContent=execution==='debug'
    ?'演示模式仅供体验操作流程，生成的片段不代表真实分析结果。'
    :execution==='local'?'使用这台电脑处理视频，素材保留在本机。'
    :cloudReady?'服务已配置。选择视频和画幅，即可开始分析。':'云端服务暂不可用，可以先选择视频预览。';
  $('privacy-title').textContent=execution==='cloud'?'安心创作':'本机处理';
  $('privacy-copy').textContent=execution==='cloud'?'素材加密传输，专注每一段精彩。':'视频在这台电脑上处理。';
  $('analysis-step-copy').textContent=execution==='debug'?'体验片段生成流程':'分析片段与推荐构图';
  $('mode-notice').parentElement.hidden=!state.model?.mock;
  $('cloud-guide').textContent=cloudReady?'服务已配置，直接选择视频开始即可。':'云端服务暂不可用，请稍后重试。';
  $('storage-guide').textContent='任务记录保存在当前浏览器，可随时返回继续编辑。';
}

function chooseExecution(mode){
  if(inputsLocked())return;
  execution=mode;try{localStorage.setItem(EXECUTION_STORE,mode);}catch{}
  clearMessage();renderControls();
}
function uid(){ return crypto.randomUUID(); }
function time(value){
  if(!Number.isFinite(Number(value)))return '—';
  const ticks=Math.round(Math.max(0,Number(value))*100);
  return `${String(Math.floor(ticks/6000)).padStart(2,'0')}:${String(Math.floor(ticks/100)%60).padStart(2,'0')}.${String(ticks%100).padStart(2,'0')}`;
}
function basicMock(){return state.task?.result?.mock===true&&state.task.result.engine==='mock-v1';}
function connectSource(path){
  if(state.url)return;
  $('video').src=safePath(path);$('video').hidden=false;$('empty-state').hidden=true;
  $('video-caption').hidden=false;$('video-name').textContent=state.record.fileName;
}
function size(value){ return value>=1048576?`${(value/1048576).toFixed(1)} MiB`:`${(value/1024).toFixed(1)} KiB`; }
function message(text){ $('message-text').textContent=text; $('global-message').hidden=false; }
function clearMessage(){ $('global-message').hidden=true; }
const errors = {network_error:'服务连接中断，请检查网络后重试。',model_unavailable:'分析服务尚未就绪，请稍后重试。',runner_unavailable:'分析服务暂不可用，请稍后重试。',media_tools_unavailable:'预览与导出服务暂不可用，请稍后重试。',mock_media_disabled:'当前演示模式不生成高光或成片。',mock_export_disabled:'当前演示模式不提供文件导出。',unsupported_media:'此媒体测试版本支持不超过 10 分钟、4K、120 fps 的单视频流素材。',unsupported_media_profile:'这段视频的旋转信息、像素比例或音轨起点暂不受支持。请重新选择视频，或重新导出后再导入。',invalid_media:'视频文件无法读取，可能不完整或已损坏。请确认原视频能正常播放后重新选择。',request_too_large:'视频超过服务允许的文件大小，请选择较小文件。',inference_failed:'视频分析失败，请稍后重试或更换视频。',worker_interrupted:'上次处理被中断，可重新创建任务。',no_enabled_segments:'没有选中的高光帧，可改为导出 JSONL。',export_active:'当前已有导出任务，请等待完成后重试。',invalid_segments:'片段时间必须在视频范围内，按原视频顺序排列且不能重叠。'};
function describe(error){ const code=error.code||'network_error'; return errors[code]||'操作未完成，请稍后重试。'; }
function safePath(path){ const url=new URL(path,location.origin); if(url.origin!==location.origin||!url.pathname.startsWith('/api/v1/')) throw new Error('服务返回了非本机资源地址'); return url.href; }
async function api(path, options={}){
  if(!localHost) throw Object.assign(new Error('此开发页面仅允许连接本机服务。'),{code:'local_only'});
  let response;
  try{ response=await fetch(path,{...options,signal:AbortSignal.timeout(150000)}); }catch{ throw Object.assign(new Error('服务连接中断'),{code:'network_error'}); }
  let body; try{body=await response.json();}catch{throw Object.assign(new Error('服务未返回有效数据'),{code:'invalid_response'});}
  if(!response.ok) throw Object.assign(new Error(body.error?.message||`HTTP ${response.status}`),{code:body.error?.code||'http_error',status:response.status});
  return body;
}
function post(path,body,key){return api(path,{method:'POST',headers:{'Content-Type':'application/json',...(key?{'Idempotency-Key':key}:{})},body:JSON.stringify(body)});}
function persist(){
  try{localStorage.setItem(STORE,JSON.stringify({records:state.records.slice(0,20),activeId:state.record?.id||null,pending:state.pending}));}
  catch{message('浏览器无法保存任务记录，刷新前请保留当前页面。');}
}
function saveRecord(){if(state.record){ const index=state.records.findIndex(r=>r.id===state.record.id); if(index>=0)state.records[index]={...state.record};else state.records.unshift({...state.record}); state.records=state.records.slice(0,20); }persist();renderHistory();}
function view(name){for(const el of document.querySelectorAll('.view')) el.hidden=el.id!==`${name}-view`;for(const el of document.querySelectorAll('[data-view]')){el.classList.toggle('active',el.dataset.view===name);el.setAttribute('aria-current',el.dataset.view===name?'page':'false');}$('page-name').textContent={editor:'剪辑工作台',history:'任务记录',settings:'使用设置'}[name];}
function renderHistory(){
  $('history-count').textContent=String(state.records.length); const list=$('history-list');list.replaceChildren();
  if(!state.records.length){const empty=document.createElement('div');empty.className='panel empty-list';empty.textContent='还没有任务。导入第一段视频，开始创作。';list.append(empty);return;}
  for(const r of state.records){const b=document.createElement('button');b.className='history-card';const info=document.createElement('div'),title=document.createElement('strong'),sub=document.createElement('small'),badge=document.createElement('span');title.textContent=r.fileName;sub.textContent=`${r.ratio.join(':')} · ${new Date(r.created).toLocaleString('zh-CN')}`;badge.textContent=names[r.status]||'继续查看';info.append(title,sub);b.append(info,badge);b.disabled=state.busy;b.onclick=()=>{if(state.dirty&&!confirm('切换任务会舍弃尚未应用的片段调整，是否继续？'))return;restore(r);};list.append(b);}
}
function reset(){
  ++state.epoch;clearTimeout(taskTimer);clearTimeout(exportTimer);$('video').pause();if(state.url)URL.revokeObjectURL(state.url);$('video').removeAttribute('src');$('video').load();
  Object.assign(state,{file:null,url:null,task:null,plan:null,record:null,exporting:null,autoDownloadId:null,dirty:false,busy:false,crop:false,cropMap:new Map(),previewSegment:null,previewEnded:false,pending:null,ratio:[9,16]});
  $('video').hidden=true;$('empty-state').hidden=false;$('video-caption').hidden=true;$('crop-canvas').hidden=true;$('crop-placeholder').hidden=true;$('file-input').value='';$('video-meta').textContent='尚未导入素材';$('file-summary').textContent='选择视频后即可开始';$('task-state').textContent='等待导入视频';$('task-state').classList.remove('busy');$('export-state').textContent='';$('export-state').classList.remove('busy');$('download').hidden=true;$('upload-progress').hidden=true;$('playhead').textContent='00:00.00';$('resume-task').hidden=true;$('source-view').classList.add('selected');$('crop-view').classList.remove('selected');$('source-view').setAttribute('aria-pressed','true');$('crop-view').setAttribute('aria-pressed','false');clearMessage();renderPlan();renderControls();persist();view('editor');
}
function renderControls(){
  renderExecution();
  const locked=inputsLocked();
  $('start-analysis').disabled=!state.file||locked||needsNewSource()||!canProcess();
  $('choose-file').disabled=locked; $('new-project').disabled=state.busy;
  $('choose-file').textContent=canRestartTask()?'＋ 重新选择视频':'＋ 选择视频';
  $('replace-file').hidden=!state.file&&!canRestartTask();$('replace-file').disabled=locked;
  $('resume-task').disabled=state.busy;
  document.querySelectorAll('.history-card').forEach(b=>b.disabled=state.busy);
  $('start-analysis').textContent=needsNewSource()?'请重新选择视频':execution==='debug'?'✦ 体验分析':canProcess()?'✦ 开始分析':'服务暂不可用';
  document.querySelectorAll('[data-ratio]').forEach(b=>{b.disabled=locked;const chosen=b.dataset.ratio===state.ratio.join(',');b.classList.toggle('selected',chosen);b.setAttribute('aria-pressed',String(chosen));});
  $('export-ratio').textContent=state.ratio.join(':');
  $('cancel-task').hidden=state.task?.status!=='queued';$('retry-task').hidden=needsNewSource()||!canRestartTask()&&!state.pending;
  $('retry-task').disabled=state.busy||!canProcess();$('crop-view').disabled=!state.plan?.frame_crops.length;
  $('prev-frame').disabled=$('next-frame').disabled=!state.plan;
  $('play-toggle').disabled=$('seek-video').disabled=!$('video').getAttribute('src');
  const exportActive=state.exporting&&ACTIVE.has(state.exporting.status);
  const pendingExport=!!state.record?.pendingExport;
  $('export-format').disabled=!!exportActive||state.busy||pendingExport;
  let selected=state.plan?.segments||[];
  if(state.dirty){try{selected=readEdits();}catch{}}
  const enabled=selected.filter(s=>s.enabled),empty=!enabled.length;
  $('export-count').textContent=state.plan?`${enabled.length} 段`:'—';
  $('export-duration').textContent=state.plan?`${(enabled.reduce((sum,s)=>sum+s.end_frame_exclusive-s.start_frame,0)*state.plan.metadata.fps_den/state.plan.metadata.fps_num).toFixed(2)} 秒`:'—';
  $('export-resolution').textContent=state.dirty?'应用后确定':state.plan?.output?`${state.plan.output.width} × ${state.plan.output.height}`:'—';
  $('export-button').disabled=!state.plan||state.busy||!!exportActive||(!pendingExport&&$('export-format').value==='mp4'&&empty);
  $('export-button').textContent=pendingExport?'↻ 重试上次导出':$('export-format').value==='mp4'?'↓ 拼接导出已选片段':'↓ 导出已选片段 JSONL';
  $('export-hint').textContent=basicMock()?'当前演示不生成高光或成片。':pendingExport?'上次创建结果尚未确认。点击重试将继续同一份导出，不会重复创建。':!state.plan?'完成分析和预览后即可导出。':empty?'请勾选要拼接的片段，或点击片段旁的「单独导出」。空选择仍可导出 JSONL。':'按原视频顺序拼接，导出前自动应用调整；文件生成后自动下载，也可点击下载按钮重试。';
  for(const [id,on] of [['step-import',!state.task],['step-analyze',!!state.task&&state.task.status!=='succeeded'],['step-export',state.task?.status==='succeeded']])$(id).classList.toggle('current',on);
  $('reset-segments').disabled=!state.plan||state.busy||!!exportActive||pendingExport;$('apply-edits').disabled=!state.dirty||state.busy||!!exportActive||pendingExport;
  document.querySelectorAll('.segment-row input').forEach(el=>el.disabled=state.busy||!!exportActive||pendingExport);
  document.querySelectorAll('.segment-export').forEach(el=>el.disabled=!state.record||state.busy||!!exportActive||pendingExport);
}
async function checkService(){
  try { deployment=await api('/api/v1/deployment'); } catch { deployment={location:'local'}; }
  try{
    const model=await api('/api/v1/model');
    const health=await api('/ready').catch(()=>({ready:false}));
    state.model=model;state.ready=!!model.available&&!!health.ready;
    if(remoteService()&&!state.task&&!state.pending)execution=model.mock?'debug':'cloud';
    $('service-dot').className=`status-dot ${state.ready?'online':'offline'}`;
    $('service-label').textContent=state.ready?'服务已连接':'服务暂不可用';
    $('mode-notice').textContent=model.mock?'演示模式：仅供体验操作流程，不进行真实内容分析。':'';
    $('mode-explanation').textContent=model.mock
      ?'演示模式可以体验导入、任务与编辑操作；不会调用 AI 分析视频内容。'
      :'自动发现高光，支持片段调整、裁剪预览与成片下载。';
    $('settings-engine').textContent=model.mock?'演示分析':'智能分析';
    $('settings-mode').textContent=model.mock?'演示':remoteService()?'云端处理':'本机处理';
    $('settings-ready').textContent=state.ready?'可以使用':'暂不可用';
  }catch(error){
    state.ready=false;$('service-dot').className='status-dot offline';$('service-label').textContent='连接已中断';
    $('mode-notice').textContent=describe(error);$('settings-ready').textContent='连接失败';
    $('mode-explanation').textContent='暂时无法连接服务，请重新检查连接。';
  }
  renderControls();
}

function selectFile(file){
  if(!file||inputsLocked())return;
  if(!/\.(mp4|mov|mkv|webm|avi)$/i.test(file.name)){message('请选择 MP4、MOV、MKV、WEBM 或 AVI 视频文件。');return;}
  if(!file.size||file.size>512*1024*1024){message('视频不能为空，且大小不能超过 512 MiB。');return;}
  if(canRestartTask()||state.record&&!state.task){const ratio=[...state.ratio];reset();state.ratio=ratio;}
  clearMessage();if(state.url)URL.revokeObjectURL(state.url);state.file=file;state.url=URL.createObjectURL(file);$('video').src=state.url;$('video').hidden=false;$('empty-state').hidden=true;$('video-caption').hidden=false;$('video-name').textContent=file.name;$('file-summary').textContent=`${file.name} · ${size(file.size)}`;$('video-meta').textContent='正在读取视频信息…';$('task-state').textContent='素材已就绪，选择画幅后开始分析';renderControls();
}
function upload(file){return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest();xhr.open('POST',`/api/v1/uploads?filename=${encodeURIComponent(file.name)}`);xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.timeout=600000;
  xhr.upload.onprogress=e=>{if(e.lengthComputable){const pct=Math.round(e.loaded/e.total*100);$('upload-progress').value=pct;$('task-state').textContent=pct===100?'正在上传并保存素材…':`正在${remoteService()?'上传远程':'导入本机'} · ${pct}%`;}};
  xhr.onload=()=>{let body;try{body=JSON.parse(xhr.responseText);}catch{reject(Object.assign(new Error('上传响应无效'),{code:'invalid_response'}));return;}if(xhr.status>=200&&xhr.status<300)resolve(body);else reject(Object.assign(new Error(body.error?.message),{code:body.error?.code}));};
  xhr.onerror=xhr.ontimeout=()=>reject(Object.assign(new Error('导入连接中断'),{code:'network_error'}));xhr.send(file);
});}
async function start(){
  if(state.busy||needsNewSource()||!canProcess())return;state.busy=true;clearMessage();renderControls();renderHistory();
  try{
    if(!state.pending){let uploadId=state.record?.uploadId;if(!uploadId){if(!state.file)throw new Error('请先选择视频');$('upload-progress').hidden=false;const result=await upload(state.file);uploadId=result.upload_id;}
      state.pending={execution,uploadId,ratio:[...state.ratio],fileName:state.file?.name||state.record.fileName,key:uid()};persist();}
    $('upload-progress').hidden=true;$('task-state').textContent='正在创建分析任务…';
    const pending=state.pending;const task=await post('/api/v1/tasks',{api_version:'1',upload_id:pending.uploadId,target_ratio:pending.ratio},pending.key);
    state.record={execution:pending.execution||execution,id:task.task_id,uploadId:pending.uploadId,fileName:pending.fileName,ratio:pending.ratio,created:task.created_at,status:task.status};state.pending=null;state.plan=null;state.exporting=null;state.dirty=false;state.record.edits=null;saveRecord();renderPlan();state.busy=false;handleTask(task);pollTask(state.epoch);
  }catch(error){message(describe(error));$('task-state').textContent=state.pending?'任务创建未确认，重试将复用同一请求。':'导入未完成，可重新开始。';}
  finally{state.busy=false;$('upload-progress').hidden=true;renderControls();renderHistory();}
}
function handleTask(task){state.task=task;state.record.status=task.status;saveRecord();$('task-state').classList.toggle('busy',ACTIVE.has(task.status));$('task-state').textContent=names[task.status]||task.status;
  if(task.status==='running')$('task-state').textContent='正在分析视频，请稍候…';
  if(task.status==='succeeded'){$('task-state').textContent=task.result?.mock?'演示已完成 · 未进行真实分析':task.result?.predictions.length?'高光分析完成':'分析完成 · 未检测到高光';}
  if(task.status==='failed'){$('task-state').textContent=describe(task.error);}
  renderControls();
}
async function pollTask(epoch){
  clearTimeout(taskTimer);if(epoch!==state.epoch||!state.task)return;
  if(!ACTIVE.has(state.task.status)){if(state.task.status==='succeeded'&&!state.plan)await loadPlan(epoch);return;}
  taskTimer=setTimeout(async()=>{try{const task=await api(`/api/v1/tasks/${state.task.task_id}`);if(epoch!==state.epoch)return;handleTask(task);pollTask(epoch);}catch(error){if(epoch!==state.epoch)return;message(describe(error));$('resume-task').hidden=false;}},1500);
}
async function loadPlan(epoch){
  if(basicMock()){
    connectSource(`/api/v1/tasks/${state.task.task_id}/source`);
    renderPlan();renderControls();return;
  }
  try{const path=`/api/v1/tasks/${state.task.task_id}/preview`;const plan=state.record.edits?await post(path,{api_version:'1',segments:state.record.edits}):await api(path);if(epoch!==state.epoch)return;state.plan=plan;state.cropMap=new Map(plan.frame_crops.map(f=>[f.frame,f.crop]));state.dirty=false;
    connectSource(plan.source_url);
    $('video-meta').textContent=`${plan.metadata.width} × ${plan.metadata.height} · ${time(plan.metadata.duration_seconds)} · ${plan.metadata.has_audio?'有音轨':'无音轨'}`;renderPlan();renderControls();drawCrop();
  }catch(error){if(epoch!==state.epoch)return;message(describe(error));$('timeline-summary').textContent='预览暂不可用';$('resume-task').hidden=false;}
}
function renderPlan(){
  const plan=state.plan,list=$('segment-list'),timeline=$('timeline-blocks');list.replaceChildren();timeline.replaceChildren();
  if(state.crop)$('video').pause();state.previewSegment=null;state.previewEnded=false;updatePlayhead();
  $('edit-actions').hidden=!plan?.segments.length;$('segment-count').textContent=String(plan?.suggestions.length||0);
  $('timeline-middle').textContent=plan?time(plan.metadata.duration_seconds/2):'—';$('timeline-end').textContent=plan?time(plan.metadata.duration_seconds):'—';
  $('export-count').textContent=plan?`${plan.segments.filter(s=>s.enabled).length} 段`:'—';$('export-duration').textContent=plan?`${plan.duration_seconds.toFixed(2)} 秒`:'—';$('export-resolution').textContent=plan?.output?`${plan.output.width} × ${plan.output.height}`:'—';
  $('timeline-summary').textContent=basicMock()?'演示已完成':plan?`已选 ${plan.segments.filter(s=>s.enabled).length} 段 · ${plan.duration_seconds.toFixed(2)} 秒`:'等待分析';
  if(!plan?.segments.length){const empty=document.createElement('div');empty.className='empty-list';empty.textContent=basicMock()?'演示已完成。当前模式不生成高光片段。':plan?'未发现高光片段，可以换一段视频重试。':'完成分析后，可以选择片段并微调起止时间。';list.append(empty);const note=document.createElement('div');note.className='timeline-empty';note.textContent=basicMock()?'演示模式不生成高光时间轴':plan?'当前没有高光片段':'分析完成后，高光片段将在这里呈现';timeline.append(note);return;}
  const fps=plan.metadata.fps_num/plan.metadata.fps_den;
  for(const [index,segment] of plan.segments.entries()){
    const row=document.createElement('div');row.className='segment-row';row.dataset.sid=segment.suggestion_id;
    const check=document.createElement('input');check.type='checkbox';check.checked=segment.enabled;check.className='segment-enabled';check.setAttribute('aria-label',`选择片段 ${index+1}`);
    const title=document.createElement('div');title.className='segment-title';title.textContent=`高光 ${String(index+1).padStart(2,'0')}`;const duration=document.createElement('small');duration.textContent=`${((segment.end_frame_exclusive-segment.start_frame)/fps).toFixed(2)} 秒`;title.append(duration);
    row.append(check,title);
    for(const [field,label] of [['start_frame','起点 / 秒'],['end_frame_exclusive','终点 / 秒']]){const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('input');input.type='number';input.min='0';input.max=String(plan.metadata.duration_seconds);input.step='any';input.value=String(Number((segment[field]/fps).toFixed(6)));input.dataset.field=field;input.setAttribute('aria-label',`片段 ${index+1} ${label}`);wrap.append(input);row.append(wrap);}
    const actions=document.createElement('div');actions.className='segment-actions';
    const seek=document.createElement('button');seek.className='icon-button';seek.textContent='▷';seek.setAttribute('aria-label',`预览片段 ${index+1}`);seek.onclick=()=>{if(!state.crop||segment.enabled)seekVideo(segment.start_frame/fps);};
    const download=document.createElement('button');download.className='segment-export';download.textContent='单独导出';download.setAttribute('aria-label',`单独导出高光 ${index+1} 为 MP4`);download.onclick=()=>createExport(segment.suggestion_id);
    actions.append(seek,download);row.append(actions);
    row.addEventListener('input',()=>{state.dirty=true;$('download').hidden=true;renderControls();});list.append(row);
    const block=document.createElement('button');block.className=`timeline-block${segment.enabled?'':' off'}`;block.style.left=`${segment.start_frame/plan.metadata.frame_count*100}%`;block.style.width=`${(segment.end_frame_exclusive-segment.start_frame)/plan.metadata.frame_count*100}%`;block.textContent=String(index+1).padStart(2,'0');block.setAttribute('aria-label',`跳转高光 ${index+1}，${time(segment.start_frame/fps)}`);block.onclick=seek.onclick;timeline.append(block);
  }
}
function readEdits(){
  const fps=state.plan.metadata.fps_num/state.plan.metadata.fps_den;let previous=-1;
  return [...document.querySelectorAll('.segment-row')].map(row=>{const a=row.querySelector('[data-field=start_frame]'),b=row.querySelector('[data-field=end_frame_exclusive]');const start=Number(a.value),end=Number(b.value);const from=Math.round(start*fps),to=Math.round(end*fps);
    if(a.value===''||b.value===''||!Number.isFinite(start)||!Number.isFinite(end)||from<0||to<=from||to>state.plan.metadata.frame_count||from<previous)throw Object.assign(new Error('片段时间无效'),{code:'invalid_segments'});previous=to;
    return {suggestion_id:row.dataset.sid,start_frame:from,end_frame_exclusive:to,enabled:row.querySelector('.segment-enabled').checked};});
}
async function saveEdits(edits,epoch){
  const plan=await post(`/api/v1/tasks/${state.task.task_id}/preview`,{api_version:'1',segments:edits});if(epoch!==state.epoch)return false;
  state.plan=plan;state.record.edits=edits;state.record.exportId=null;state.record.exportContext=null;state.exporting=null;state.autoDownloadId=null;
  $('download').hidden=true;$('export-state').textContent='';state.cropMap=new Map(plan.frame_crops.map(f=>[f.frame,f.crop]));state.dirty=false;
  saveRecord();renderPlan();drawCrop();return true;
}
async function applyEdits(resetToSuggested=false){
  if(!state.plan||state.busy||state.record?.pendingExport||ACTIVE.has(state.exporting?.status))return;
  const epoch=state.epoch;state.busy=true;renderControls();clearMessage();
  try{await saveEdits(resetToSuggested?state.plan.suggestions.map(s=>({...s})):readEdits(),epoch);}
  catch(error){if(epoch===state.epoch)message(describe(error));}
  finally{if(epoch===state.epoch){state.busy=false;renderControls();}}
}
function timelineDuration(){return state.plan?.metadata.duration_seconds||(Number.isFinite($('video').duration)?$('video').duration:0);}
function updatePlayhead(){
  const video=$('video'),duration=timelineDuration(),current=Math.max(0,Math.min(duration,video.currentTime||0)),pointer=$('timeline-playhead');
  $('playhead').textContent=time(video.currentTime);$('seek-video').value=String(video.currentTime);
  pointer.hidden=!state.plan?.segments?.length||!video.getAttribute('src');
  pointer.style.left=`${duration?current/duration*100:0}%`;pointer.setAttribute('aria-valuemax',String(duration));pointer.setAttribute('aria-valuenow',String(Number(current.toFixed(3))));pointer.setAttribute('aria-valuetext',`${time(current)} / ${time(duration)}`);
}
function seekVideo(seconds){
  const video=$('video');video.pause();state.previewSegment=null;state.previewEnded=false;
  video.currentTime=Math.max(0,Math.min(timelineDuration(),seconds));drawCrop();
}
function prepareCropPlayback(){
  if(!state.crop||!state.plan)return true;
  const video=$('video'),fps=state.plan.metadata.fps_num/state.plan.metadata.fps_den,segments=state.plan.segments.filter(s=>s.enabled);
  if(!segments.length){video.pause();return false;}
  const frame=Math.floor(video.currentTime*fps+0.0001);
  let segment;
  if(state.previewEnded&&state.previewSegment){
    const index=segments.findIndex(s=>s.suggestion_id===state.previewSegment.suggestion_id);segment=segments[(index+1)%segments.length];
    video.currentTime=segment.start_frame/fps;
  }else{
    segment=segments.find(s=>s.start_frame<=frame&&frame<s.end_frame_exclusive);
    if(!segment){segment=segments.find(s=>s.start_frame>frame)||segments[0];video.currentTime=segment.start_frame/fps;}
  }
  state.previewSegment=segment;state.previewEnded=false;return true;
}
function drawCrop(){
  const video=$('video');updatePlayhead();if(!state.crop||!state.plan)return;
  const fps=state.plan.metadata.fps_num/state.plan.metadata.fps_den,clock=video.currentTime;
  const segment=state.previewSegment;
  if(segment&&(!video.paused||video.ended)&&clock*fps+0.0001>=segment.end_frame_exclusive){
    // End is exclusive. Keep the last cropped frame visible instead of entering the gap.
    video.pause();state.previewEnded=true;video.currentTime=(segment.end_frame_exclusive-1)/fps;updatePlayhead();return;
  }
  const frame=Math.min(state.plan.metadata.frame_count-1,Math.floor(clock*fps+0.0001)),box=state.cropMap.get(frame);
  $('crop-placeholder').hidden=!!box;$('crop-canvas').hidden=!box;if(!box||video.readyState<2||video.seeking)return;
  const canvas=$('crop-canvas');if(canvas.width!==box[2])canvas.width=box[2];if(canvas.height!==box[3])canvas.height=box[3];canvas.getContext('2d').drawImage(video,...box,0,0,canvas.width,canvas.height);
}
function videoFrame(){drawCrop();$('video').requestVideoFrameCallback(videoFrame);}
function playbackTick(){drawCrop();if(!$('video').paused)playbackAnimation=requestAnimationFrame(playbackTick);}
function exportContext(suggestionId,segments,format){
  const index=state.plan.segments.findIndex(s=>s.suggestion_id===suggestionId),single=suggestionId!==null;
  const label=single?`高光 ${String(index+1).padStart(2,'0')}`:`已选 ${segments.filter(s=>s.enabled).length} 段高光`;
  const source=(state.record.fileName||'highlight').replace(/\.[^.]+$/,'').replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').slice(0,80);
  return {kind:single?'single':'selection',suggestionId,label,filename:`${source}-${single?label:`高光拼接-${segments.filter(s=>s.enabled).length}段`}.${format}`};
}
async function createExport(suggestionId=null){
  if(!state.plan||!state.record||state.busy||ACTIVE.has(state.exporting?.status))return;
  const epoch=state.epoch;state.busy=true;renderControls();clearMessage();
  try{
    let pending=state.record.pendingExport;
    if(!pending){
      const edits=state.dirty?readEdits():state.plan.segments.map(s=>({...s})),single=suggestionId!==null;
      const target=single?edits.find(s=>s.suggestion_id===suggestionId):null;
      if(single&&!target)throw Object.assign(new Error('Unknown segment'),{code:'invalid_segments'});
      const format=single?'mp4':$('export-format').value,segments=single?[{...target,enabled:true}]:edits;
      if(format==='mp4'&&!segments.some(s=>s.enabled))throw Object.assign(new Error('Empty selection'),{code:'no_enabled_segments'});
      const context=exportContext(suggestionId,segments,format);
      if(state.dirty&&!await saveEdits(edits,epoch))return;
      pending={key:uid(),request:{api_version:'1',task_id:state.task.task_id,format,segments},context};
      clearTimeout(exportTimer);state.exporting=null;state.autoDownloadId=null;state.record.exportId=null;state.record.exportContext=context;
      state.record.pendingExport=pending;saveRecord();$('download').hidden=true;
    }
    $('export-state').textContent=`${pending.context?.label||'所选片段'}：正在创建导出…`;
    const item=await post('/api/v1/exports',pending.request,pending.key);if(epoch!==state.epoch)return;
    state.record.pendingExport=null;state.record.exportId=item.export_id;state.record.exportContext=pending.context||null;state.autoDownloadId=item.export_id;
    saveRecord();handleExport(item);pollExport(epoch);
  }catch(error){
    if(epoch!==state.epoch)return;message(describe(error));
    if(error.code!=='network_error'){state.record.pendingExport=null;saveRecord();}
    $('export-state').classList.remove('busy');$('export-state').textContent=state.record.pendingExport?'创建结果未确认，请点击「重试上次导出」。':'导出未开始，请检查片段设置后重试。';
  }finally{if(epoch===state.epoch){state.busy=false;renderControls();}}
}
function handleExport(item){state.exporting=item;$('download').hidden=true;$('export-state').classList.toggle('busy',ACTIVE.has(item.status));
  const context=state.record?.exportContext,label=context?.label||'所选片段';
  $('export-state').textContent={queued:`${label}：导出排队中…`,running:`${label}：正在生成文件…`,succeeded:`${label} · ${item.mock?'模拟文件':'文件'}已生成 · ${size(item.artifact?.size_bytes||0)}`,failed:`${label}：${describe(item.error||{})}`}[item.status];
  if(item.status==='succeeded'){
    const link=$('download');link.href=safePath(item.artifact.url);link.download=context?.filename||`highlight-${item.export_id.slice(0,8)}.${item.format}`;
    link.textContent=`↓ 下载 ${item.mock?'模拟 ':''}${label} ${item.format.toUpperCase()}`;link.hidden=false;
    if(state.autoDownloadId===item.export_id){state.autoDownloadId=null;link.click();}
  }else if(item.status==='failed')state.autoDownloadId=null;
  renderControls();
}
async function pollExport(epoch){
  clearTimeout(exportTimer);if(epoch!==state.epoch||!state.exporting||!ACTIVE.has(state.exporting.status))return;
  const id=state.exporting.export_id;
  exportTimer=setTimeout(async()=>{try{const item=await api(`/api/v1/exports/${id}`);if(epoch!==state.epoch||state.exporting?.export_id!==id)return;handleExport(item);pollExport(epoch);}catch(error){if(epoch!==state.epoch||state.exporting?.export_id!==id)return;message(describe(error));$('export-state').textContent='导出状态连接中断，点击「重新连接任务」恢复。';$('resume-task').hidden=false;}},1500);
}
async function restore(record){
  reset();view('editor');execution=record.execution||'debug';state.record={...record};state.ratio=[...record.ratio];state.busy=true;renderControls();const epoch=state.epoch;
  $('file-summary').textContent=record.fileName;$('task-state').textContent='正在恢复任务…';
  try{const task=await api(`/api/v1/tasks/${record.id}`);if(epoch!==state.epoch)return;handleTask(task);if(task.status==='succeeded')await loadPlan(epoch);else pollTask(epoch);
    if(record.exportId){try{const item=await api(`/api/v1/exports/${record.exportId}`);if(epoch!==state.epoch)return;handleExport(item);pollExport(epoch);}catch(error){message(describe(error));}}
  }catch(error){if(epoch!==state.epoch)return;message(describe(error));$('task-state').textContent='任务恢复失败；请检查服务连接与运行目录。';$('resume-task').hidden=false;}
  finally{if(epoch===state.epoch){state.busy=false;renderControls();}}
}
$('dismiss-message').onclick=clearMessage;
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
$('refresh-service').onclick=$('settings-refresh').onclick=checkService;
$('mode-help').onclick=$('execution-guide').onclick=()=>view('settings');
for(const button of document.querySelectorAll('[data-execution]'))button.onclick=()=>chooseExecution(button.dataset.execution);
$('debug-mode').onclick=()=>chooseExecution('debug');
$('choose-file').onclick=$('replace-file').onclick=()=>{$('file-input').value='';$('file-input').click();};$('file-input').onchange=e=>selectFile(e.target.files[0]);
$('dropzone').ondragover=e=>{e.preventDefault();if(!inputsLocked())$('dropzone').classList.add('dragover');};
$('dropzone').ondragleave=()=>$('dropzone').classList.remove('dragover');$('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('dragover');if(e.dataTransfer.files.length!==1){message('请一次选择一个视频，每段视频建立独立任务。');return;}selectFile(e.dataTransfer.files[0]);};
for(const b of document.querySelectorAll('[data-ratio]'))b.onclick=()=>{state.ratio=b.dataset.ratio.split(',').map(Number);renderControls();};
$('start-analysis').onclick=$('retry-task').onclick=start;
$('new-project').onclick=()=>{if(state.task||state.file||state.pending)$('new-dialog').showModal();else $('file-input').click();};
$('new-dialog').addEventListener('close',()=>{if($('new-dialog').returnValue==='confirm')reset();});
$('cancel-task').onclick=async()=>{const epoch=state.epoch;try{const task=await post(`/api/v1/tasks/${state.task.task_id}/cancel`,{});if(epoch!==state.epoch)return;handleTask(task);}catch(error){message(describe(error));}};
$('resume-task').onclick=()=>{if(state.busy)return;if(state.dirty&&!confirm('重新连接会舍弃尚未应用的片段调整，是否继续？'))return;if(state.record)restore({...state.record});else if(state.pending)start();};
$('apply-edits').onclick=()=>applyEdits();$('reset-segments').onclick=()=>applyEdits(true);$('export-button').onclick=()=>createExport();$('export-format').onchange=renderControls;
$('source-view').onclick=()=>{state.crop=false;state.previewSegment=null;state.previewEnded=false;$('crop-canvas').hidden=true;$('crop-placeholder').hidden=true;$('source-view').classList.add('selected');$('crop-view').classList.remove('selected');$('source-view').setAttribute('aria-pressed','true');$('crop-view').setAttribute('aria-pressed','false');};
$('crop-view').onclick=()=>{state.crop=true;seekVideo($('video').currentTime);prepareCropPlayback();$('crop-view').classList.add('selected');$('source-view').classList.remove('selected');$('crop-view').setAttribute('aria-pressed','true');$('source-view').setAttribute('aria-pressed','false');drawCrop();};
for(const [id,direction] of [['prev-frame',-1],['next-frame',1]])$(id).onclick=()=>{const v=$('video'),fps=state.plan.metadata.fps_num/state.plan.metadata.fps_den;seekVideo(Math.min((state.plan.metadata.frame_count-1)/fps,(Math.floor(v.currentTime*fps+0.0001)+direction)/fps));};
$('play-toggle').onclick=async()=>{try{if($('video').paused)await $('video').play();else $('video').pause();}catch{message('视频暂时无法播放，请检查文件编码。');}};
$('video').addEventListener('play',()=>{if(!prepareCropPlayback())return;$('play-toggle').textContent='Ⅱ';$('play-toggle').setAttribute('aria-label','暂停视频');if(!('requestVideoFrameCallback'in $('video')))playbackAnimation=requestAnimationFrame(playbackTick);});
$('video').addEventListener('pause',()=>{cancelAnimationFrame(playbackAnimation);$('play-toggle').textContent='▷';$('play-toggle').setAttribute('aria-label','播放视频');});
$('seek-video').oninput=e=>seekVideo(Number(e.target.value));
function seekTimeline(clientX){const rect=$('timeline').getBoundingClientRect();seekVideo((clientX-rect.left)/rect.width*timelineDuration());}
$('timeline').addEventListener('pointerdown',event=>{
  if(event.button!==0||!state.plan?.segments?.length||!$('video').getAttribute('src')||event.target.closest('.timeline-block'))return;
  event.preventDefault();timelinePointerId=event.pointerId;$('timeline').setPointerCapture(event.pointerId);$('timeline-playhead').focus({preventScroll:true});seekTimeline(event.clientX);
});
$('timeline').addEventListener('pointermove',event=>{if(event.pointerId===timelinePointerId)seekTimeline(event.clientX);});
for(const name of ['pointerup','pointercancel','lostpointercapture'])$('timeline').addEventListener(name,event=>{if(event.pointerId===timelinePointerId)timelinePointerId=null;});
$('timeline-playhead').addEventListener('keydown',event=>{
  const fps=state.plan?.metadata.fps_num/state.plan?.metadata.fps_den,step=Number.isFinite(fps)?(event.shiftKey?10:1)/fps:.1;
  const moves={ArrowLeft:$('video').currentTime-step,ArrowRight:$('video').currentTime+step,Home:0,End:timelineDuration()};
  if(Object.hasOwn(moves,event.key)){event.preventDefault();seekVideo(moves[event.key]);}
});
$('video').addEventListener('loadedmetadata',()=>{$('seek-video').max=String($('video').duration);renderControls();if(!state.plan)$('video-meta').textContent=`${$('video').videoWidth} × ${$('video').videoHeight} · ${time($('video').duration)}`;drawCrop();});
$('video').addEventListener('error',()=>{if($('video').getAttribute('src'))message('浏览器暂时无法预览此视频，可能是编码不兼容或文件不完整。可以尝试提交分析，或重新选择视频。');});
$('video').addEventListener('timeupdate',drawCrop);$('video').addEventListener('seeked',drawCrop);
window.addEventListener('beforeunload',e=>{if(state.busy||state.dirty){e.preventDefault();e.returnValue='';}});
async function init(){
  try{deployment=await api('/api/v1/deployment');if(remoteService())STORE+=`.remote.${deployment.mode==='real'?'real':'mock'}`;}catch{}
  let saved=null;try{saved=JSON.parse(localStorage.getItem(STORE)||'null');if(saved&&Array.isArray(saved.records))state.records=saved.records.filter(r=>/^[a-f0-9]{32}$/.test(r.id)&&Array.isArray(r.ratio)&&typeof r.fileName==='string').slice(0,20);}catch{message('本地任务记录无法读取，将开启新的工作区。');}
  renderHistory();await checkService();const active=state.records.find(r=>r.id===saved?.activeId);
  if(saved?.pending){execution=saved.pending.execution||'debug';state.pending=saved.pending;state.ratio=saved.pending.ratio;$('file-summary').textContent=saved.pending.fileName;$('task-state').textContent='有待确认的创建请求，点击重试继续。';renderControls();}
  else if(active)await restore(active);
  if('requestVideoFrameCallback'in $('video'))$('video').requestVideoFrameCallback(videoFrame);
}
init();
