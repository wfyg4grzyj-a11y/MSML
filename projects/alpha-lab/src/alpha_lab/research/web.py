"""Read-only local dashboard. Imports and page loads never configure an LLM."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse


PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaLab · US equity research</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#10151b;color:#e7edf4}body{max-width:1140px;margin:40px auto;padding:0 22px}
h1{font-size:26px}p{color:#aebed0;line-height:1.55}button,select{font:inherit;padding:9px 13px;border:1px solid #52667a;border-radius:6px;background:#1a2938;color:inherit}
button{cursor:pointer}main{display:grid;grid-template-columns:300px 1fr;gap:24px}#runs{display:flex;flex-direction:column;gap:8px}#runs button{text-align:left;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:20px;background:#17212c;border-radius:8px;line-height:1.6}#status{min-height:24px;color:#89d5c4}
@media(max-width:760px){main{grid-template-columns:1fr}}</style>
<h1>AlphaLab · US equity research</h1><p>Saved hypotheses, accounting, signal coverage and risk evidence. Results remain provisional research. Opening this viewer starts no experiments or model requests.</p>
<p><button id="refresh">Refresh library</button> <span id="status" role="status"></span></p><main><nav id="runs" aria-label="Saved research"></nav><section><h2 id="title">Select a saved run</h2><pre id="detail">Inspect a run to verify its artifacts and read the report.</pre></section></main>
<script>
const $=id=>document.getElementById(id); let selected=null;
async function get(path){const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw Error('Unable to read verified research evidence.');return response}
async function inspect(id){selected=id;$('title').textContent=id;$('status').textContent='Verifying saved artifacts…';
try{const data=await(await get('/api/research/inspect?run='+encodeURIComponent(id))).json();
if(data.artifacts_verified){$('detail').textContent=await(await get('/api/research/report?run='+encodeURIComponent(id))).text();$('status').textContent='Completed artifacts verified · research acceptance remains separate';}
else{$('detail').textContent=JSON.stringify(data,null,2);$('status').textContent='Partial or failed evidence · no automatic retry';}}
catch(error){$('status').textContent=error.message}}
async function refresh(){try{const data=await(await get('/api/research/runs')).json();$('runs').replaceChildren();for(const run of data.runs){const button=document.createElement('button');button.textContent=run.id+' · '+run.status+' / '+run.stage;button.onclick=()=>inspect(run.id);$('runs').append(button)}$('status').textContent=data.runs.length+' saved runs'+(data.truncated?' · library scan capped':'');if(!data.runs.length)$('detail').textContent='No supported runs in the configured research root. Run studies through the explicit CLI or notebook API.';}catch(error){$('status').textContent=error.message}}
$('refresh').onclick=refresh;refresh();</script></html>'''


def research_router(root_provider):
    router=APIRouter()
    def service():
        from .service import ResearchService
        root=root_provider()
        if not root:raise ValueError('Configure a research library root')
        return ResearchService(root)
    def failure():return JSONResponse({'error':'Saved research is unavailable, incomplete or failed verification.'},status_code=400)

    @router.get('/research',response_class=HTMLResponse)
    def dashboard():return HTMLResponse(PAGE,headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"})

    @router.get('/api/research/runs')
    def runs():
        try:return JSONResponse(service().list_runs(),headers={'Cache-Control':'no-store'})
        except Exception:return failure()

    @router.get('/api/research/inspect')
    def inspect(run:str):
        try:return JSONResponse(service().inspect(run),headers={'Cache-Control':'no-store'})
        except Exception:return failure()

    @router.get('/api/research/progress')
    def progress(run:str):
        try:return JSONResponse(service().progress(run),headers={'Cache-Control':'no-store'})
        except Exception:return failure()

    @router.get('/api/research/report')
    def report(run:str):
        try:return PlainTextResponse(service().report(run),headers={'Cache-Control':'no-store'})
        except Exception:return failure()
    return router


def serve(root,host='127.0.0.1',port=8766):
    import uvicorn
    from fastapi import FastAPI
    app=FastAPI(title='AlphaLab US equity research',docs_url=None,redoc_url=None)
    app.include_router(research_router(lambda:Path(root).resolve()))
    from fastapi.responses import RedirectResponse
    @app.get('/')
    def home():return RedirectResponse('/research')
    uvicorn.run(app,host=host,port=port)
