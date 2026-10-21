import os, re, json, pickle, random, time
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types, errors

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DATA = Path("data"); DATA.mkdir(exist_ok=True)
INDEX = DATA / "index.pkl"
app = Flask(__name__)

@app.errorhandler(Exception)
def fail(e):
    code = e.code if isinstance(e, HTTPException) else 500
    if code >= 500:
        app.logger.exception(e)
    return jsonify(error=f"{type(e).__name__}: {str(e)[:400]}"), code

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = genai.Client()  # reads GEMINI_API_KEY
chunks, vecs = [], np.zeros((0, 384), dtype="float32")
if INDEX.exists():
    chunks, vecs = pickle.loads(INDEX.read_bytes())

# ---------- ingest ----------
def extract(name, raw):
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        import io
        for i, p in enumerate(PdfReader(io.BytesIO(raw)).pages, 1):
            yield f"p. {i}", p.extract_text() or ""
    elif ext == ".docx":
        import io
        loc, buf = "start", []
        for p in Document(io.BytesIO(raw)).paragraphs:
            if p.style.name.startswith("Heading") and buf:
                yield loc, "\n".join(buf); buf = []
            if p.style.name.startswith("Heading"):
                loc = p.text[:60]
            buf.append(p.text)
        yield loc, "\n".join(buf)
    else:
        yield "text", raw.decode("utf-8", errors="ignore")

def split(text, size=180, overlap=30):
    words = text.split()
    for i in range(0, max(len(words), 1), size - overlap):
        piece = " ".join(words[i:i + size])
        if len(piece) > 40:
            yield piece

def ingest(name, raw):
    global chunks, vecs
    keep = [i for i, c in enumerate(chunks) if c["source"] != name]  # re-upload replaces
    chunks = [chunks[i] for i in keep]; vecs = vecs[keep]
    new = [{"source": name, "loc": loc, "text": t}
           for loc, text in extract(name, raw) for t in split(text)]
    if new:
        v = embedder.encode([c["text"] for c in new], normalize_embeddings=True).astype("float32")
        chunks += new; vecs = np.vstack([vecs, v])
    INDEX.write_bytes(pickle.dumps((chunks, vecs)))
    return len(new)

# ---------- retrieve + generate ----------
def retrieve(q, k=6):
    if not chunks: return []
    qv = embedder.encode([q], normalize_embeddings=True)[0]
    return [chunks[i] for i in np.argsort(-(vecs @ qv))[:k]]

def sources_block(cs):
    return "\n\n".join(f"[{n}] ({c['source']}, {c['loc']})\n{c['text']}" for n, c in enumerate(cs, 1))

def call(system, user, max_tokens=2000):
    for attempt in range(4):
        try:
            r = llm.models.generate_content(
                model=MODEL, contents=user,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=8192))
            return r.text
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))  # waits 2s, 4s, 8s then gives up

def parse_json(s):
    s = re.sub(r"```(?:json)?", "", s)
    return json.loads(s[s.index("["): s.rindex("]") + 1])

def material(topic, k=8):
    return retrieve(topic, k) if topic else random.sample(chunks, min(k, len(chunks)))

@app.post("/upload")
def upload():
    out = {f.filename: ingest(f.filename, f.read()) for f in request.files.getlist("files")}
    return jsonify(files=out, total=len(chunks))

@app.get("/status")
def status():
    return jsonify(files=sorted({c["source"] for c in chunks}), total=len(chunks))

@app.post("/ask")
def ask():
    cs = retrieve(request.json["question"])
    if not cs: return jsonify(error="Upload some study materials first."), 400
    sys = ("You are a study tutor. Answer using ONLY the numbered sources. Cite each claim "
           "like [1]. If the sources don't cover the question, say so plainly.")
    ans = call(sys, f"Sources:\n{sources_block(cs)}\n\nQuestion: {request.json['question']}")
    return jsonify(answer=ans, sources=[{"n": n, "source": c["source"], "loc": c["loc"], "text": c["text"]}
                                        for n, c in enumerate(cs, 1)])

@app.post("/quiz")
def quiz():
    if not chunks: return jsonify(error="Upload some study materials first."), 400
    cs = material(request.json.get("topic", ""))
    n = int(request.json.get("n", 5))
    out = call("You write exam practice questions from study material. Output ONLY a JSON array.",
               f"Sources:\n{sources_block(cs)}\n\nWrite {n} multiple-choice questions. Each item: "
               '{"q": str, "options": [4 strings], "answer": index 0-3, "explanation": str, "source": "[n]"}. '
               "Distractors must be plausible. Use only the sources.", 3000)
    return jsonify(items=parse_json(out))

@app.post("/flashcards")
def flashcards():
    if not chunks: return jsonify(error="Upload some study materials first."), 400
    cs = material(request.json.get("topic", ""))
    n = int(request.json.get("n", 8))
    out = call("You make concise study flashcards. Output ONLY a JSON array.",
               f"Sources:\n{sources_block(cs)}\n\nMake {n} flashcards. Each item: "
               '{"front": question or term, "back": short answer, "source": "[n]"}. Use only the sources.', 3000)
    return jsonify(items=parse_json(out))

@app.get("/")
def home():
    return PAGE

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study RAG</title>
<style>
:root{--bg:#f7f6f1;--fg:#1d2a33;--mut:#66727a;--card:#fff;--line:#d9d8d0;--hi:#ffe66d;--ok:#2f7d4f;--bad:#b3402f}
@media(prefers-color-scheme:dark){:root{--bg:#141b20;--fg:#e8ecee;--mut:#93a0a8;--card:#1c262d;--line:#2d3a42;--hi:#6b5d10;--ok:#6fcf97;--bad:#f0907f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 18px 60px}
h1{font:600 2rem Georgia,serif;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 22px}
.drop{border:2px dashed var(--line);border-radius:10px;padding:18px;text-align:center;cursor:pointer;background:var(--card)}
.drop:hover,.drop.on{border-color:var(--fg)}#files{color:var(--mut);font-size:.9rem;margin:8px 0 20px}
nav{display:flex;gap:6px;margin-bottom:16px}nav button{flex:1}
button{font:inherit;padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid var(--fg);outline-offset:2px}
button.sel,button.go{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.row{display:flex;gap:8px;margin-bottom:14px}input{flex:1;font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.ans{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;white-space:pre-wrap}
.src{margin-top:10px;font-size:.88rem;color:var(--mut)}.src b{background:var(--hi);color:var(--fg);padding:0 4px;border-radius:3px}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
.q h3{margin:0 0 10px;font-size:1.05rem}.opt{display:block;width:100%;text-align:left;margin:6px 0}
.opt.ok{border-color:var(--ok);color:var(--ok)}.opt.bad{border-color:var(--bad);color:var(--bad)}.exp{margin-top:8px;color:var(--mut);font-size:.92rem}
.card{min-height:170px;display:flex;align-items:center;justify-content:center;text-align:center;font:1.25rem Georgia,serif;padding:24px;width:100%}
.hidden{display:none}.msg{color:var(--mut)}
</style>
<main>
<h1>Study RAG</h1><p class="sub">Ask your notes, get quizzed, flip flashcards. Every answer points back to your source.</p>
<label class="drop" id="drop"><input id="up" type="file" multiple accept=".pdf,.docx,.txt,.md" hidden>Drop PDFs, Word docs or notes here, or click to choose</label>
<div id="files">No files yet.</div>
<nav><button class="sel" data-t="ask">Ask</button><button data-t="quiz">Quiz</button><button data-t="cards">Flashcards</button></nav>

<section id="ask"><div class="row"><input id="qin" placeholder="What does chapter 3 say about..."><button class="go" id="askb">Ask</button></div><div id="aout"></div></section>
<section id="quiz" class="hidden"><div class="row"><input id="qt" placeholder="Topic (optional)"><button class="go" id="quizb">Make quiz</button></div><div id="qout"></div></section>
<section id="cards" class="hidden"><div class="row"><input id="ct" placeholder="Topic (optional)"><button class="go" id="cardb">Make flashcards</button></div><div id="cout"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
const post=async(u,b)=>{const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});const t=await r.text();let j;try{j=JSON.parse(t)}catch{throw new Error(`Server sent ${r.status} ${r.statusText} from ${u}: ${t.replace(/<[^>]*>/g,' ').trim().slice(0,200)}`)}if(!r.ok)throw new Error(j.error||'Request failed');return j};
async function status(){const j=await (await fetch('/status')).json();$('files').textContent=j.files.length?`${j.files.length} file(s), ${j.total} chunks: ${j.files.join(', ')}`:'No files yet.'}
status();
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('sel',x===b));['ask','quiz','cards'].forEach(t=>$(t).classList.toggle('hidden',t!==b.dataset.t))});
async function send(files){const fd=new FormData();[...files].forEach(f=>fd.append('files',f));$('files').textContent='Indexing...';await fetch('/upload',{method:'POST',body:fd});status()}
$('up').onchange=e=>send(e.target.files);
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('on')};$('drop').ondragleave=()=>$('drop').classList.remove('on');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('on');send(e.dataTransfer.files)};
async function run(btn,out,fn){const t=btn.textContent;btn.disabled=true;btn.textContent='Working...';out.replaceChildren(el('p','msg','Thinking...'));
 try{await fn()}catch(e){out.replaceChildren(el('p','msg',e.message))}btn.disabled=false;btn.textContent=t}
$('askb').onclick=()=>run($('askb'),$('aout'),async()=>{const j=await post('/ask',{question:$('qin').value});
 const box=el('div');box.append(el('div','ans',j.answer));const s=el('div','src');
 j.sources.forEach(x=>{const p=el('p');p.append(el('b',0,`[${x.n}]`),` ${x.source}, ${x.loc}: ${x.text.slice(0,140)}...`);s.append(p)});box.append(s);$('aout').replaceChildren(box)});
$('quizb').onclick=()=>run($('quizb'),$('qout'),async()=>{const j=await post('/quiz',{topic:$('qt').value});let score=0,done=0;const box=el('div');const tally=el('p','msg','');
 j.items.forEach((it,i)=>{const q=el('div','q');q.append(el('h3',0,`${i+1}. ${it.q}`));const ex=el('div','exp');let locked=false;
  it.options.forEach((o,k)=>{const b=el('button','opt',o);b.onclick=()=>{if(locked)return;locked=true;done++;if(k===it.answer)score++;
   b.classList.add(k===it.answer?'ok':'bad');q.querySelectorAll('.opt')[it.answer].classList.add('ok');ex.textContent=`${it.explanation} ${it.source||''}`;tally.textContent=`Score: ${score}/${done}`};q.append(b)});q.append(ex);box.append(q)});
 box.append(tally);$('qout').replaceChildren(box)});
$('cardb').onclick=()=>run($('cardb'),$('cout'),async()=>{const j=await post('/flashcards',{topic:$('ct').value});let i=0,back=false;
 const card=el('button','card'),info=el('p','msg'),nav=el('div','row'),prev=el('button',0,'Previous'),next=el('button',0,'Next');
 const draw=()=>{const c=j.items[i];card.textContent=back?`${c.back} ${c.source||''}`:c.front;info.textContent=`Card ${i+1} of ${j.items.length}. Click the card to flip.`};
 card.onclick=()=>{back=!back;draw()};prev.onclick=()=>{i=(i+j.items.length-1)%j.items.length;back=false;draw()};next.onclick=()=>{i=(i+1)%j.items.length;back=false;draw()};
 nav.append(prev,next);$('cout').replaceChildren(card,info,nav);draw()});
</script></html>"""

if __name__ == "__main__":
    app.run(port=5057, debug=False)
