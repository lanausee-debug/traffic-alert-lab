// Actual embedded engine/UI checks, plus comparisons with the Python reference.
// Minimal DOM checks do not replace a browser visual/compatibility check.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const engine=fs.readFileSync(path.join(root,'demo/csv_engine.js'),'utf8');
const context=vm.createContext({});vm.runInContext(engine,context);
const api=context.TrafficCSV;
const run=spawnSync(process.argv[2] || process.env.PYTHON || (process.platform==='win32'?'python':'python3'),
  [path.join(root,'tests/web_csv_fixture.py')],{encoding:'utf8',maxBuffer:12*1024*1024});
assert.equal(run.status,0,run.error ? String(run.error) : run.stderr);
const fixtures=JSON.parse(run.stdout);
for(const fixture of fixtures){
  const actual=JSON.parse(JSON.stringify(api.evaluate(api.parseCSV(fixture.csv),fixture.settings)));
  assert.deepEqual(actual,fixture.expected,fixture.name);
}
const header='station_id,timestamp,score,is_event\n';
const invalid=[header+'A,2020-01-01T00:00:00,0.4,0',header+'A,2020-02-30T00:00:00Z,0.4,0',
  header+'A,2020-01-01T00:00:00Z,NaN,0',header+'A,2020-01-01T00:00:00Z,1.1,0',
  header+'A,2020-01-01T00:00:00Z,0.3,no',header+'"A,2020-01-01T00:00:00Z,0.3,0',
  'station_id,timestamp,score,score,is_event\nA,2020-01-01T00:00:00Z,0.1,0.2,0',
  header+'A,2020-01-01T00:00:00Z,0.2,0,extra'];
for(const csv of invalid)assert.throws(()=>api.parseCSV(csv));
const duplicates=api.parseCSV(header+'A,2020-01-01T00:00:00Z,0.1,0\nA,2020-01-01T00:00:00Z,0.3,0');
assert.throws(()=>api.evaluate(duplicates));
assert.throws(()=>api.parseCSV('x'.repeat(2*1024*1024+1)));

const html=fs.readFileSync(path.join(root,'demo/index.html'),'utf8');
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].filter(match=>!match[0].startsWith('<script id='));
const embedded=scripts.find(match=>match[1].includes('function parseCSV(text)'))[1];
assert(embedded.includes(engine.trim()),'tested engine must be embedded unchanged');
class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.textContent='';this.handlers={};this.value='';this.hidden=false;this.disabled=false;this.innerHTML='';}
  append(...elements){this.children.push(...elements);if(this.tag==='select'&&this.value===''&&elements.length)this.value=elements[0].value;}
  replaceChildren(...elements){this.children=[...elements];this.value='';}
  addEventListener(name,callback){this.handlers[name]=callback;}
  get options(){return this.children;}
  click(){this.clicked=true;}
  remove(){}
}
const ids={};
for(const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g))ids[match[2]]=new Element(match[1]);
for(const [key,value] of Object.entries({threshold:'0.50',step:'5',horizon:'15',early:'10',preset:'standard'}))ids['lab-'+key].value=value;
const blobs=[];
class BlobStub{constructor(parts,options){this.parts=parts;this.type=options.type;blobs.push(this);}}
const runtime=vm.createContext({document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag),body:new Element('body')},
  console,Blob:BlobStub,URL:{createObjectURL:()=> 'blob:local',revokeObjectURL:()=>{}},setTimeout:fn=>fn()});
vm.runInContext(embedded,runtime,{timeout:3000});
assert.equal(ids['lab-detection'].textContent,'100.0%');
assert.equal(ids['lab-recall'].textContent,'33.3%');
assert.equal(ids['lab-false'].textContent,1);
assert.equal(ids['lab-json'].disabled,false);
ids['lab-threshold'].value='0.80';await ids['lab-threshold'].handlers.input();
assert.equal(ids['lab-recall'].textContent,'0.0%');
assert.equal(ids['lab-false'].textContent,0);
ids['lab-threshold'].value='0.50';ids['lab-preset'].value='outage';ids['lab-preset'].handlers.change();
assert.equal(ids['lab-detection'].textContent,'66.7%');
assert.equal(ids['lab-recall'].textContent,'0.0%');
ids['lab-preset'].value='noisy';ids['lab-preset'].handlers.change();
assert.equal(ids['lab-false'].textContent,4);
ids['lab-json'].handlers.click();
assert.equal(JSON.parse(blobs.at(-1).parts[0]).metrics.false_alarm_episodes,4);
ids['lab-html'].handlers.click();
assert(blobs.at(-1).parts[0].includes('traffic-alert-csv-v1'));
ids['lab-csv'].value=ids['lab-csv'].value.replaceAll('synthetic-A','<script>oops</script>');
ids['lab-csv'].handlers.input();assert.equal(ids['lab-json'].disabled,true);
await ids['lab-run'].handlers.click();ids['lab-html'].handlers.click();
assert(!blobs.at(-1).parts[0].includes('<script>oops</script>'));
assert(blobs.at(-1).parts[0].includes('&lt;script&gt;oops&lt;/script&gt;'));
ids['lab-csv'].value=header+'bad';ids['lab-csv'].handlers.input();await ids['lab-run'].handlers.click();
assert.equal(ids['lab-error'].hidden,false);assert.equal(ids['lab-results'].hidden,true);assert.equal(ids['lab-json'].disabled,true);
console.log(JSON.stringify({python_browser_parity_cases:fixtures.length,invalid_csv_cases:invalid.length,
  embedded_ui_checks_passed:true,html_user_content_escaped:true,browser_visual_test:false}));
