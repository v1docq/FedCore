import { spawnSync } from 'node:child_process';
import { existsSync, copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here=dirname(fileURLToPath(import.meta.url));
const [plugin, output]=process.argv.slice(2);
if (!plugin || !output || !isAbsolute(plugin) || !isAbsolute(output) || existsSync(output)) {
  throw new Error('Provide an absolute Data plugin root and a new absolute output directory.');
}
const expectedRuntime='18896172786ae9fbbfed81fdc93754ee0d68d5a8502c99ee0c27066d5f9edf51';
const tool=join(plugin,'output_formats/html/scripts/data-app.mjs');
function run(script,args) {
  const child=spawnSync(process.execPath,[script,...args],{encoding:'utf8',maxBuffer:16*1024**2});
  if(child.stdout)process.stdout.write(child.stdout);
  if(child.stderr)process.stderr.write(child.stderr);
  if(child.status!==0)throw new Error('Report tool failed; preserve the diagnostic output.');
}
run(join(plugin,'output_formats/html/scripts/prepare-data-app.mjs'),
  ['--surface','report','--output',output,'--snapshot',join(here,'src/data.json')]);
for(const name of ['src/data.json','src/content/report/ReportContent.jsx','src/content/report/report.css']) {
  mkdirSync(dirname(join(output,name)),{recursive:true});
  copyFileSync(join(here,name),join(output,name));
}
run(tool,['build','--project-dir',output,'--separate-data']);
const manifest=JSON.parse(readFileSync(join(output,'dist/data-app-build.json'),'utf8'));
const runtime=manifest.runtimeSha256??manifest.runtime?.sha256;
if(runtime!==expectedRuntime)throw new Error('The installed report runtime differs from the reviewed version.');
run(tool,['export-html','--project-dir',output,'--output',
  join(output,'.data-app-offline/exports/independent-m0-a100-report.html')]);
console.log(JSON.stringify({output:resolve(output),runtimeSha256:runtime,numericalLaunches:0}));
