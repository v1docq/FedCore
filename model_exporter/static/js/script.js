/* Token is kept in this page's memory only. Every operation uses the v1 API. */
const element = id => document.getElementById(id);
const headers = () => ({Authorization: `Bearer ${element('token').value}`});
let poll;
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers(), ...(options.headers || {})}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || JSON.stringify(data));
  return data;
}
async function upload(field, kind) {
  const file = element(field).files[0];
  if (!file) throw new Error(`Choose the ${field} file.`);
  const form = new FormData(); form.append('file', file); form.append('kind', kind);
  return (await api('/upload', {method: 'POST', body: form})).id;
}
function report(error) { element('result').textContent = error.message; }
async function refresh() {
  const id = element('job').value;
  if (!/^[a-f0-9]{32}$/.test(id)) throw new Error('Choose a valid job id.');
  const value = await api(`/jobs/${id}`);
  element('result').textContent = JSON.stringify(value, null, 2);
  if (!['queued', 'running'].includes(value.state)) clearInterval(poll);
  return value;
}
element('submit').onclick = async () => {
  element('submit').disabled = true;
  try {
    const model = await upload('model', 'model');
    const example = await upload('example', 'example');
    const validation = await upload('validation', 'dataset');
    const train = element('train').files.length ? await upload('train', 'dataset') : null;
    const request = {version: 1, model, example, input_spec: {shape: element('shape').value.split(',').map(Number), dtype: element('dtype').value},
      data: {validation, train, calibration: null}, task: element('task').value, method: element('method').value,
      retained_energy: Number(element('energy').value), max_relative_error: Number(element('error').value), artifact_format: element('format').value};
    const job = await api('/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(request)});
    element('job').value = job.id; await refresh(); clearInterval(poll);
    poll = setInterval(() => refresh().catch(report), 1500);
  } catch (error) { report(error); } finally { element('submit').disabled = false; }
};
element('refresh').onclick = () => refresh().catch(report);
element('cancel').onclick = async () => { try { await api(`/jobs/${element('job').value}/cancel`, {method: 'POST'}); await refresh(); } catch (error) { report(error); } };
element('download').onclick = async () => {
  try {
    const job = await refresh();
    if (job.state !== 'succeeded') throw new Error('This job has no successful artifact.');
    const response = await fetch(`/jobs/${job.id}/artifact`, {headers: headers()});
    if (!response.ok) throw new Error('Artifact is unavailable.');
    const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a');
    link.href = url; link.download = job.result.artifact; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { report(error); }
};
