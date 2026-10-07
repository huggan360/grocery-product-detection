// ---------------------------------------------------------
// SHARED STATE AND SMALL UI HELPERS
// ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const state = {
  token: sessionStorage.getItem('fridge-token'), name: sessionStorage.getItem('fridge-name'),
  library: null, current: null, detail: null, zones: null, zoneTool: null, drag: null,
  selected: null, signature: '',
};
const ZONE_COLORS = {inside: '#2b9a48', outside: '#e07b20'};

async function api(path, method = 'GET', body = undefined) {
  const headers = {};
  if (state.token) headers['X-Session'] = state.token;
  if (body !== undefined && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(body);
  }
  const response = await fetch(path, {method, headers, body});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`);
    error.status = response.status;
    if (response.status === 401) join();
    throw error;
  }
  return response.json();
}

function toast(message, error = false) {
  $('toast').textContent = message;
  $('toast').className = `toast${error ? ' error' : ''}`;
  $('toast').hidden = false;
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 5000);
}

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

const seconds = (value) => `${value.toFixed(1)} s`;

// ---------------------------------------------------------
// JOIN WITH A NAME
// ---------------------------------------------------------
function join() {
  if (!$('join-dialog').open) $('join-dialog').showModal();
}

$('join-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const result = await api('/api/editors', 'POST', {name: $('editor-name').value});
  state.token = result.token; state.name = result.name;
  sessionStorage.setItem('fridge-token', result.token);
  sessionStorage.setItem('fridge-name', result.name);
  $('user-badge').textContent = result.name;
  $('join-dialog').close();
});

// ---------------------------------------------------------
// CLIP LIST, POLLED SO PROGRESS AND OTHER USERS' CLIPS APPEAR
// ---------------------------------------------------------
async function refresh() {
  try {
    state.library = await api('/api/videos');
  } catch (error) {
    $('model-status').textContent = error.message;
    return;
  }
  const library = state.library;
  $('model-status').textContent = library.model_message;
  if (!$('upload-camera').options.length) {
    for (const camera of Object.keys(library.cameras)) $('upload-camera').append(new Option(camera, camera));
    buildRecordControls(library.recordable);
  }
  const signature = JSON.stringify(library.items.map((item) => [item.id, item.status, item.progress, item.reviewed, item.movements]));
  if (signature !== state.signature) {
    state.signature = signature;
    renderList();
    const current = library.items.find((item) => item.id === state.current);
    if (current && current.status === 'done' && (!state.detail || state.detail.meta.status !== 'done')) openVideo(current.id);
    else if (current && state.detail) { state.detail.meta = current; renderHeading(); }
  }
}

function renderList() {
  const list = $('video-list');
  list.replaceChildren();
  if (!state.library.items.length) list.append(element('p', {class: 'queue-empty'}, 'No clips yet.'));
  for (const item of state.library.items) {
    const status = item.status === 'done' ? `${item.movements ?? 0} movements` : item.status === 'failed' ? 'failed' : `${item.progress}%`;
    const card = element('button', {class: `image-card${item.id === state.current ? ' active' : ''}`, onclick: () => openVideo(item.id)},
      element('div', {class: 'card-info'},
        element('div', {class: 'card-top'}, element('strong', {}, item.filename), element('span', {}, item.camera)),
        element('div', {class: 'card-bottom'}, `${status}${item.reviewed ? ' · reviewed' : ''} · ${new Date(item.created * 1000).toLocaleString()}`)));
    if (item.status === 'running' || item.status === 'queued') {
      card.append(element('div', {class: 'card-progress', style: `width:${item.progress}%`}));
    }
    list.append(card);
  }
}

function buildRecordControls(cameras) {
  const box = $('record-controls');
  if (!cameras.length) return;
  const length = element('input', {type: 'number', min: '1', max: '120', value: '10', title: 'Seconds'});
  box.append(element('span', {}, 'Record'), length, element('span', {}, 's'));
  for (const camera of cameras) {
    box.append(element('button', {class: 'primary', onclick: async (event) => {
      event.target.disabled = true;
      toast(`Recording ${camera} for ${length.value} s…`);
      try {
        const result = await api('/api/videos/record', 'POST', {camera, seconds: Number(length.value)});
        state.current = result.id;
        toast(`Recorded ${camera}. Processing…`);
        refresh();
      } catch (error) { toast(error.message, true); }
      event.target.disabled = false;
    }}, camera));
  }
}

// ---------------------------------------------------------
// OPEN ONE CLIP
// ---------------------------------------------------------
async function openVideo(id) {
  state.current = id;
  state.selected = null;
  state.detail = await api(`/api/videos/${id}`);
  const camera = state.detail.meta.camera;
  state.zones = structuredClone(state.library.cameras[camera]);
  renderList();
  renderHeading();
  const ready = state.detail.meta.status === 'done' && state.detail.result;
  $('video-empty').hidden = ready;
  $('video-frame').hidden = !ready;
  $('video-empty').textContent = state.detail.meta.status === 'failed' ? state.detail.meta.message : 'Processing… the clip opens when the models are done.';
  if (ready) {
    const source = `/api/videos/${id}/files/video.mp4`;
    if (!$('player').src.endsWith(source)) $('player').src = source;
  }
  renderInspector();
  draw();
}

function renderHeading() {
  const {meta, result} = state.detail;
  $('video-title').textContent = meta.filename;
  const info = result ? ` · ${seconds(result.video.duration)} · ${result.video.width}×${result.video.height} · ${result.backend}` : '';
  $('video-meta').textContent = `${meta.camera} camera · event ${meta.event_id}${info}`;
  $('video-state').textContent = meta.status === 'done' ? meta.message : `${meta.status}: ${meta.message}`;
  const warnings = result ? result.warnings : [];
  $('video-warning').hidden = !warnings.length;
  $('video-warning').textContent = warnings.join(' ');
}

// ---------------------------------------------------------
// MOVEMENTS AND TRACK CORRECTIONS
// ---------------------------------------------------------
function review(trackId) {
  const saved = state.detail.review?.tracks?.[trackId];
  const track = state.detail.result.tracks.find((row) => row.id === trackId);
  return saved || {category: track.category.startsWith('imagenet-') ? 'unknown' : track.category, ignored: track.ignored, direction: 'auto'};
}

function renderInspector() {
  const result = state.detail?.result;
  $('movement-list').replaceChildren();
  $('track-list').replaceChildren();
  $('notes').value = state.detail?.review?.notes || '';
  $('reviewed').checked = !!state.detail?.review?.reviewed;
  if (!result) { $('movement-count').textContent = '0'; $('track-count').textContent = '0'; return; }
  $('movement-count').textContent = result.movements.length;
  if (!result.movements.length) $('movement-list').append(element('p', {class: 'muted'}, 'No in/out movement detected.'));
  for (const movement of result.movements) {
    // Movement times are door-event times; the player shows this clip, which starts at the offset.
    const clipTime = Math.max(0, movement.start - (result.offset_seconds || 0));
    $('movement-list').append(element('div', {class: `movement-row ${movement.direction}`, onclick: () => seek(movement.track_id, clipTime)},
      element('b', {}, movement.direction.toUpperCase()), element('span', {}, `#${movement.track_id} ${movement.category}`),
      element('span', {class: 'muted'}, `${seconds(movement.start)}${movement.classification_confidence != null ? ` · ${(movement.classification_confidence * 100).toFixed(0)}%` : ''}`)));
  }
  // Moving or classified tracks first; static clutter last.
  const tracks = [...result.tracks].sort((a, b) => (b.events.length - a.events.length) || ((b.movement || 0) - (a.movement || 0)));
  $('track-count').textContent = tracks.length;
  for (const track of tracks) {
    const values = review(track.id);
    const category = element('select', {'data-track': track.id, 'data-field': 'category'});
    category.append(new Option('unknown', 'unknown'));
    for (const name of state.library.categories) category.append(new Option(name, name));
    category.value = values.category;
    const direction = element('select', {'data-track': track.id, 'data-field': 'direction'});
    for (const [value, label] of [['auto', `auto (${track.events.map((e) => e.direction).join(', ') || 'none'})`], ['in', 'in'], ['out', 'out'], ['none', 'no movement']]) direction.append(new Option(label, value));
    direction.value = values.direction;
    const ignore = element('input', {type: 'checkbox', 'data-track': track.id, 'data-field': 'ignored'});
    ignore.checked = values.ignored;
    const guesses = (track.top_categories || []).map((row) => `${row.category.replace('imagenet-', '')} ${(row.confidence * 100).toFixed(0)}%`).join(', ');
    const first = track.observations[0];
    const image = track.crop_file ? element('img', {src: `/api/videos/${state.current}/files/${track.crop_file}`, alt: `Track ${track.id}`, onclick: () => seek(track.id, first.time)}) : element('div');
    $('track-list').append(element('div', {class: `track-row${state.selected === track.id ? ' active' : ''}`, id: `track-${track.id}`}, image,
      element('div', {},
        element('div', {}, element('strong', {}, `#${track.id} `), `${track.source_label} · ${seconds(first.time)}–${seconds(track.observations.at(-1).time)}`),
        element('div', {class: 'muted'}, guesses || 'not classified (did not move)'),
        category, direction, element('label', {class: 'ignore'}, ignore, 'Ignore (not a product)'))));
  }
}

function seek(trackId, time) {
  state.selected = trackId;
  $('player').pause();
  $('player').currentTime = time;
  document.querySelectorAll('.track-row').forEach((row) => row.classList.toggle('active', row.id === `track-${trackId}`));
  $(`track-${trackId}`)?.scrollIntoView({block: 'nearest'});
  draw();
}

$('save-review').addEventListener('click', async () => {
  if (!state.detail?.result) return;
  const tracks = {};
  for (const track of state.detail.result.tracks) tracks[track.id] = review(track.id);
  for (const input of document.querySelectorAll('[data-track]')) {
    tracks[input.dataset.track][input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value;
  }
  try {
    state.detail.review = await api(`/api/videos/${state.current}/review`, 'PUT', {tracks, notes: $('notes').value, reviewed: $('reviewed').checked});
    toast('Review saved.');
    state.signature = '';
    refresh();
  } catch (error) { toast(error.message, true); }
});

// ---------------------------------------------------------
// OVERLAY: ZONES AND THE BOXES VISIBLE AT THE CURRENT TIME
// ---------------------------------------------------------
function draw() {
  const canvas = $('overlay'), video = $('player');
  if ($('video-frame').hidden || !video.videoWidth) return;
  const width = video.clientWidth, height = video.clientHeight - 40;  // Leave the controls usable.
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  // Map normalized coordinates onto the letterboxed picture inside the <video> element.
  const scale = Math.min(width / video.videoWidth, video.clientHeight / video.videoHeight);
  const pictureWidth = video.videoWidth * scale, pictureHeight = video.videoHeight * scale;
  const left = (width - pictureWidth) / 2, top = (video.clientHeight - pictureHeight) / 2;
  const box = ([x1, y1, x2, y2]) => [left + x1 * pictureWidth, top + y1 * pictureHeight, (x2 - x1) * pictureWidth, (y2 - y1) * pictureHeight];
  state.frame = {left, top, pictureWidth, pictureHeight};
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, width, height);
  context.lineWidth = 2;
  context.font = '12px Arial';
  for (const [name, rect] of Object.entries(state.zones || {})) {
    if (!rect) continue;
    context.strokeStyle = ZONE_COLORS[name];
    context.setLineDash([6, 4]);
    context.strokeRect(...box(rect));
    context.setLineDash([]);
    context.fillStyle = ZONE_COLORS[name];
    const [x, y] = box(rect);
    context.fillText(name, x + 4, y + 14);
  }
  if (state.drag) {
    context.strokeStyle = ZONE_COLORS[state.zoneTool];
    context.strokeRect(state.drag.x, state.drag.y, state.drag.w, state.drag.h);
  }
  const result = state.detail?.result;
  if (!result) return;
  const window = 1 / result.sample_fps;
  const now = video.currentTime;
  for (const track of result.tracks) {
    if (!$('show-all').checked && !track.events.length && (track.movement || 0) < 0.03 && track.id !== state.selected) continue;
    const nearest = track.observations.reduce((best, row) => Math.abs(row.time - now) < Math.abs(best.time - now) ? row : best);
    if (Math.abs(nearest.time - now) > window) continue;
    const values = review(track.id);
    if (values.ignored) continue;
    context.strokeStyle = track.id === state.selected ? '#ffd400' : (track.events.length ? '#2f7fd8' : '#9aa');
    context.strokeRect(...box(nearest.box));
    const label = `#${track.id} ${values.category !== 'unknown' ? values.category : track.category.replace('imagenet-', '~')}`;
    const [x, y] = box(nearest.box);
    context.fillStyle = '#000a';
    context.fillRect(x, y - 16, context.measureText(label).width + 8, 16);
    context.fillStyle = 'white';
    context.fillText(label, x + 4, y - 4);
  }
}

$('player').addEventListener('timeupdate', draw);
$('player').addEventListener('seeked', draw);
$('player').addEventListener('loadedmetadata', draw);
$('show-all').addEventListener('change', draw);
window.addEventListener('resize', draw);
(function animate() { if (!$('player').paused) draw(); requestAnimationFrame(animate); })();

// ---------------------------------------------------------
// DRAW AND SAVE INSIDE / OUTSIDE ZONES FOR THIS CAMERA
// ---------------------------------------------------------
function chooseZone(name) {
  state.zoneTool = state.zoneTool === name ? null : name;
  $('zone-inside').classList.toggle('active', state.zoneTool === 'inside');
  $('zone-outside').classList.toggle('active', state.zoneTool === 'outside');
  $('overlay').classList.toggle('drawing', !!state.zoneTool);
  if (state.zoneTool) $('player').pause();
}
$('zone-inside').addEventListener('click', () => chooseZone('inside'));
$('zone-outside').addEventListener('click', () => chooseZone('outside'));
$('zone-clear').addEventListener('click', () => { if (state.zones) { state.zones = {inside: null, outside: null}; draw(); } });

$('overlay').addEventListener('pointerdown', (event) => {
  if (!state.zoneTool) return;
  state.drag = {x0: event.offsetX, y0: event.offsetY, x: event.offsetX, y: event.offsetY, w: 0, h: 0};
  $('overlay').setPointerCapture(event.pointerId);
});
$('overlay').addEventListener('pointermove', (event) => {
  if (!state.drag) return;
  const drag = state.drag;
  drag.x = Math.min(drag.x0, event.offsetX); drag.y = Math.min(drag.y0, event.offsetY);
  drag.w = Math.abs(event.offsetX - drag.x0); drag.h = Math.abs(event.offsetY - drag.y0);
  draw();
});
$('overlay').addEventListener('pointerup', () => {
  const drag = state.drag, frame = state.frame;
  state.drag = null;
  if (!drag || drag.w < 5 || drag.h < 5) { draw(); return; }
  const clamp = (value) => Math.max(0, Math.min(1, value));
  state.zones[state.zoneTool] = [clamp((drag.x - frame.left) / frame.pictureWidth), clamp((drag.y - frame.top) / frame.pictureHeight),
    clamp((drag.x + drag.w - frame.left) / frame.pictureWidth), clamp((drag.y + drag.h - frame.top) / frame.pictureHeight)].map((v) => Math.round(v * 1000) / 1000);
  chooseZone(state.zoneTool);
  draw();
});

$('zone-save').addEventListener('click', async () => {
  if (!state.detail) return;
  const camera = state.detail.meta.camera;
  try {
    state.library.cameras[camera] = await api(`/api/zones/${camera}`, 'PUT', state.zones);
    if (state.detail.result) state.detail.result = await api(`/api/videos/${state.current}/reanalyse`, 'POST');
    toast(`Zones saved for the ${camera} camera; production uses them too.`);
    renderHeading(); renderInspector(); draw();
    state.signature = ''; refresh();
  } catch (error) { toast(error.message, true); }
});

$('rerun-button').addEventListener('click', async () => {
  if (!state.current) return;
  try { await api(`/api/videos/${state.current}/rerun`, 'POST'); state.detail.meta.status = 'queued'; openVideo(state.current); refresh(); }
  catch (error) { toast(error.message, true); }
});

// ---------------------------------------------------------
// EXPORT REVIEWED TRACK CROPS FOR VIT TRAINING
// ---------------------------------------------------------
$('export-button').addEventListener('click', async () => {
  const response = await fetch('/api/videos/export', {headers: {'X-Session': state.token || ''}});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    toast(payload.detail || 'Export failed.', true);
    if (response.status === 401) join();
    return;
  }
  const link = element('a', {href: URL.createObjectURL(await response.blob()), download: 'fridge-video-crops.zip'});
  link.click();
  URL.revokeObjectURL(link.href);
});

// ---------------------------------------------------------
// UPLOAD
// ---------------------------------------------------------
$('upload-button').addEventListener('click', () => { $('upload-result').textContent = ''; $('upload-dialog').showModal(); });
$('cancel-upload').addEventListener('click', () => $('upload-dialog').close());
$('upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData();
  form.append('file', $('upload-file').files[0]);
  form.append('camera', $('upload-camera').value);
  form.append('event_id', $('upload-event').value);
  $('upload-result').textContent = 'Uploading…';
  try {
    const result = await api('/api/videos', 'POST', form);
    state.current = result.id;
    $('upload-dialog').close();
    toast('Uploaded. Processing…');
    state.signature = '';
    await refresh();
    openVideo(result.id);
  } catch (error) { $('upload-result').textContent = error.message; }
});

// ---------------------------------------------------------
// START
// ---------------------------------------------------------
if (state.token) $('user-badge').textContent = state.name; else join();
refresh();
setInterval(refresh, 2000);
