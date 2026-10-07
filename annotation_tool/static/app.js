// ---------------------------------------------------------
// SHARED STATE AND SMALL UI HELPERS
// ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const state = {
  token: sessionStorage.getItem('fridge-token'), name: sessionStorage.getItem('fridge-name'),
  workspace: null, tab: 'pending', current: null, image: null, boxes: [], selected: null,
  tool: 'select', nextCategory: '', zoom: 1, fitScale: 1, drag: null, undo: [], redo: [],
  version: 0, savedVersion: 0, savePromise: null, timer: null, writable: false,
  navigating: false, finalizing: false, ingestMode: 'upload', queueSignature: '', pollBusy: false,
  polyPoints: null, polyTarget: null, selectedVertex: null,
};

// Send one request using this tab's editor identity.
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
    throw error;
  }
  return response.json();
}

// Show readable feedback without blocking anyone else's work.
function toast(message, error = false) {
  $('toast').textContent = message;
  $('toast').className = `toast${error ? ' error' : ''}`;
  $('toast').hidden = false;
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 5000);
}

// Connect a button to an async action and keep errors visible.
function action(id, callback) {
  $(id).addEventListener('click', () => Promise.resolve().then(callback).catch(error => toast(error.message, true)));
}

// Make text-only DOM nodes; image filenames and names are never treated as HTML.
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

// Check whether it is currently safe to change the annotation.
function canEdit() {
  return Boolean(state.current && state.writable && !state.finalizing && !state.navigating);
}

// ---------------------------------------------------------
// LIVE QUEUE AND COLLABORATOR PRESENCE
// ---------------------------------------------------------
async function refreshWorkspace() {
  if (state.pollBusy) return;
  state.pollBusy = true;
  try {
    const data = await api('/api/workspace');
    state.workspace = data;
    $('connection').classList.remove('offline');
    $('collaborators').textContent = data.editors.length ? `${data.editors.length} online · ${data.editors.join(', ')}` : 'Workspace online';
    $('model-status').textContent = data.model_available ? data.model_message : 'Manual mode · no model suggestions';
    $('model-status').title = data.model_message;
    $('capture-button').disabled = !data.cameras.length;
    $('capture-button').title = data.cameras.length ? `Capture ${data.cameras.length} shelves` : 'Enable shelf cameras in annotation_tool/config.yaml. Uploads work now.';
    updateCategories();
    const filter = $('shelf-filter');
    const previous = filter.value;
    const shelves = [...new Set(data.images.map(image => image.shelf))].sort();
    if (JSON.stringify(shelves) !== filter.dataset.shelves) {
      filter.replaceChildren(new Option('All shelves', ''), ...shelves.map(shelf => new Option(shelf, shelf)));
      filter.value = shelves.includes(previous) ? previous : '';
      filter.dataset.shelves = JSON.stringify(shelves);
    }
    renderQueue();
  } catch (error) {
    $('connection').classList.add('offline');
    $('collaborators').textContent = 'Connection lost · retrying';
    if (error.status === 401) {
      state.token = null;
      sessionStorage.removeItem('fridge-token');
      if (!$('join-dialog').open) $('join-dialog').showModal();
    }
  } finally {
    state.pollBusy = false;
  }
}

// Redraw the queue only when its data changes, preserving scroll while editing.
function renderQueue() {
  if (!state.workspace) return;
  const all = state.workspace.images;
  $('pending-count').textContent = all.filter(image => image.state === 'pending').length;
  $('reviewed-count').textContent = all.filter(image => image.state === 'reviewed').length;
  $('pending-tab').classList.toggle('active', state.tab === 'pending');
  $('reviewed-tab').classList.toggle('active', state.tab === 'reviewed');
  $('queue-title').textContent = state.tab === 'pending' ? 'Images' : 'Saved images';
  const images = all.filter(image => image.state === state.tab && (!$('shelf-filter').value || image.shelf === $('shelf-filter').value));
  const signature = JSON.stringify([images, state.current?.id, state.tab, $('shelf-filter').value]);
  if (signature === state.queueSignature) return;
  state.queueSignature = signature;
  const list = $('image-list');
  list.replaceChildren();
  if (!images.length) {
    list.append(node('p', 'queue-empty', state.tab === 'pending' ? 'No images.' : 'No saved images.'));
  }
  for (const image of images) {
    const card = node('button', `image-card${state.current?.id === image.id ? ' active' : ''}`);
    card.dataset.imageId = image.id;
    const thumbnail = document.createElement('img');
    thumbnail.src = image.thumbnail_url;
    thumbnail.alt = image.filename;
    thumbnail.loading = 'lazy';
    const info = node('div', 'card-info');
    const top = node('div', 'card-top');
    top.append(node('strong', '', image.shelf), node('span', '', `${image.box_count} boxes`));
    info.append(top, node('div', 'card-bottom', `${image.collection} · ${image.split}`));
    const busy = ['queued', 'running'].includes(image.prediction_state);
    const label = image.locked_by ? (image.mine ? 'You are editing' : `${image.locked_by} is editing`) : busy ? 'Preparing suggestions…' : image.state === 'reviewed' ? '✓ Reviewed' : image.revision ? 'Draft' : 'New';
    card.append(thumbnail, info, node('span', 'card-badge', label));
    card.disabled = busy || Boolean(image.locked_by && !image.mine);
    card.addEventListener('click', () => openImage(image.id).catch(error => toast(error.message, true)));
    list.append(card);
  }
}

// Keep the category selector current without changing a user's selection.
function updateCategories() {
  const select = $('category-select');
  const categories = state.workspace?.categories || [];
  if (select.dataset.categories === JSON.stringify(categories)) return;
  select.dataset.categories = JSON.stringify(categories);
  select.replaceChildren(new Option('Choose category', ''), ...categories.map(name => new Option(name.replaceAll('-', ' '), name)));
  const selected = state.boxes.find(box => box.id === state.selected);
  select.value = selected?.category || state.nextCategory;
}

// ---------------------------------------------------------
// EXCLUSIVE EDITING, AUTOSAVE AND REVIEW
// ---------------------------------------------------------
async function openImage(id = null) {
  if (state.navigating || state.finalizing || (id && id === state.current?.id)) return;
  state.navigating = true;
  try {
    if (!(await closeImage())) return;
    const image = await api(id ? `/api/images/${id}/claim` : '/api/claim-next', 'POST');
    state.current = image;
    state.boxes = structuredClone(image.boxes);
    state.selected = state.boxes[0]?.id || null;
    state.version = state.savedVersion = 0;
    state.undo = []; state.redo = [];
    state.writable = true;
    state.zoom = 1;
    $('notes').value = image.notes;
    $('empty-confirmed').checked = false;
    $('masks-reviewed').checked = Boolean(image.masks_reviewed);
    state.selectedVertex = null;
    $('editor-warning').hidden = true;
    $('download-draft').hidden = true;
    $('image-title').textContent = image.filename;
    $('image-meta').textContent = `${image.shelf} · ${image.collection} · ${image.split} · ${image.width} × ${image.height} px`;
    $('save-state').textContent = image.state === 'reviewed' ? 'Reviewed · editing lease held' : 'Draft is up to date';
    if (image.prediction_error) {
      $('editor-warning').textContent = `Suggestions failed: ${image.prediction_error}. You can draw boxes manually.`;
      $('editor-warning').hidden = false;
    }
    const pixels = new Image();
    pixels.src = image.image_url;
    await pixels.decode();
    state.image = pixels;
    $('empty-editor').hidden = true;
    $('image-canvas').hidden = false;
    setTool('select');
    renderInspector();
    fitCanvas();
    await refreshWorkspace();
  } finally {
    state.navigating = false;
    updateControls();
  }
}

// Flush edits before releasing a frame; never silently discard an unsaved conflict.
async function closeImage() {
  if (!state.current) return true;
  if (state.polyPoints) throw new Error('Finish or cancel the mask outline first.');
  clearTimeout(state.timer);
  if (state.writable) {
    await persistDraft();
  } else if (state.version !== state.savedVersion && !confirm('This tab has unsaved changes. Download the draft first if you need them. Leave this image?')) {
    return false;
  }
  await api(`/api/images/${state.current.id}/release`, 'POST');
  state.current = null; state.image = null; state.boxes = []; state.selected = null;
  state.writable = false;
  $('image-canvas').hidden = true;
  $('empty-editor').hidden = false;
  $('image-title').textContent = 'Select an image';
  $('image-meta').textContent = '';
  $('save-state').textContent = 'No image selected';
  $('notes').value = '';
  $('editor-warning').hidden = true;
  $('download-draft').hidden = true;
  renderInspector();
  updateControls();
  await refreshWorkspace();
  return true;
}

// Save only one request at a time; changes made during a request stay dirty.
async function persistDraft(reviewed = false) {
  clearTimeout(state.timer);
  while (state.savePromise) await state.savePromise;
  if (!state.current || (!reviewed && state.version === state.savedVersion)) return;
  if (!state.writable) throw new Error('This image is no longer locked by you. Download your draft or reopen it.');
  const imageId = state.current.id;
  const version = state.version;
  const payload = {revision: state.current.revision, boxes: structuredClone(state.boxes), notes: $('notes').value,
    reviewed, empty_confirmed: $('empty-confirmed').checked, masks_reviewed: reviewed && $('masks-reviewed').checked};
  $('save-state').textContent = reviewed ? 'Saving reviewed image…' : 'Saving draft…';
  state.savePromise = api(`/api/images/${imageId}/annotations`, 'PUT', payload).then(result => {
    if (state.current?.id !== imageId) return;
    state.current = result;
    state.savedVersion = version;
    $('save-state').textContent = state.version === version ? (reviewed ? '✓ Reviewed and saved' : 'Draft saved') : 'Unsaved changes…';
  }).catch(error => {
    if (error.status === 409) loseLease(error.message);
    else $('save-state').textContent = 'Not saved · check connection';
    throw error;
  });
  try {
    await state.savePromise;
  } finally {
    state.savePromise = null;
  }
  // A slow request must not lose changes entered while it was in flight.
  if (state.version !== state.savedVersion) await persistDraft();
}

// Mark a local change and start a short debounce before saving the draft.
function changed() {
  state.version += 1;
  $('masks-reviewed').checked = false;
  $('save-state').textContent = 'Unsaved changes…';
  if (state.boxes.length) $('empty-confirmed').checked = false;
  renderInspector();
  drawCanvas();
  clearTimeout(state.timer);
  state.timer = setTimeout(() => persistDraft().catch(error => toast(error.message, true)), 650);
}

// Keep enough local history for ordinary correction mistakes.
function remember() {
  state.undo.push(structuredClone(state.boxes));
  if (state.undo.length > 60) state.undo.shift();
  state.redo = [];
}

// Undo or redo a box edit and autosave the restored version.
function undo(redo = false) {
  if (!canEdit() || state.drag) return;
  const source = redo ? state.redo : state.undo;
  const target = redo ? state.undo : state.redo;
  if (!source.length) return;
  target.push(structuredClone(state.boxes));
  state.boxes = source.pop();
  state.selected = state.boxes[0]?.id || null;
  changed();
}

// Final review is explicit. Zero-box frames need the empty-shelf confirmation.
async function review(next) {
  if (!canEdit() || state.drag) return;
  if (state.polyPoints) throw new Error('Finish or cancel the mask outline first.');
  if ($('masks-reviewed').checked && state.boxes.some(box => !box.polygon)) throw new Error('Draw a mask for every product before checking masks.');
  if (state.boxes.some(box => !state.workspace.categories.includes(box.category))) throw new Error('Choose a real category for every box before reviewing.');
  if (!state.boxes.length && !$('empty-confirmed').checked) throw new Error('Confirm that this image has no visible products, or draw its missing boxes.');
  state.finalizing = true;
  updateControls();
  try {
    await persistDraft(true);
    await closeImage();
    toast('Reviewed image saved for training.');
  } finally {
    state.finalizing = false;
    updateControls();
  }
  if (next) {
    try { await openImage(); } catch (error) { toast(error.message, error.status !== 404); }
  }
}

// Freeze local editing if another tab has taken over after a disconnect.
function loseLease(message) {
  state.writable = false;
  clearTimeout(state.timer);
  $('editor-warning').textContent = `${message} Your local changes are still visible. Download the draft if needed, then leave and reopen the image.`;
  $('editor-warning').hidden = false;
  $('download-draft').hidden = false;
  $('save-state').textContent = 'Editing lease lost';
  updateControls();
}

// Renew leases even when the pointer is not moving.
async function heartbeat() {
  if (!state.current || !state.writable) return;
  try {
    await api(`/api/images/${state.current.id}/heartbeat`, 'POST');
  } catch (error) {
    if (error.status === 409 || error.status === 401) loseLease(error.message);
    else $('save-state').textContent = 'Connection interrupted · keep this tab open';
  }
}

// ---------------------------------------------------------
// BOX SELECTION, DRAWING, MOVING AND EIGHT RESIZE HANDLES
// ---------------------------------------------------------
function setTool(tool) {
  if (state.polyPoints && tool !== 'polygon') {
    toast('Finish or cancel the mask outline first.', true);
    return;
  }
  state.tool = tool;
  $('select-tool').classList.toggle('active', tool === 'select');
  $('draw-tool').classList.toggle('active', tool === 'draw');
  $('image-canvas').style.cursor = tool === 'draw' ? 'crosshair' : 'default';
  if (typeof updateMaskControls === 'function') updateMaskControls();
}

// Use actual image pixels for every coordinate, regardless of screen size or zoom.
function pointerPosition(event) {
  const rect = $('image-canvas').getBoundingClientRect();
  return [Math.max(0, Math.min(state.image.width, (event.clientX - rect.left) * state.image.width / rect.width)),
    Math.max(0, Math.min(state.image.height, (event.clientY - rect.top) * state.image.height / rect.height))];
}

// Locate the corner and edge handles in image coordinates.
function handles(box) {
  const [x1, y1, x2, y2] = box.xyxy, mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return {nw: [x1, y1], n: [mx, y1], ne: [x2, y1], e: [x2, my], se: [x2, y2], s: [mx, y2], sw: [x1, y2], w: [x1, my]};
}

// Fit the image to the centre panel while keeping a separate user zoom value.
function fitCanvas() {
  if (!state.image) return;
  const view = $('canvas-viewport');
  state.fitScale = Math.min((view.clientWidth - 48) / state.image.width, (view.clientHeight - 48) / state.image.height, 1);
  state.fitScale = Math.max(0.02, state.fitScale);
  drawCanvas();
}

// Draw the unchanged source pixels and then the editable annotation overlay.
function drawCanvas() {
  if (!state.image) return;
  const canvas = $('image-canvas');
  const scale = state.fitScale * state.zoom;
  const width = state.image.width * scale, height = state.image.height * scale;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  const context = canvas.getContext('2d');
  context.setTransform(scale * dpr, 0, 0, scale * dpr, 0, 0);
  context.drawImage(state.image, 0, 0);
  for (const [index, box] of state.boxes.entries()) {
    const [x1, y1, x2, y2] = box.xyxy;
    const selected = box.id === state.selected;
    const colour = !box.category || box.category === 'unknown' ? '#f3b746' : selected ? '#d9f28e' : '#5de4bc';
    context.strokeStyle = colour;
    context.lineWidth = (selected ? 2.5 : 1.5) / scale;
    context.fillStyle = selected ? '#d9f28e18' : '#32bb9a0b';
    if (box.polygon) {
      context.beginPath();
      box.polygon.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
      context.closePath();
      context.fillStyle = selected ? '#55b7ff44' : '#32bb9a33';
      context.fill(); context.stroke();
    } else context.fillRect(x1, y1, x2 - x1, y2 - y1);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${index + 1} · ${box.category || 'choose category'}`;
    context.font = `500 ${11 / scale}px system-ui`;
    const labelWidth = context.measureText(text).width + 12 / scale;
    const labelY = y1 >= 21 / scale ? y1 - 21 / scale : y1;
    context.fillStyle = colour;
    context.fillRect(x1, labelY, labelWidth, 20 / scale);
    context.fillStyle = '#153d31';
    context.fillText(text, x1 + 6 / scale, labelY + 14 / scale);
    if (selected) {
      for (const [x, y] of box.polygon || Object.values(handles(box))) {
        context.fillStyle = 'white';
        context.fillRect(x - 4 / scale, y - 4 / scale, 8 / scale, 8 / scale);
        context.strokeStyle = '#176953'; context.lineWidth = 1 / scale;
        context.strokeRect(x - 4 / scale, y - 4 / scale, 8 / scale, 8 / scale);
      }
    }
  }
  $('zoom-label').textContent = `${Math.round(scale * 100)}%`;
  if (typeof drawMaskDraft === 'function') drawMaskDraft(context, scale);
  if (typeof renderMaskPreview === 'function') renderMaskPreview();
}

// Start a new box or select the box/resize handle under the pointer.
$('image-canvas').addEventListener('pointerdown', event => {
  if (!canEdit() || event.button !== 0 || !state.image) return;
  event.preventDefault();
  const point = pointerPosition(event);
  if (typeof maskPointerDown === 'function' && maskPointerDown(event, point)) return;
  const selected = state.boxes.find(box => box.id === state.selected);
  const radius = 9 / (state.fitScale * state.zoom);
  const handle = state.tool === 'select' && selected && !selected.polygon ? Object.entries(handles(selected)).find(([, pos]) => Math.hypot(pos[0] - point[0], pos[1] - point[1]) <= radius)?.[0] : null;
  const original = structuredClone(state.boxes);
  if (state.tool === 'draw') {
    const box = {id: crypto.randomUUID(), category: state.nextCategory, xyxy: [...point, ...point]};
    state.boxes.push(box); state.selected = box.id;
    state.drag = {kind: 'draw', point, original};
  } else if (handle) {
    state.drag = {kind: 'resize', point, handle, box: [...selected.xyxy], original};
  } else {
    const target = [...state.boxes].reverse().find(box => box.polygon ? maskContains(point, box.polygon) : point[0] >= box.xyxy[0] && point[0] <= box.xyxy[2] && point[1] >= box.xyxy[1] && point[1] <= box.xyxy[3]);
    state.selected = target?.id || null;
    if (target) state.drag = {kind: 'move', point, box: [...target.xyxy], original};
  }
  if (state.drag) {
    clearTimeout(state.timer);
    $('image-canvas').setPointerCapture(event.pointerId);
  }
  renderInspector(); drawCanvas();
});

// Apply dragging in original pixels; keep boxes inside the image.
$('image-canvas').addEventListener('pointermove', event => {
  if (!state.drag || !canEdit()) return;
  const point = pointerPosition(event), drag = state.drag;
  const box = state.boxes.find(item => item.id === state.selected);
  if (!box) return;
  if (drag.kind === 'vertex') {
    box.polygon[drag.vertex] = point;
    box.xyxy = maskBounds(box.polygon);
    drawCanvas();
    return;
  }
  if (drag.kind === 'draw') {
    box.xyxy = [Math.min(point[0], drag.point[0]), Math.min(point[1], drag.point[1]), Math.max(point[0], drag.point[0]), Math.max(point[1], drag.point[1])];
  } else if (drag.kind === 'move') {
    const dx = Math.max(-drag.box[0], Math.min(state.image.width - drag.box[2], point[0] - drag.point[0]));
    const dy = Math.max(-drag.box[1], Math.min(state.image.height - drag.box[3], point[1] - drag.point[1]));
    box.xyxy = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
  } else {
    const result = [...drag.box];
    const minimum = Math.min(1, (drag.box[2] - drag.box[0]) / 2, (drag.box[3] - drag.box[1]) / 2);
    if (drag.handle.includes('w')) result[0] = Math.min(point[0], result[2] - minimum);
    if (drag.handle.includes('e')) result[2] = Math.max(point[0], result[0] + minimum);
    if (drag.handle.includes('n')) result[1] = Math.min(point[1], result[3] - minimum);
    if (drag.handle.includes('s')) result[3] = Math.max(point[1], result[1] + minimum);
    box.xyxy = result;
  }
  if (box.polygon && drag.kind === 'move') {
    const originalPolygon = drag.original.find(item => item.id === box.id).polygon;
    const dx = box.xyxy[0] - drag.box[0], dy = box.xyxy[1] - drag.box[1];
    box.polygon = originalPolygon.map(([x, y]) => [x + dx, y + dy]);
  }
  drawCanvas();
});

// Commit a drag as one undo step, never saving a temporary zero-size box.
function finishDrag(event, cancelled = false) {
  if (!state.drag) return;
  const drag = state.drag;
  const selected = state.boxes.find(box => box.id === state.selected);
  if (cancelled || (drag.kind === 'draw' && selected && (selected.xyxy[2] - selected.xyxy[0] < 2 || selected.xyxy[3] - selected.xyxy[1] < 2))) {
    state.boxes = drag.original;
    state.selected = state.boxes.find(box => box.id === state.selected)?.id || null;
  } else if (JSON.stringify(drag.original) !== JSON.stringify(state.boxes)) {
    state.undo.push(drag.original); state.redo = [];
    state.drag = null;
    changed();
  }
  state.drag = null;
  if (event && $('image-canvas').hasPointerCapture(event.pointerId)) $('image-canvas').releasePointerCapture(event.pointerId);
  renderInspector(); drawCanvas();
  if (state.version !== state.savedVersion) {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => persistDraft().catch(error => toast(error.message, true)), 650);
  }
}
$('image-canvas').addEventListener('pointerup', event => finishDrag(event));
$('image-canvas').addEventListener('pointercancel', event => finishDrag(event, true));

// ---------------------------------------------------------
// INSPECTOR AND CATEGORY EDITING
// ---------------------------------------------------------
function renderInspector() {
  $('box-count').textContent = state.boxes.length;
  const list = $('box-list'); list.replaceChildren();
  if (!state.boxes.length) list.append(node('p', 'muted', state.current ? 'No boxes.' : ''));
  for (const [index, box] of state.boxes.entries()) {
    const button = node('button', `box-row${box.id === state.selected ? ' active' : ''}`);
    const missing = !box.category || box.category === 'unknown';
    button.append(node('span', 'box-number', index + 1), node('span', `box-category${missing ? ' unlabelled' : ''}`, missing ? 'Choose category' : box.category));
    const suggestion = state.current?.suggestions?.[Number(box.id.replace('model-', ''))];
    if (box.id.startsWith('model-') && suggestion) button.append(node('span', 'box-score', suggestion.classification_confidence == null ? (suggestion.source_label || 'mask') : `model ${Math.round(suggestion.classification_confidence * 100)}%`));
    button.addEventListener('click', () => { state.selected = box.id; state.selectedVertex = null; renderInspector(); drawCanvas(); });
    list.append(button);
  }
  const box = state.boxes.find(item => item.id === state.selected);
  $('selected-box').hidden = !box;
  $('category-select').value = box?.category || state.nextCategory;
  if (box) ['x1', 'y1', 'x2', 'y2'].forEach((key, index) => { $(`coord-${key}`).value = Math.round(box.xyxy[index] * 10) / 10; });
  updateControls();
}

// Disable mutations when there is no lease or a final save is running.
function updateControls() {
  const enabled = canEdit();
  for (const id of ['select-tool', 'draw-tool', 'category-select', 'notes', 'empty-confirmed', 'review-next', 'review-only', 'delete-box', 'coord-x1', 'coord-y1', 'coord-x2', 'coord-y2']) $(id).disabled = !enabled;
  $('undo-button').disabled = !enabled || !state.undo.length;
  $('redo-button').disabled = !enabled || !state.redo.length;
  $('leave-button').disabled = !state.current || state.finalizing;
  $('next-button').disabled = state.navigating || state.finalizing;
  if (typeof updateMaskControls === 'function') updateMaskControls();
}

// Assign the selected category to a box and remember it for future drawn boxes.
$('category-select').addEventListener('change', () => {
  if (!canEdit()) return;
  state.nextCategory = $('category-select').value;
  const box = state.boxes.find(item => item.id === state.selected);
  if (box) { remember(); box.category = state.nextCategory; changed(); }
});

// Allow precise edits using numeric pixel coordinates as well as dragging.
for (const key of ['x1', 'y1', 'x2', 'y2']) $(`coord-${key}`).addEventListener('change', () => {
  if (!canEdit()) return;
  const box = state.boxes.find(item => item.id === state.selected);
  if (!box) return;
  const values = ['x1', 'y1', 'x2', 'y2'].map(name => Number($(`coord-${name}`).value));
  const [x1, y1, x2, y2] = values;
  if (!values.every(Number.isFinite) || !(x1 >= 0 && x1 < x2 && x2 <= state.image.width && y1 >= 0 && y1 < y2 && y2 <= state.image.height)) {
    toast('Keep a positive-size box inside the image.', true); renderInspector(); return;
  }
  remember();
  if (box.polygon) {
    const old = box.xyxy;
    box.polygon = box.polygon.map(([x, y]) => [x1 + (x - old[0]) * (x2 - x1) / (old[2] - old[0]), y1 + (y - old[1]) * (y2 - y1) / (old[3] - old[1])]);
  }
  box.xyxy = values; changed();
});

function deleteSelected() {
  if (!canEdit() || !state.selected || state.drag) return;
  remember();
  state.boxes = state.boxes.filter(box => box.id !== state.selected);
  state.selected = state.boxes[0]?.id || null;
  changed();
}
$('notes').addEventListener('input', () => { if (canEdit()) changed(); });

// ---------------------------------------------------------
// CAPTURE, UPLOAD AND EXPORT
// ---------------------------------------------------------
function openIngest(mode) {
  state.ingestMode = mode;
  $('ingest-title').textContent = mode === 'upload' ? 'Upload shelf images' : 'Capture your shelves';
  $('ingest-description').textContent = mode === 'upload' ? '' : `Cameras: ${state.workspace.cameras.map(camera => camera.name).join(', ')}.`;
  $('upload-fields').hidden = mode !== 'upload';
  $('image-files').required = mode === 'upload';
  $('shelf-name').required = mode === 'upload';
  $('collection-name').value = localStorage.getItem('fridge-collection') || `fridge-${new Date().toISOString().slice(0, 10)}`;
  $('split-select').value = localStorage.getItem('fridge-split') || 'train';
  $('ingest-result').textContent = '';
  $('ingest-dialog').showModal();
}

$('ingest-form').addEventListener('submit', async event => {
  event.preventDefault();
  const collection = $('collection-name').value, split = $('split-select').value;
  $('submit-ingest').disabled = true;
  $('cancel-ingest').disabled = true;
  $('ingest-result').textContent = state.ingestMode === 'upload' ? 'Uploading images…' : 'Capturing shelves…';
  try {
    let result;
    if (state.ingestMode === 'upload') {
      const form = new FormData();
      for (const file of $('image-files').files) form.append('files', file);
      form.append('collection', collection); form.append('split', split); form.append('shelf', $('shelf-name').value);
      result = await api('/api/upload', 'POST', form);
    } else result = await api('/api/capture', 'POST', {collection, split});
    localStorage.setItem('fridge-collection', collection); localStorage.setItem('fridge-split', split);
    await refreshWorkspace();
    if (result.errors.length) $('ingest-result').textContent = `${result.ids.length} images added.\n${result.errors.map(error => `${error.filename}: ${error.message}`).join('\n')}`;
    else { $('ingest-dialog').close(); toast(`${result.ids.length} images added to the shared queue.`); $('image-files').value = ''; }
  } catch (error) { $('ingest-result').textContent = error.message; }
  finally { $('submit-ingest').disabled = false; $('cancel-ingest').disabled = false; }
});

// Download a Blob without navigating away from an active editor.
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

async function exportReviewed() {
  $('export-button').disabled = true;
  try {
    const response = await fetch('/api/export', {headers: {'X-Session': state.token}});
    if (!response.ok) { const result = await response.json(); throw new Error(result.detail); }
    downloadBlob(await response.blob(), 'fridge-reviewed.zip');
    toast('Export downloaded. Images currently being edited are excluded.');
  } finally { $('export-button').disabled = false; }
}

// ---------------------------------------------------------
// JOINING, KEYBOARD SHORTCUTS AND STARTUP
// ---------------------------------------------------------
$('join-dialog').addEventListener('cancel', event => event.preventDefault());
$('join-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const editor = await api('/api/editors', 'POST', {name: $('editor-name').value});
    state.token = editor.token; state.name = editor.name;
    sessionStorage.setItem('fridge-token', editor.token); sessionStorage.setItem('fridge-name', editor.name);
    localStorage.setItem('fridge-name', editor.name);
    $('user-badge').textContent = editor.name;
    $('join-dialog').close();
    await refreshWorkspace();
  } catch (error) { toast(error.message, true); }
});
$('category-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const result = await api('/api/categories', 'POST', {name: $('new-category').value});
    await refreshWorkspace();
    state.nextCategory = result.name;
    $('category-select').value = result.name;
    if (canEdit()) $('category-select').dispatchEvent(new Event('change'));
    $('category-dialog').close(); $('new-category').value = '';
    toast(`Category ${result.name} added for everyone.`);
  } catch (error) { toast(error.message, true); }
});

action('pending-tab', () => { state.tab = 'pending'; renderQueue(); });
action('reviewed-tab', () => { state.tab = 'reviewed'; renderQueue(); });
$('shelf-filter').addEventListener('change', renderQueue);
action('next-button', () => openImage()); action('empty-next', () => openImage());
action('leave-button', async () => {
  if (state.navigating || state.finalizing) return;
  state.navigating = true;
  updateControls();
  try { await closeImage(); }
  finally { state.navigating = false; updateControls(); }
});
action('review-next', () => review(true)); action('review-only', () => review(false));
action('select-tool', () => setTool('select')); action('draw-tool', () => setTool('draw'));
action('undo-button', () => undo()); action('redo-button', () => undo(true));
action('delete-box', deleteSelected);
action('zoom-in', () => { state.zoom = Math.min(8, state.zoom * 1.3); drawCanvas(); });
action('zoom-out', () => { state.zoom = Math.max(0.4, state.zoom / 1.3); drawCanvas(); });
action('fit-button', () => { state.zoom = 1; fitCanvas(); });
action('upload-button', () => openIngest('upload')); action('capture-button', () => openIngest('capture'));
action('cancel-ingest', () => $('ingest-dialog').close());
action('export-button', exportReviewed);
action('add-category-button', () => $('category-dialog').showModal());
action('cancel-category', () => $('category-dialog').close());
action('download-draft', () => downloadBlob(new Blob([JSON.stringify({image_id: state.current?.id, revision: state.current?.revision, boxes: state.boxes, notes: $('notes').value}, null, 2)], {type: 'application/json'}), 'unsaved-draft.json'));

document.addEventListener('keydown', event => {
  if (event.defaultPrevented) return;
  if (document.querySelector('dialog[open]') || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
  if (!canEdit()) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(event.shiftKey); }
  else if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); review(true).catch(error => toast(error.message, true)); }
  else if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); deleteSelected(); }
  else if (event.key.toLowerCase() === 'd') setTool('draw');
  else if (event.key.toLowerCase() === 'v') setTool('select');
});
window.addEventListener('beforeunload', event => {
  if (state.version !== state.savedVersion && state.current) { event.preventDefault(); event.returnValue = ''; }
});
new ResizeObserver(fitCanvas).observe($('canvas-viewport'));
$('editor-name').value = localStorage.getItem('fridge-name') || '';
$('user-badge').textContent = state.name || '';
if (!state.token) $('join-dialog').showModal();
updateControls();
refreshWorkspace();
setInterval(refreshWorkspace, 2000);
setInterval(heartbeat, 15000);
