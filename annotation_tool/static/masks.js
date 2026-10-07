// ---------------------------------------------------------
// DRAW AND CORRECT ONE VISIBLE OUTLINE PER PRODUCT
// ---------------------------------------------------------
function maskBounds(points) {
  return [Math.min(...points.map(p => p[0])), Math.min(...points.map(p => p[1])),
    Math.max(...points.map(p => p[0])), Math.max(...points.map(p => p[1]))];
}

// Select the visible shape rather than another object's overlapping box.
function maskContains(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a[1] > point[1]) !== (b[1] > point[1]) && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

// Start a new object, or replace only the selected object's outline.
function startPolygon(replace = false) {
  if (!canEdit() || state.drag || state.polyPoints) return;
  state.polyPoints = [];
  state.polyTarget = replace ? state.selected : null;
  state.selectedVertex = null;
  setTool('polygon');
  $('image-canvas').style.cursor = 'crosshair';
  updateMaskControls();
}

// Closing an outline is one undoable edit; the enclosing box follows it.
function finishPolygon() {
  if (!canEdit() || !state.polyPoints) return;
  if (state.polyPoints.length < 3) { toast('Click at least three outline points.', true); return; }
  const points = state.polyPoints;
  const bounds = maskBounds(points);
  if (bounds[2] - bounds[0] < 1 || bounds[3] - bounds[1] < 1) { toast('Draw a non-empty outline.', true); return; }
  remember();
  const target = state.boxes.find(box => box.id === state.polyTarget);
  if (target) { target.polygon = points; target.xyxy = bounds; state.selected = target.id; }
  else {
    const box = {id: crypto.randomUUID(), category: state.nextCategory, xyxy: bounds, polygon: points};
    state.boxes.push(box); state.selected = box.id;
  }
  state.polyPoints = null; state.polyTarget = null;
  setTool('select'); changed();
}

function cancelPolygon() {
  state.polyPoints = null; state.polyTarget = null;
  setTool('select'); drawCanvas(); updateMaskControls();
}

// Find how close a click is to an outline edge.
function edgeDistance(point, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy);
}

// Outline clicks take priority over ordinary box dragging.
function maskPointerDown(event, point) {
  const radius = 8 / (state.fitScale * state.zoom);
  if (state.polyPoints) {
    if (state.polyPoints.length >= 3 && Math.hypot(point[0] - state.polyPoints[0][0], point[1] - state.polyPoints[0][1]) < radius) finishPolygon();
    else if (state.polyPoints.length < 500) {
      const last = state.polyPoints.at(-1);
      if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) > 0.5) state.polyPoints.push(point);
      drawCanvas();
    }
    return true;
  }
  const box = state.boxes.find(item => item.id === state.selected);
  if (state.tool !== 'select' || !box?.polygon) return false;
  const vertex = box.polygon.findIndex(p => Math.hypot(p[0] - point[0], p[1] - point[1]) < radius);
  if (vertex >= 0) {
    state.selectedVertex = vertex;
    if (event.altKey) { removeVertex(); return true; }
    state.drag = {kind: 'vertex', vertex, original: structuredClone(state.boxes)};
    clearTimeout(state.timer);
    $('image-canvas').setPointerCapture(event.pointerId);
    updateMaskControls();
    return true;
  }
  if (event.shiftKey) {
    const distances = box.polygon.map((p, i) => edgeDistance(point, p, box.polygon[(i + 1) % box.polygon.length]));
    const edge = distances.indexOf(Math.min(...distances));
    if (distances[edge] < radius && box.polygon.length < 500) {
      remember(); box.polygon.splice(edge + 1, 0, point);
      box.xyxy = maskBounds(box.polygon); state.selectedVertex = edge + 1; changed();
      return true;
    }
  }
  state.selectedVertex = null;
  return false;
}

function removeVertex() {
  const box = state.boxes.find(item => item.id === state.selected);
  if (!canEdit() || !box?.polygon || state.selectedVertex == null) return;
  if (box.polygon.length <= 3) { toast('Keep at least three points.', true); return; }
  remember(); box.polygon.splice(state.selectedVertex, 1);
  box.xyxy = maskBounds(box.polygon); state.selectedVertex = null; changed();
}

// Show the in-progress outline without saving an unfinished shape.
function drawMaskDraft(context, scale) {
  if (!state.polyPoints?.length) return;
  context.beginPath();
  state.polyPoints.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
  context.strokeStyle = '#008cff'; context.lineWidth = 2 / scale; context.stroke();
  context.fillStyle = '#008cff';
  state.polyPoints.forEach(([x, y]) => context.fillRect(x - 3 / scale, y - 3 / scale, 6 / scale, 6 / scale));
}

// Show the actual visible pixels the classifier will receive after masking.
function renderMaskPreview() {
  const box = state.boxes.find(item => item.id === state.selected);
  const preview = $('mask-preview');
  preview.hidden = !box?.polygon || !state.image;
  if (preview.hidden) return;
  const [x1, y1, x2, y2] = box.xyxy;
  const dx = (x2 - x1) * .05, dy = (y2 - y1) * .05;
  const left = Math.max(0, Math.floor(x1 - dx)), top = Math.max(0, Math.floor(y1 - dy));
  const right = Math.min(state.image.width, Math.ceil(x2 + dx)), bottom = Math.min(state.image.height, Math.ceil(y2 + dy));
  const width = right - left, height = bottom - top;
  if (width <= 0 || height <= 0) return;
  const context = preview.getContext('2d');
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.fillStyle = '#7f7f7f'; context.fillRect(0, 0, preview.width, preview.height);
  const scale = Math.min(preview.width / width, preview.height / height);
  context.save();
  context.translate((preview.width - width * scale) / 2, (preview.height - height * scale) / 2);
  context.scale(scale, scale); context.translate(-left, -top);
  context.beginPath();
  box.polygon.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
  context.closePath(); context.clip(); context.drawImage(state.image, 0, 0); context.restore();
}

function updateMaskControls() {
  const box = state.boxes.find(item => item.id === state.selected);
  const drawing = state.polyPoints !== null;
  $('polygon-tool').disabled = !canEdit() || drawing;
  $('finish-polygon').hidden = $('cancel-polygon').hidden = !drawing;
  $('finish-polygon').disabled = !canEdit();
  $('suggest-masks').disabled = !canEdit() || drawing || !state.workspace?.model_available;
  $('mask-controls').hidden = !box;
  $('redraw-mask').disabled = !canEdit() || drawing;
  $('remove-mask').disabled = !canEdit() || !box?.polygon || drawing;
  $('remove-point').disabled = !canEdit() || !box?.polygon || state.selectedVertex == null || drawing;
  $('masks-reviewed').disabled = !canEdit() || drawing;
  if (drawing) { $('review-next').disabled = true; $('review-only').disabled = true; }
  renderMaskPreview();
}

// Explicitly replace this editor's draft after requesting fresh model suggestions.
async function suggestMasks() {
  if (!canEdit() || state.polyPoints || state.drag) return;
  if (state.boxes.length && !confirm('Replace the current draft with model suggestions? You can undo this.')) return;
  state.finalizing = true; updateControls();
  try {
    await persistDraft();
    $('save-state').textContent = 'Segmenting shelf…';
    const result = await api(`/api/images/${state.current.id}/suggest`, 'POST');
    if (!result.suggestions.length) { toast('No objects found. Draw masks manually.'); return; }
    remember();
    state.current.suggestions = result.suggestions;
    state.boxes = result.suggestions.map((record, i) => ({id: `model-${i}`, category: record.category,
      xyxy: record.box_xyxy, polygon: record.polygon}));
    state.selected = state.boxes[0]?.id || null; state.selectedVertex = null;
    await refreshWorkspace();
    changed();
    await persistDraft();
  } catch (error) {
    if (error.status === 409) loseLease(error.message);
    throw error;
  } finally {
    state.finalizing = false; updateControls();
    if (state.version === state.savedVersion) $('save-state').textContent = 'Draft saved';
  }
}

// ---------------------------------------------------------
// SIMPLE BUTTONS AND POLYGON KEYBOARD SHORTCUTS
// ---------------------------------------------------------
action('polygon-tool', () => startPolygon());
action('redraw-mask', () => startPolygon(true));
action('finish-polygon', finishPolygon);
action('cancel-polygon', cancelPolygon);
action('remove-point', removeVertex);
action('remove-mask', () => {
  const box = state.boxes.find(item => item.id === state.selected);
  if (!canEdit() || !box?.polygon) return;
  remember(); delete box.polygon; changed();
});
action('suggest-masks', suggestMasks);
document.addEventListener('keydown', event => {
  if (document.querySelector('dialog[open]') || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
  if (!canEdit()) return;
  if (state.polyPoints) {
    event.preventDefault(); event.stopImmediatePropagation();
    if (event.key === 'Enter') finishPolygon();
    else if (event.key === 'Escape') cancelPolygon();
    else if (event.key === 'Backspace') { state.polyPoints.pop(); drawCanvas(); }
  } else if (event.key.toLowerCase() === 'p') { event.preventDefault(); startPolygon(); }
}, true);
updateMaskControls();
