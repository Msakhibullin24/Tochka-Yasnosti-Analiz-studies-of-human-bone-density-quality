/* Offline blind annotation. No model coordinates or decisions are imported. */
(() => {
  'use strict';
  const original = JSON.parse(document.getElementById('review-data').textContent);
  let data = structuredClone(original), index = 0;
  const el = id => document.getElementById(id);
  const names = {Th12: 'Th12 — центр тела', iliac_crest_left: 'Левый подвздошный гребень — верхний край',
    iliac_crest_right: 'Правый подвздошный гребень — верхний край',
    greater_trochanter: 'Большой вертел — верхний край', lesser_trochanter: 'Малый вертел',
    femoral_neck: 'Шейка бедра — центр', ischium: 'Седалищная кость — нижний край'};
  const message = (text, error = false) => { el('message').textContent = text; el('message').dataset.error = String(error); };
  const current = () => data.cases[index];
  const selected = () => current().landmarks[el('landmark').value];
  const hasPoint = (point, c) => Array.isArray(point) && point.length === 2 &&
    point.every(Number.isFinite) && point[0] >= 0 && point[1] >= 0 && point[0] < c.width && point[1] < c.height;
  const rotations = ['none', 'insufficient', 'excessive', 'not_assessable'];
  function validPolygon(polygon, c, concave = false) {
    if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 128 || !polygon.every(p => hasPoint(p, c))) return false;
    if (concave) {
      const cross = (a, b, c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
      const on = (a, b, c) => Math.abs(cross(a,b,c)) <= 1e-8 && [0,1].every(k => c[k] >= Math.min(a[k],b[k])-1e-8 && c[k] <= Math.max(a[k],b[k])+1e-8);
      const intersects = (a,b,c,d) => (cross(a,b,c)*cross(a,b,d) < 0 && cross(c,d,a)*cross(c,d,b) < 0) || on(a,b,c) || on(a,b,d) || on(c,d,a) || on(c,d,b);
      if (new Set(polygon.map(p => p.join(','))).size !== polygon.length) return false;
      for (let i=0; i<polygon.length; i++) {
        const a=polygon[i], b=polygon[(i+1)%polygon.length];
        if (Math.abs(cross(polygon[(i+polygon.length-1)%polygon.length],a,b)) <= 1e-8 && on(polygon[(i+polygon.length-1)%polygon.length],a,b)) return false;
        for (let j=i+1; j<polygon.length; j++) {
          if (j===i+1 || (i===0 && j===polygon.length-1)) continue;
          if (intersects(a,b,polygon[j],polygon[(j+1)%polygon.length])) return false;
        }
      }
      return Math.abs(polygon.reduce((area,p,i) => { const q=polygon[(i+1)%polygon.length]; return area+p[0]*q[1]-p[1]*q[0]; },0)) > 1e-8;
    }
    const signs = polygon.map((p, i) => {
      const q = polygon[(i + 1) % polygon.length], r = polygon[(i + 2) % polygon.length];
      return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
    });
    // Every edge must keep all other vertices on the same side; also excludes star polygons.
    const sign = Math.sign(signs.find(v => Math.abs(v) > 1e-8) || 0);
    if (!sign) return false;
    return polygon.every((p, i) => {
      const q = polygon[(i + 1) % polygon.length];
      return (p[0] !== q[0] || p[1] !== q[1]) && polygon.every(r =>
        sign * ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) >= -1e-8);
    });
  }
  const validRegions = c => !c.named_regions || Object.entries(c.named_regions).every(([name,r]) =>
    r.visible === false ? r.polygon === null : r.visible === true && validPolygon(r.polygon, c, original.review_package_version >= 3 && name !== 'femoral_neck_roi'));
  const validAxis = c => !c.axes || Object.values(c.axes).every(a => a.visible === false ? a.points === null : a.visible === true && Array.isArray(a.points) && a.points.length === 2 && a.points.every(p => hasPoint(p,c)) && a.points[0].some((v,i) => v !== a.points[1][i]));
  function validFullQuad(points) {
    if (!Array.isArray(points) || points.length !== 4 || !points.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) return false;
    const cross = (a,b,c) => (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);
    const signs = points.map((p,i) => cross(p,points[(i+1)%4],points[(i+2)%4]));
    return signs.every(s => s > 1e-8) || signs.every(s => s < -1e-8);
  }
  const validAmodal = c => !Object.hasOwn(c,'th12_full_body') ||
    c.th12_full_body.status === 'unassessable' && c.th12_full_body.quadrilateral === null ||
    c.th12_full_body.status === 'assessable' && validFullQuad(c.th12_full_body.quadrilateral);
  const validCase = c => ['frontal', 'lateral', 'other'].includes(c.projection) &&
    Object.values(c.landmarks).every(l => l.visible === false ? l.point === null : l.visible === true && hasPoint(l.point, c)) &&
    validRegions(c) && validAxis(c) && validAmodal(c) && (!Object.hasOwn(c, 'rotation') || rotations.includes(c.rotation));
  function changed() { current().status = 'unreviewed'; el('state').textContent = 'Черновик — разметка не подтверждена'; progress(); }
  function progress() { el('progress').textContent = `Изображение ${index + 1} из ${data.cases.length}. Подтверждено: ${data.cases.filter(c => c.status === 'confirmed').length}.`; }
  function markers() {
    el('markers').replaceChildren();
    const full=current().th12_full_body?.quadrilateral;
    if (Array.isArray(full) && full.length) {
      const polygon=document.createElementNS('http://www.w3.org/2000/svg','polygon');
      polygon.setAttribute('points',full.map(([x,y]) => `${x+.5},${y+.5}`).join(' '));
      polygon.setAttribute('fill','none'); polygon.setAttribute('stroke','#ffb000');
      polygon.setAttribute('stroke-dasharray','5 3'); polygon.setAttribute('stroke-width',String(Math.max(1,current().width/200)));
      el('markers').append(polygon);
    }
    Object.entries(current().named_regions || {}).forEach(([name, r]) => {
      if (!Array.isArray(r.polygon) || !r.polygon.length) return;
      const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      polygon.setAttribute('points', r.polygon.map(([x, y]) => `${x + .5},${y + .5}`).join(' '));
      polygon.setAttribute('fill', 'none'); polygon.setAttribute('stroke', '#00ffff');
      polygon.setAttribute('stroke-width', String(Math.max(1, current().width / 200)));
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title'); title.textContent = name; polygon.append(title);
      el('markers').append(polygon);
    });
    Object.entries(current().axes || {}).forEach(([name,a]) => {
      if (!Array.isArray(a.points) || !a.points.every(p => hasPoint(p,current()))) return;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ['x1','y1','x2','y2'].forEach((key,i) => line.setAttribute(key,String(a.points[Math.floor(i/2)][i%2]+.5)));
      line.setAttribute('stroke','#ff88ff'); line.setAttribute('stroke-width',String(Math.max(1,current().width/200)));
      const title=document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent=name; line.append(title); el('markers').append(line);
    });
    Object.entries(current().landmarks).forEach(([name, l]) => {
      if (!hasPoint(l.point, current())) return;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', l.point[0] + .5); circle.setAttribute('cy', l.point[1] + .5);
      circle.setAttribute('r', Math.max(2, current().width / 150)); circle.setAttribute('fill', name === el('landmark').value ? '#ffdd00' : '#fff');
      circle.setAttribute('stroke', '#111'); circle.setAttribute('stroke-width', '1');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title'); title.textContent = names[name] || name; circle.append(title);
      el('markers').append(circle);
    });
  }
  function landmark() {
    const l = selected();
    document.querySelector(`input[name="visible"][value="${l.visible === true ? 'yes' : l.visible === false ? 'no' : 'unknown'}"]`).checked = true;
    el('x').value = l.point?.[0] ?? ''; el('y').value = l.point?.[1] ?? '';
    el('x').max = current().width - .000001; el('y').max = current().height - .000001;
    el('landmark-state').textContent = l.visible === false ? 'Не видим; координаты отсутствуют' : l.point ? `Точка: ${l.point.join(', ')}` : 'Точка не указана';
    markers();
  }
  function render() {
    const c = current();
    el('case').value = String(index); el('projection').value = c.projection || '';
    el('state').textContent = c.status === 'confirmed' ? 'Разметка подтверждена' : 'Черновик — разметка не подтверждена';
    el('viewer').setAttribute('viewBox', `0 0 ${c.width} ${c.height}`);
    el('image').setAttribute('href', c.image); el('image').setAttribute('width', c.width); el('image').setAttribute('height', c.height);
    el('landmark').replaceChildren(...Object.keys(c.landmarks).map(name => { const o = document.createElement('option'); o.value = name; o.textContent = names[name] || name; return o; }));
    el('prev').disabled = index === 0; el('next').disabled = index === data.cases.length - 1;
    el('region-editor').hidden = !c.named_regions; el('mode').disabled = !c.named_regions; el('mode').value = 'points';
    el('region').replaceChildren(...Object.keys(c.named_regions || {}).map(name => { const o = document.createElement('option'); o.value = name; o.textContent = ({femoral_neck_roi:'ROI шейки бедра', femur:'Бедренная кость', greater_trochanter:'Большой вертел', lesser_trochanter:'Малый вертел', ischium:'Седалищная кость'})[name] || `Тело ${name}`; return o; }));
    el('rotation-label').hidden = !Object.hasOwn(c, 'rotation'); el('rotation').value = c.rotation || '';
    el('axis-editor').hidden = !c.axes;
    el('mode').querySelector('[value=axes]').disabled = !c.axes;
    el('amodal-editor').hidden = !c.th12_full_body;
    el('mode').querySelector('[value=amodal]').disabled = !c.th12_full_body;
    axisFields(); regionFields(); amodalFields(); landmark(); progress(); message('');
  }
  data.cases.forEach((c, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = `Изображение ${i + 1} — ${Object.hasOwn(c.landmarks, 'Th12') ? 'позвоночник' : 'бедро'}`; el('case').append(o); });
  el('case').addEventListener('change', () => { index = Number(el('case').value); render(); });
  for (const [id, step] of [['prev', -1], ['next', 1]]) el(id).addEventListener('click', () => { index += step; render(); });
  el('landmark').addEventListener('change', landmark);
  el('projection').addEventListener('change', () => { current().projection = el('projection').value || null; changed(); });
  document.querySelectorAll('input[name="visible"]').forEach(input => input.addEventListener('change', () => {
    const l = selected(); l.visible = input.value === 'yes' ? true : input.value === 'no' ? false : null;
    if (l.visible !== true) l.point = null;
    changed(); landmark();
  }));
  function setPoint(point) {
    if (selected().visible !== true) return message('Сначала выберите «Видим».', true);
    if (!hasPoint(point, current())) return message('Укажите обе координаты внутри исходного изображения.', true);
    selected().point = point; changed(); landmark(); message('Точка сохранена в черновике.');
  }
  el('viewer').addEventListener('click', event => {
    const transform = el('viewer').getScreenCTM();
    if (!transform) return;
    const point = el('viewer').createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    const p = point.matrixTransform(transform.inverse());
    // Image extents are [0,width], but pixel centres are [0,width-1].
    if (p.x < 0 || p.y < 0 || p.x >= current().width || p.y >= current().height) return;
    const coordinates = [Math.max(0, Math.min(current().width - 1, p.x - .5)), Math.max(0, Math.min(current().height - 1, p.y - .5))];
    if (el('mode').value === 'regions') addVertex(coordinates); else if (el('mode').value === 'axes') setAxisPoint(coordinates); else if (el('mode').value === 'amodal') addAmodalPoint(coordinates); else setPoint(coordinates);
  });
  el('point').addEventListener('click', () => {
    if (!el('x').value.trim() || !el('y').value.trim()) return message('Введите X и Y.', true);
    setPoint([Number(el('x').value), Number(el('y').value)]);
  });
  el('clear').addEventListener('click', () => { selected().point = null; changed(); landmark(); });
  function selectedRegion() { return current().named_regions?.[el('region').value]; }
  function regionFields() {
    const r = selectedRegion(); if (!r) return;
    el('region-visible').value = r.visible === true ? 'yes' : r.visible === false ? 'no' : 'unknown';
    el('region-state').textContent = r.visible === false ? 'Область не видима' : `Вершин: ${r.polygon?.length || 0}. ${validPolygon(r.polygon, current(), original.review_package_version >= 3 && el('region').value !== 'femoral_neck_roi') ? 'Контур готов' : 'Контур не завершён'}`;
    markers();
  }
  function addVertex(point) {
    const r = selectedRegion();
    if (!r || r.visible !== true) return message('Выберите видимую область.', true);
    if (!hasPoint(point, current())) return message('Вершина должна находиться внутри изображения.', true);
    if ((r.polygon?.length || 0) >= 128) return message('Для области допускается до 128 вершин.', true);
    r.polygon = [...(r.polygon || []), point]; changed(); regionFields();
  }
  el('region').addEventListener('change', regionFields);
  el('region-visible').addEventListener('change', () => {
    const r = selectedRegion(); if (!r) return;
    r.visible = el('region-visible').value === 'yes' ? true : el('region-visible').value === 'no' ? false : null;
    if (r.visible !== true) r.polygon = null;
    changed(); regionFields();
  });
  el('vertex').addEventListener('click', () => {
    if (!el('rx').value.trim() || !el('ry').value.trim()) return message('Введите координаты вершины X и Y.', true);
    addVertex([Number(el('rx').value), Number(el('ry').value)]);
  });
  el('undo-vertex').addEventListener('click', () => {
    const r = selectedRegion(); if (!r?.polygon?.length) return;
    r.polygon.pop(); if (!r.polygon.length) r.polygon = null;
    changed(); regionFields();
  });
  function axisFields() {
    const a=current().axes?.femoral_neck_axis; if (!a) return;
    el('axis-visible').value=a.visible === true ? 'yes' : a.visible === false ? 'no' : 'unknown';
    const p=a.points?.[Number(el('axis-end').value)]; el('ax').value=p?.[0] ?? ''; el('ay').value=p?.[1] ?? '';
    el('axis-state').textContent=validAxis(current()) ? 'Ось оценена' : 'Ось не завершена'; markers();
  }
  function setAxisPoint(point) {
    const a=current().axes?.femoral_neck_axis;
    if (!a || a.visible !== true) return message('Сначала выберите видимую ось.',true);
    if (!hasPoint(point,current())) return message('Конец оси должен находиться внутри изображения.',true);
    a.points ||= [null,null]; a.points[Number(el('axis-end').value)]=point; changed(); axisFields();
  }
  el('axis-end').addEventListener('change',axisFields);
  el('axis-visible').addEventListener('change',() => {
    const a=current().axes?.femoral_neck_axis; if (!a) return;
    a.visible=el('axis-visible').value === 'yes' ? true : el('axis-visible').value === 'no' ? false : null;
    if (a.visible !== true) a.points=null;
    changed(); axisFields();
  });
  el('axis-point').addEventListener('click',() => {
    if (!el('ax').value.trim() || !el('ay').value.trim()) return message('Введите координаты конца оси X и Y.',true);
    setAxisPoint([Number(el('ax').value),Number(el('ay').value)]);
  });
  function amodalFields() {
    const full=current().th12_full_body; if (!full) return;
    el('amodal-status').value=full.status || '';
    el('amodal-state').textContent=full.status === 'unassessable' ? 'Полная граница не определяется' :
      `Углов: ${full.quadrilateral?.length || 0}/4. ${validAmodal(current()) ? 'Граница готова' : 'Граница не завершена'}`;
    markers();
  }
  function addAmodalPoint(point) {
    const full=current().th12_full_body;
    if (!full || full.status !== 'assessable') return message('Сначала выберите «Можно восстановить».',true);
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite))
      return message('Укажите конечные координаты угла.',true);
    if ((full.quadrilateral?.length || 0) >= 4) return message('Допускается ровно четыре угла.',true);
    full.quadrilateral=[...(full.quadrilateral || []),point]; changed(); amodalFields();
  }
  el('amodal-status').addEventListener('change',() => {
    const full=current().th12_full_body; if (!full) return;
    full.status=el('amodal-status').value || null;
    if (full.status !== 'assessable') full.quadrilateral=null;
    changed(); amodalFields();
  });
  el('amodal-point').addEventListener('click',() => {
    if (!el('amodal-x').value.trim() || !el('amodal-y').value.trim()) return message('Введите X и Y угла.',true);
    addAmodalPoint([Number(el('amodal-x').value),Number(el('amodal-y').value)]);
  });
  el('amodal-undo').addEventListener('click',() => {
    const full=current().th12_full_body; if (!full?.quadrilateral?.length) return;
    full.quadrilateral.pop(); if (!full.quadrilateral.length) full.quadrilateral=null;
    changed(); amodalFields();
  });
  el('rotation').addEventListener('change', () => { if (Object.hasOwn(current(), 'rotation')) { current().rotation = el('rotation').value || null; changed(); } });
  el('confirm').addEventListener('click', () => {
    if (!validCase(current())) return message('Выберите проекцию и оцените все ориентиры и области. Видимые ориентиры требуют точки, видимые области — завершённого контура, ось — двух разных точек. Оцените полную границу Th12 или укажите, что это невозможно. Оцените ротацию бедра, если поле доступно.', true);
    current().status = 'confirmed'; el('state').textContent = 'Разметка подтверждена'; progress(); message('Разметка изображения подтверждена.');
  });
  function exportData(confirmed) {
    const result = structuredClone(data);
    result.reviewer = el('reviewer').value.trim(); result.annotation_source = el('source').value.trim(); result.independent_of_predictions = el('independent').checked;
    if (confirmed) {
      if (!result.reviewer || !result.annotation_source || !result.independent_of_predictions) { message('Укажите специалиста, основание разметки и подтвердите её независимость.', true); return null; }
      result.cases = result.cases.filter(c => c.status === 'confirmed' && validCase(c));
      if (!result.cases.length) { message('Сначала подтвердите разметку хотя бы одного изображения.', true); return null; }
    }
    return result;
  }
  function download(confirmed) {
    const result = exportData(confirmed); if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = confirmed ? 'anatomy-reference.json' : 'anatomy-draft.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    message(confirmed ? `Экспортировано подтверждённых изображений: ${result.cases.length}.` : 'Черновик скачан. Сохраните файл для продолжения работы.');
  }
  el('draft').addEventListener('click', () => download(false)); el('export').addEventListener('click', () => download(true));
  el('import').addEventListener('change', async () => {
    try {
      const file = el('import').files[0]; if (!file) return;
      if (file.size > 10_000_000) throw new Error('Файл превышает 10 МБ.');
      const incoming = JSON.parse(await file.text());
      if (incoming.version !== 1 || incoming.review_package_version !== original.review_package_version || incoming.results_sha256 !== original.results_sha256 || !Array.isArray(incoming.cases) || !incoming.cases.length) throw new Error('Черновик относится к другому пакету или имеет неверный формат.');
      const cases = structuredClone(original.cases), byId = new Map(cases.map(c => [c.image_uid, c])), seen = new Set();
      for (const c of incoming.cases) {
        const dest = byId.get(c.image_uid);
        if (!dest || seen.has(c.image_uid) || c.source_sha256 !== dest.source_sha256 || c.pixel_sha256 !== dest.pixel_sha256 || !['unreviewed', 'confirmed'].includes(c.status) || ![null, 'frontal', 'lateral', 'other'].includes(c.projection)) throw new Error('Изображения черновика не совпадают с пакетом.');
        seen.add(c.image_uid);
        if (!c.landmarks || Object.keys(c.landmarks).sort().join() !== Object.keys(dest.landmarks).sort().join()) throw new Error('Неверный список ориентиров.');
        for (const l of Object.values(c.landmarks)) {
          if (![true, false, null].includes(l.visible) || (l.point !== null && (!hasPoint(l.point, dest) || l.visible !== true))) throw new Error('Неверные координаты или видимость ориентира.');
        }
        if (dest.named_regions) {
          if (!c.named_regions || Object.keys(c.named_regions).sort().join() !== Object.keys(dest.named_regions).sort().join()) throw new Error('Неверный список областей.');
          for (const r of Object.values(c.named_regions)) {
            if (![true, false, null].includes(r.visible) || (r.polygon !== null && (r.visible !== true || !Array.isArray(r.polygon) || r.polygon.length > 128 || !r.polygon.every(p => hasPoint(p, dest))))) throw new Error('Неверные вершины контура.');
          }
          dest.named_regions = structuredClone(c.named_regions);
        }
        if (dest.axes) {
          if (!c.axes || Object.keys(c.axes).join() !== Object.keys(dest.axes).join()) throw new Error('Неверный список осей.');
          const a=c.axes.femoral_neck_axis;
          if (![true,false,null].includes(a.visible) || (a.points !== null && (a.visible !== true || !Array.isArray(a.points) || a.points.length !== 2 || !a.points.every(p => p === null || hasPoint(p,dest))))) throw new Error('Неверные концы оси.');
          dest.axes=structuredClone(c.axes);
        }
        if (dest.th12_full_body) {
          if (!c.th12_full_body || ![null,'assessable','unassessable'].includes(c.th12_full_body.status) ||
              c.th12_full_body.quadrilateral !== null &&
              (!Array.isArray(c.th12_full_body.quadrilateral) || c.th12_full_body.quadrilateral.length > 4 ||
               !c.th12_full_body.quadrilateral.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))))
            throw new Error('Неверная полная граница Th12.');
          dest.th12_full_body=structuredClone(c.th12_full_body);
        }
        if (Object.hasOwn(dest, 'rotation')) {
          if (c.rotation !== null && !rotations.includes(c.rotation)) throw new Error('Неверная метка ротации.');
          dest.rotation = c.rotation;
        }
        dest.landmarks = structuredClone(c.landmarks); dest.projection = c.projection;
        if (c.status === 'confirmed' && !validCase(dest)) throw new Error('Подтверждённое изображение содержит незавершённую разметку.');
        dest.status = c.status;
      }
      data = {...structuredClone(original), cases};
      el('reviewer').value = typeof incoming.reviewer === 'string' ? incoming.reviewer : '';
      el('source').value = typeof incoming.annotation_source === 'string' ? incoming.annotation_source : '';
      el('independent').checked = incoming.independent_of_predictions === true;
      render(); message('Черновик загружен.');
    } catch (error) { message(`Не удалось загрузить черновик: ${error.message}`, true); }
    finally { el('import').value = ''; }
  });
  window.addEventListener('beforeunload', event => { if (data.cases.some(c => c.projection || Object.values(c.landmarks).some(l => l.visible !== null) || Object.values(c.named_regions || {}).some(r => r.visible !== null) || c.rotation || c.th12_full_body?.status)) { event.preventDefault(); event.returnValue = ''; } });
  render();
})();
