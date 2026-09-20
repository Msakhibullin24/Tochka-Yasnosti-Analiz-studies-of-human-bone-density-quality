"""Portable, self-contained report of a specific append-only review revision."""
import base64
from html import escape
from .result_context import model_verdict, assessment_notes


def render_review(row, detail, review, png, fragment=False):
    def e(value):
        return escape(str(value), quote=True)

    width, height = detail["width"] - 1, detail["height"] - 1
    shapes, notes = [], []
    for index, g in enumerate(review["geometry"], 1):
        points = [(p["x"] * width, p["y"] * height) for p in g["points"]]
        if g["kind"] == "polygon":
            points.append(points[0])
        if len(points) > 1:
            coords = " ".join(f"{x},{y}" for x, y in points)
            shapes.append(f'<polyline points="{coords}"/>')
        x, y = points[0]
        shapes.append(f'<circle cx="{x}" cy="{y}" r="2"/><text x="{x+3}" y="{y-3}" fill="#63ffdb" stroke="none" font-size="8">{index}</text>')
        notes.append(f'<li><strong>{e(g["name"])}</strong>: {e(g.get("note", "") or "Без замечания")}</li>')
    image = base64.b64encode(png).decode("ascii")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width * detail["pixel_mm_x"]} {height * detail["pixel_mm_y"]}" role="img" aria-label="Снимок с разметкой версии {review["revision"]}">'
           f'<g transform="scale({detail["pixel_mm_x"]} {detail["pixel_mm_y"]})"><image href="data:image/png;base64,{image}" x="-0.5" y="-0.5" width="{detail["width"]}" height="{detail["height"]}"/>'
           '<g fill="none" stroke="#63ffdb" stroke-width="1">' + ''.join(shapes) + '</g></g></svg>')
    status = {"draft": "Черновик", "confirmed": "Подтверждено", "not_evaluable": "Невозможно оценить"}[review["status"]]
    verdict = "Невозможно оценить" if review["quality_class"] is None else "Есть нарушения" if review["quality_class"] else "Нарушений нет"
    actions = ''.join(f'<li>{e(a["question"])} — {"Открыто" if a["state"] == "open" else "Выполнено"}. {e(a["resolution"])}</li>' for a in review.get("followups", []))
    measures = ''.join(f'<li>{e(name)}: {e(m["length_mm"])} мм; {e(m["angle_from_vertical_deg"])}° от вертикали</li>' for name, m in review.get("measurements", {}).items())
    roi_checks = ''.join(f'<li>{e(r["name"])}: {e(r["reason"])} Отступы, мм: {e(r.get("margins_mm", {}))}</li>' for r in review.get("roi_evaluation", []))
    header = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Osseo AI — версия проверки</title>'
            '<style>body{font:16px system-ui;color:#14213a;max-width:1000px;margin:24px auto;padding:16px}svg{display:block;max-width:100%;max-height:650px;background:#10151c}p,li{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}@media print{svg{max-height:180mm}h2{break-after:avoid}}</style>')
    body = (
            f'<h1>Проверка изображения · версия {review["revision"]}</h1><p>{e(row["path_to_file"])}</p>'
            f'<p>Область: {e(row["anatomical_region"])}. Автор: {e(review["author"])}. Статус: {status}.</p>'
            '<p>Имя автора введено пользователем; документ не содержит электронной подписи. Черновик не является подтверждённым решением.</p>'
            f'<h2>Результат модели</h2><p>{e(model_verdict(row))}</p>'
            + '<h2>Ограничения автоматической проверки</h2><ul>' + ''.join('<li>' + e(note) + '</li>' for note in assessment_notes(row)) + '</ul>'
            + f'<h2>Решение проверяющего</h2><p>{verdict}</p><p>{e(review["comment"])}</p>'
            + svg + '<h2>Отметки на снимке</h2><ol>' + ''.join(notes) + '</ol>'
            '<h2>Измерения сохранённой версии</h2><ul>' + measures + '</ul>'
            f'<p>Источник масштаба: {e(review.get("measurement_scale_source", "не указан"))}. Редактирование разметки не пересчитывает BMD.</p>'
            '<h2>Отступы выбранных ROI</h2><ul>' + roi_checks + '</ul>'
            '<h2>Последующие действия</h2><ul>' + actions + '</ul>')
    return body if fragment else header + body + '</html>'
