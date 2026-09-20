"""Geometric margins of a user-selected ROI, not anatomical localisation."""
import math


def evaluate_roi(g, detail):
    result = {"name": g.name, "status": "unavailable", "proposal": None,
              "reason": "Выберите полигон ROI и боковой край изображения."}
    if g.kind != "polygon" or not g.roi_edge:
        return result
    if detail["region"] not in ("hip_left", "hip_right"):
        return {**result, "reason": "Проверка отступов предназначена для бедра."}
    w = (detail["width"] - 1) * detail["pixel_mm_x"]
    h = (detail["height"] - 1) * detail["pixel_mm_y"]
    if not all(math.isfinite(v) and v > 0 for v in (w, h)):
        return {**result, "reason": "Недоступен физический масштаб."}
    x0, x1 = min(p.x for p in g.points), max(p.x for p in g.points)
    y0, y1 = min(p.y for p in g.points), max(p.y for p in g.points)
    if x0 == x1 or y0 == y1:
        return {**result, "reason": "ROI имеет нулевую ширину или высоту."}
    margins = {"top": y0 * h, "bottom": (1 - y1) * h,
               "side": (x0 if g.roi_edge == "left" else 1 - x1) * w}
    thresholds = {"top": 30., "bottom": 30., "side": 20.}
    result.update(margins_mm={k: round(v, 3) for k, v in margins.items()}, thresholds_mm=thresholds,
                  scale_source=detail["pixel_mm_source"], edge=g.roi_edge)
    if detail["pixel_mm_source"] in ("device_default", "", None):
        return {**result, "status": "estimated", "reason": "Масштаб принят по аппарату. Отступы ориентировочные; предложение недоступно."}
    if all(margins[k] >= thresholds[k] - 1e-8 for k in margins):
        return {**result, "status": "pass", "reason": "Геометрические отступы соблюдены. Анатомическая корректность ROI не проверена."}
    # Translate only. Never shrink an anatomical ROI just to make it pass.
    lx, ux = ((20 / w - x0, 1 - x1) if g.roi_edge == "left" else (-x0, 1 - 20 / w - x1))
    ly, uy = 30 / h - y0, 1 - 30 / h - y1
    if lx > ux or ly > uy:
        return {**result, "status": "fail", "reason": "Недостаточно места для переноса без изменения размера ROI."}
    dx, dy = min(max(0, lx), ux), min(max(0, ly), uy)
    proposal = g.model_dump()
    proposal["points"] = [{"x": max(0, min(1, p.x + dx)), "y": max(0, min(1, p.y + dy))} for p in g.points]
    return {**result, "status": "fail", "proposal": proposal,
            "reason": "Предложен только перенос ROI с сохранением формы. Проверьте соответствие анатомии перед принятием."}


def evaluate_geometry(geometry, detail):
    return [evaluate_roi(g, detail) for g in geometry if g.kind == "polygon"]
