"""Follow-up (dynamics) analysis: is a BMD change between two visits interpretable at all?

Clinical logic (guidelines, section 4.3; ISCD): a BMD change is meaningful only if it exceeds the least
significant change (LSC = 2.77 x precision error) AND both scans were acquired comparably. A different
hip rotation or a tilted spine changes the projected bone area and therefore BMD by several percent -
as much as a year of treatment. This module automates the second condition from the images themselves.

What it does
  1. both images pass through the normal quality control (region, quality class, violations);
  2. positioning is compared measurement by measurement (axis tilt, centring, field of view, shaft
     angle, lesser-trochanter size = rotation, margins);
  3. verdict: comparable / review / not_comparable, with the list of reasons;
  4. optional: if the operator supplies BMD values, the change is compared with the LSC - and it is
     reported as significant ONLY when the scans are comparable.

Honest status: the tolerances below are engineering defaults derived from the acquisition criteria of the
task statement (5 degree axis limit, 3/2 cm margins). The organiser data contain no repeated studies of
one patient (PatientID is anonymised, no near-identical anatomy across studies), so these tolerances are
NOT clinically validated. They are configurable and must be tuned on paired scans during a pilot.
BMD is never derived from pixel intensities - only operator-supplied values are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import LATERALITY, REGION_LABEL

LSC_FACTOR = 2.77  # guidelines, section 4.3: LSC = 2.77 x CV

# measurement -> (tolerance for 'review', tolerance for 'not comparable', unit, Russian label)
TOLERANCES = {
    "spine": {
        "spine_angle_deg": (2.0, 4.0, "°", "Наклон оси позвоночника"),
        "spine_centre_offset_mm": (10.0, 20.0, "мм", "Центрирование позвоночника"),
        "image_height_mm": (15.0, 30.0, "мм", "Протяжённость зоны сканирования"),
        "spine_curve_max_mm": (4.0, 8.0, "мм", "Кривизна оси (сколиоз / укладка)"),
    },
    "hip": {
        "shaft_angle_deg": (3.0, 6.0, "°", "Угол оси диафиза (отведение/приведение)"),
        "lesser_troch_protrusion_mm": (3.0, 6.0, "мм", "Выступ малого вертела (ротация)"),
        "troch_top_margin_mm": (10.0, 20.0, "мм", "Отступ над большим вертелом"),
        "ischium_bottom_margin_mm": (10.0, 20.0, "мм", "Отступ под седалищной костью"),
    },
}
ORDER = {"pass": 0, "review": 1, "block": 2}
VERDICT = {0: "comparable", 1: "review", 2: "not_comparable"}
VERDICT_RU = {"comparable": "Исследования сопоставимы", "review": "Сопоставимость требует проверки специалистом",
              "not_comparable": "Исследования несопоставимы: динамику МПК интерпретировать нельзя"}


@dataclass
class Check:
    id: str
    label: str
    status: str  # pass | review | block
    detail: str
    baseline: float | None = None
    followup: float | None = None
    delta: float | None = None
    unit: str = ""


@dataclass
class DynamicsResult:
    verdict: str
    verdict_ru: str
    region: str
    checks: list[Check] = field(default_factory=list)
    bmd: dict | None = None
    validated: bool = False  # tolerances are engineering defaults, see module docstring

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "verdict_ru": self.verdict_ru, "anatomical_region": self.region,
                "checks": [c.__dict__ for c in self.checks], "bmd_change": self.bmd,
                "tolerances_clinically_validated": self.validated}


def _num(x) -> float | None:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def compare(baseline: dict, followup: dict, *, baseline_bmd: float | None = None, followup_bmd: float | None = None,
            lsc_percent: float | None = None, precision_cv_percent: float | None = None,
            tolerances: dict | None = None) -> DynamicsResult:
    """baseline / followup are results of Analyzer.analyze()."""
    tol = tolerances or TOLERANCES
    checks: list[Check] = []
    same_region = REGION_LABEL[baseline["region"]] == REGION_LABEL[followup["region"]]
    checks.append(Check("region", "Одна анатомическая область", "pass" if same_region else "block",
                        f"{REGION_LABEL[baseline['region']]} / {REGION_LABEL[followup['region']]}"))
    if not same_region:
        return DynamicsResult("not_comparable", VERDICT_RU["not_comparable"], REGION_LABEL[followup["region"]], checks)
    group = "spine" if baseline["region"] == "spine" else "hip"
    if group == "hip":
        same_side = baseline["region"] == followup["region"]
        checks.append(Check("side", "Одна и та же сторона", "pass" if same_side else "block",
                            f"{LATERALITY[baseline['region']]} / {LATERALITY[followup['region']]}"))
    for name, who in (("baseline", baseline), ("followup", followup)):
        bad = bool(who["quality"])
        checks.append(Check(f"quality_{name}", f"Качество исследования ({'исходное' if name == 'baseline' else 'повторное'})",
                            "review" if bad else "pass",
                            "; ".join(who["violations"]) if bad else "нарушений не выявлено"))
    for key, (soft, hard, unit, label) in tol[group].items():
        a, b = _num(baseline["features"].get(key)), _num(followup["features"].get(key))
        if a is None or b is None:
            checks.append(Check(key, label, "review", "измерение недоступно на одном из снимков", a, b, None, unit))
            continue
        d = abs(b - a)
        status = "pass" if d <= soft else "review" if d <= hard else "block"
        checks.append(Check(key, label, status, f"Δ = {d:.1f} {unit} (допуск {soft:g}, предел {hard:g})",
                            round(a, 2), round(b, 2), round(b - a, 2), unit))
    worst = max(ORDER[c.status] for c in checks)
    verdict = VERDICT[worst]
    bmd = None
    if baseline_bmd and followup_bmd and baseline_bmd > 0:
        lsc = lsc_percent if lsc_percent else (LSC_FACTOR * precision_cv_percent if precision_cv_percent else None)
        change = (followup_bmd - baseline_bmd) / baseline_bmd * 100.0
        bmd = {"baseline_bmd": baseline_bmd, "followup_bmd": followup_bmd, "change_percent": round(change, 2),
               "lsc_percent": round(lsc, 2) if lsc else None}
        if verdict != "comparable":
            bmd["interpretation"] = "Изменение МПК не интерпретируется: сопоставимость исследований не подтверждена"
        elif lsc is None:
            bmd["interpretation"] = "LSC не задан: значимость изменения оценить нельзя"
        elif abs(change) < lsc:
            bmd["interpretation"] = f"Изменение {change:+.1f} % меньше LSC {lsc:.1f} % — незначимо"
        else:
            bmd["interpretation"] = f"Изменение {change:+.1f} % превышает LSC {lsc:.1f} % — значимо"
    return DynamicsResult(verdict, VERDICT_RU[verdict], REGION_LABEL[followup["region"]], checks, bmd)
