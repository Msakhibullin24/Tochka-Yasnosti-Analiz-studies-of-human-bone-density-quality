"""Bind completed council experiments to reproducible, non-promotional reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.model import VIOLATION_LABEL

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'data/gpu-runs'
FAMILIES = (
    ('seven_source_council', 'nested-council-2026-10-15', 10, None),
    ('task_specialist_council', 'specialist-council-2026-10-15', 14, 'selected_by_inner-bags100'),
    ('paired_hip_context', 'paired-hip-context-v2-2026-10-15', 3, 'contextual_blend'),
    ('pca_gradient_boost', 'anatomy-upgrade-2026-10-15/pca-boost-quality', 1, '.'),
    ('locked_qc_specialists', 'locked-gate-specialists-v2-2026-10-15', 2, 'margin005'),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_type_bootstrap(reference, candidate):
    """Paired whole-study bootstrap; no claim of repeated-search correction."""
    before = pd.read_csv(reference).sort_values('index').reset_index(drop=True)
    after = pd.read_csv(candidate).sort_values('index').reset_index(drop=True)
    labels = pd.read_csv(ROOT/'competition/labels/image_labels.csv')
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    studies = before.study.astype(str).to_numpy()
    unique = np.unique(studies)
    codes = ('spine_coverage', 'spine_axis', 'spine_artifact',
             'hip_position_rotation', 'hip_roi_coverage')
    counts = np.zeros((2, len(unique), len(codes), 3), float)
    for model, table in enumerate((before, after)):
        types = [set(str(v).split(';')) if pd.notna(v) else set() for v in table.violation_type]
        for j, code in enumerate(codes):
            truth = pd.to_numeric(labels[code], errors='coerce').to_numpy(float)
            predicted = np.array([VIOLATION_LABEL[code] in t for t in types])
            for i, study in enumerate(unique):
                valid = (studies == study) & np.isfinite(truth)
                counts[model, i, j] = [np.sum(valid & (truth == 1) & predicted),
                                      np.sum(valid & (truth == 0) & predicted),
                                      np.sum(valid & (truth == 1) & ~predicted)]
    def f1(conf):
        tp, fp, fn = np.moveaxis(conf, -1, 0)
        return np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=(2*tp+fp+fn)>0)
    weights = np.random.default_rng(17).multinomial(len(unique), np.ones(len(unique))/len(unique), size=10000)
    samples = f1(np.einsum('bs,mskc->mbkc', weights, counts))
    delta = samples[1] - samples[0]
    observed = f1(counts.sum(axis=1))
    return {'resamples': len(weights), 'unit': 'whole study, paired',
            'macro_f1_delta': float(np.mean(observed[1]-observed[0])),
            'macro_f1_delta_ci95': np.percentile(delta.mean(axis=1), [2.5, 97.5]).tolist(),
            'fraction_resamples_macro_delta_positive': float(np.mean(delta.mean(axis=1)>0)),
            'by_criterion': {c: {'delta_f1': float(observed[1,j]-observed[0,j]),
                                 'delta_f1_ci95': np.percentile(delta[:,j], [2.5,97.5]).tolist()}
                             for j,c in enumerate(codes)},
            'corrects_repeated_search': False}


def main():
    rows, summaries = [], []
    for family, directory, expected, primary in FAMILIES:
        root = RUNS/directory
        paths = sorted(root.rglob('evaluation.json'))
        if len(paths) != expected:
            raise ValueError(f'{family}: expected {expected} complete results, found {len(paths)}')
        protocol = root/'protocol.json'
        summary = {'family': family, 'variants': len(paths), 'primary': primary,
                   'protocol_path': str(protocol.relative_to(ROOT)), 'protocol_sha256': digest(protocol)}
        summaries.append(summary)
        for path in paths:
            data = json.loads(path.read_text())
            metrics, screen = data['metrics'], data['screen']
            oof = path.parent/'candidate_oof.csv'
            if metrics['predictions_sha256'] != digest(oof):
                raise ValueError(f'OOF hash differs: {oof}')
            if metrics['labelled_images'] != 249 or metrics['studies'] != 100:
                raise ValueError('Unexpected cohort')
            variant = str(path.parent.relative_to(root))
            quality = metrics['overall_quality']
            row = {'family': family, 'variant': variant, 'primary': variant == primary,
                   'f1': quality['f1'], 'roc_auc': quality['roc_auc'],
                   'average_precision': quality['average_precision'],
                   'criterion_macro_f1': metrics['criterion_macro_f1'],
                   'tn_fp_fn_tp': quality['tn_fp_fn_tp'],
                   'paired_delta_f1': screen['paired_delta_f1'],
                   'paired_delta_f1_ci95': screen['paired_delta_f1_ci95'],
                   'internal_screen_passed': screen['passes_internal_metric_screen'],
                   'screen_failure_reasons': screen['failure_reasons'],
                   'by_criterion': metrics['by_criterion'],
                   'evaluation_path': str(path.relative_to(ROOT)),
                   'evaluation_sha256': digest(path), 'oof_sha256': digest(oof)}
            rows.append(row)
    reference = RUNS/'anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'
    # Every comparison must be against the actual deployed research incumbent.
    refhash = digest(reference)
    for _, directory, _, _ in FAMILIES:
        for path in (RUNS/directory).rglob('evaluation.json'):
            if json.loads(path.read_text())['screen']['baseline_oof_sha256'] != refhash:
                raise ValueError(f'Wrong comparison reference: {path}')
    sample = json.loads(next((RUNS/FAMILIES[0][1]).rglob('evaluation.json')).read_text())['screen']
    baseline = sample['overall_score_metrics']['baseline']
    payload = {'scope': 'Repeatedly inspected 249-image / 100-study conditional-routing cohort',
               'clinical_validation': False, 'patient_disjoint_validation_established': False,
               'global_metric_maximum_established': False, 'requirements_acceptance_established': False,
               'new_completed_variants': len(rows), 'families': summaries,
               'unique_new_prediction_files': len({r['oof_sha256'] for r in rows}),
               'reference_path': str(reference.relative_to(ROOT)), 'reference_sha256': refhash,
               'incumbent_quality': baseline,
               'incumbent_macro_f1': sample['criterion_macro_f1']['baseline'],
               'automatic_promotions': 0, 'results': rows,
               'limitations': [
                   'Outer study labels excluded from fitted heads/meta models, thresholds and within-fold selection.',
                   'Repeated experiment inspection still permits cohort-level adaptive overfitting.',
                   'Historical region and left-hip canonicalization are conditional inputs, not raw routing validation.',
                   'Diagnostic best F1 across fixed variants is post-hoc and cannot establish superiority.',
                   'Bootstrap intervals do not correct repeated model search.',
                   'Saved experimental heads require full fit, export and raw-input replay before deployment.',
                   'Failed initial paired report export and locked-gate float assertion are retained separately and not counted twice.',
               ]}
    candidate = RUNS/FAMILIES[-1][1]/'margin0/candidate_oof.csv'
    payload['locked_gate_exploratory_control_paired_types'] = paired_type_bootstrap(reference, candidate)
    dest = ROOT/'docs/competition/council_search_2026_10_15.json'
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    text = ['# Консилиум моделей и разбор ошибок — 15 октября 2026', '',
            f'Завершено дополнительных вариантов: {len(rows)} в пяти направлениях. '
            'Проверены общие и специализированные советы моделей, контекст парного бедра, '
            'бустинг на сжатых признаках и выбор специалистов при замороженном общем QC.', '',
            'Текущая поставляемая модель 1.12.0 сохраняется. Её исследовательские метрики: '
            f"F1 **{baseline['f1']:.6f}**, ROC-AUC **{baseline['roc_auc']:.6f}**, "
            f"AP **{baseline['average_precision']:.6f}**, macro-F1 типов **{payload['incumbent_macro_f1']:.6f}**.", '',
            '| Направление / заранее выбранная процедура | F1 QC | AUC QC | Macro-F1 типов |',
            '|---|---:|---:|---:|']
    for row in rows:
        if row['primary']:
            text.append(f"| {row['family']} / {row['variant']} | {row['f1']:.6f} | {row['roc_auc']:.6f} | {row['criterion_macro_f1']:.6f} |")
    best = max(rows, key=lambda r: r['f1'])
    text += ['', f"Максимальный наблюдавшийся F1 среди новых диагностических вариантов — **{best['f1']:.6f}** "
             f"({best['family']}/{best['variant']}), его AUC — **{best['roc_auc']:.6f}**. "
             'Это выбор после просмотра результатов; рост одной метрики не доказывает улучшение всей системы.', '',
             'Совет общего качества объединяет семь источников признаков. Специализированный совет '
             'обучает экспертов отдельно для качества и каждого типа: DINOv3, RAD-DINO, MedSigLIP, '
             'Stanford X-ray, локальные анатомические признаки, IQA и два набора геометрии. '
             'Для обучения арбитра использованы вложенные разбиения 5/4/3 по исследованиям: '
             'внешние ответы не используются ни для обучения экспертов, ни для арбитров. '
             'Предварительная фиксация основной процедуры второго совета сохранена '
             'в [протоколе](competition/specialist_council_protocol_2026_10_15.json).', '',
             'Усреднение специалистов улучшило артефакты позвоночника: F1 0,619→0,765, '
             'но ухудшило общий рейтинг и часть решений по бедру. Поэтому отдельно проверен '
             'выбор источника для каждого типа при неизменных quality_score, quality_threshold, '
             'quality_pred и физическом правиле оси >5°. Для замены требуется внутренний '
             'прирост F1 более 0,05 без снижения AP; контроль использует нулевой запас. '
             'Если новые специалисты не выбирают тип, сохраняется прежний тип положительного снимка.', '',
             'Контроль без запаса улучшил macro-F1 типов до 0,553241 при прежних общих F1/AUC. '
             'Это один исправленный пропуск ROI бедра (TP 4→5 из семи случаев), '
             'одна дополнительная ложная тревога об артефакте и снижение AUC укладки бедра '
             '0,795→0,732. Основной вариант с запасом 0,05 дал macro-F1 0,533461. '
             f"Парный 95% интервал изменения macro-F1 контроля: {payload['locked_gate_exploratory_control_paired_types']['macro_f1_delta_ci95']}. "
             'Таким образом, контроль остаётся исследовательским кандидатом; '
             'его прирост не является доказательством общего устойчивого улучшения.', '',
             'Проверка открытых источников также не дала нового эталона ротации DXA. '
             'В [DeepFluoro](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) есть '
             'CT, маски бедренных костей и центры головок, но среди 14 ориентиров нет малого '
             'или большого вертела и перешейка шейки. '
             '[Спецификация данных](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020/blob/master/hdf5_layouts/Readme.md) '
             'позволяет эксперимент CT→DRR для центра головки и маски, '
             'но не обучение малого вертела по истинным точкам. Угол камеры нельзя '
             'переименовать в клиническую метку недостаточной или избыточной ротации. '
             'Повторная загрузка уже использованных анатомических наборов также не создаёт '
             'новой независимой проверки. Эти направления не выдаются за выполненное улучшение.', '',
             'Открытые пункты алгоритмов по ТЗ сохраняются: точные области всего большого '
             'вертела и тела Th12, надёжное распознавание мягких тканей с обеих сторон ROI, '
             'раздельное устойчивое определение недостаточной и избыточной ротации. '
             'Слабое место размеченного набора — физическая ось: 2 найденных нарушения из 10. '
             'Наращивание числа голосующих классификаторов не исправляет само измерение оси '
             'или отсутствие истинной разметки этих анатомических областей. '
             'Эти ограничения не превращаются в новые требования к пользователю предоставить врача.', '',
             'Это та же ранее исследованная выборка: 249 изображений, 100 исследований. '
             'Разбиение по исследованиям не доказывает независимость пациентов. '
             'Направление и отражение бедра в сохранённых признаках условны на исторической маршрутизации. '
             'Вложенные разбиения защищают от прямой утечки внешних ответов, но не устраняют '
             'переобучение от многократного просмотра одной выборки. Ни максимальные возможные метрики, '
             'ни полная приёмка ТЗ здесь не заявляются.', '',
             'Исходники экспериментов, сохранённые копии исполнявшегося кода, разбиения, модели, '
             'OOF-таблицы и результаты сохранены локально. Все 30 таблиц с хешами, метриками, '
             'доверительными интервалами и причинами отказа от автоматического продвижения '
             'содержатся в [машинном отчёте](competition/council_search_2026_10_15.json). '
             'Поставка и известные ограничения ТЗ описаны в [отчёте 1.12.0](AUTOMATIC_ANATOMY_1_12_0_RU.md).', '']
    (ROOT/'docs/MODEL_COUNCIL_2026_10_15_RU.md').write_text('\n'.join(text))
    print(json.dumps({'variants': len(rows), 'report': str(dest),
                      'internal_screens_passed': sum(r['internal_screen_passed'] for r in rows)}))


if __name__ == '__main__':
    main()
