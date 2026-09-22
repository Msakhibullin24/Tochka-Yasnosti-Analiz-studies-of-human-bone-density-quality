"""Build an offline HTML/JSON/CSV evaluation report from immutable QC experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from dxaqc.specialist_qc import OUTPUTS, digest
from dxaqc.model import group_of
from experiments.cnn_quality import valid_labels
from train_specialist import targets_for

LABELS = dict(zip(OUTPUTS, ['Нарушение качества', 'Охват позвоночника', 'Ось позвоночника',
                          'Артефакт позвоночника', 'Укладка бедра', 'ROI бедра']))


def _pyplot():
    """Load the optional report renderer only when a chart is actually built.

    Benchmark selection and prediction checks import ``load_run`` but do not
    need the heavyweight plotting stack. Keeping that boundary lazy lets the
    offline inference image run those checks without bundling matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Metric charts require the optional matplotlib dependency') from exc
    return plt


def measures(y, scores, threshold):
    result = {'n': len(y), 'positives': int(y.sum()), 'negatives': int(len(y)-y.sum()),
              'roc_auc': None, 'average_precision': None, 'threshold': threshold,
              'accuracy': None, 'precision': None, 'recall': None, 'specificity': None,
              'f1': None, 'balanced_accuracy': None, 'confusion_tn_fp_fn_tp': None}
    if len(np.unique(y)) == 2:
        result.update(roc_auc=float(roc_auc_score(y, scores)),
                      average_precision=float(average_precision_score(y, scores)))
    if threshold is not None and len(y):
        pred = scores >= threshold
        tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
        fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
        recall = tp/(tp+fn) if tp+fn else None
        spec = tn/(tn+fp) if tn+fp else None
        result.update(accuracy=(tn+tp)/len(y), precision=tp/(tp+fp) if tp+fp else None,
                      recall=recall, specificity=spec, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                      balanced_accuracy=(recall+spec)/2 if recall is not None and spec is not None else None,
                      confusion_tn_fp_fn_tp=[tn, fp, fn, tp])
    return result


def intervals(y, scores, threshold, groups, resamples):
    """Study-cluster percentile bootstrap; undefined resamples are excluded per metric."""
    rng = np.random.default_rng(17)
    units = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    keys = ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
    values = {k: [] for k in keys}
    if len(units) < 2:
        return {}
    for _ in range(resamples):
        idx = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        m = measures(y[idx], scores[idx], threshold)
        for key in keys:
            if m[key] is not None:
                values[key].append(m[key])
    return {k: {'low': float(np.percentile(v, 2.5)), 'high': float(np.percentile(v, 97.5)),
                'defined_resamples': len(v)} for k, v in values.items() if v}


def load_run(directory, labels_path):
    meta = json.loads((directory/'model.json').read_text())
    if meta['labels_sha256'] != digest(labels_path):
        raise ValueError('Label checksum differs from training; cannot pair predictions')
    labels = valid_labels(labels_path)
    rows = json.loads((directory/'predictions.json').read_text())
    if len(rows) != len(labels) or {r['label_row'] for r in rows} != set(range(len(labels))):
        raise ValueError('Prediction rows must cover the exact training label table')
    rows.sort(key=lambda r: r['label_row'])
    splits = np.array([r['split'] for r in rows])
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Missing or invalid partitions')
    for r, label in zip(rows, labels.itertuples()):
        if r['study_hash'] != hashlib.sha256(str(label.study_key).encode()).hexdigest():
            raise ValueError('Study pairing mismatch')
    groups = np.array([r['study_hash'] for r in rows])
    if any(len(set(splits[groups == g])) != 1 for g in set(groups)):
        raise ValueError('Study leakage in saved predictions')
    scores = np.array([[r[name+'_score'] for name in OUTPUTS] for r in rows])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid prediction probabilities')
    targets = targets_for(labels)
    if meta.get('region_scope','all') != 'all':
        targets[np.array([group_of(r) != meta['region_scope'] for r in labels.region])] = np.nan
    return meta, targets, scores, splits, groups


def build(runs, labels_path, output, resamples=1000):
    if output.exists():
        raise ValueError('Choose a new report directory')
    plt = _pyplot()
    loaded = [load_run(path, labels_path) for path in runs]
    output.mkdir(parents=True)
    summary = {'scope': 'internal study-held-out; no external/patient-level validation',
               'positive_class': 'quality violation', 'bootstrap_resamples': resamples,
               'yolo26x': {'status': 'awaiting_target_annotations', 'metrics': None,
                           'reason': 'No DXA implant/artifact boxes or masks supplied to this QC report'},
               'runs': []}
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>Метрики обучения DXA QC</title>',
            '<body><main><h1>Метрики обучения DXA QC</h1>',
            '<p>Положительный класс — нарушение качества. Режим обучения и область указаны отдельно для каждого запуска. '
            'Порог выбран только на validation; test не участвовал в обучении и выборе порога. '
            'Это внутреннее разделение по исследованиям, а не внешняя клиническая проверка.</p>',
            '<p>YOLO26x: ожидается целевая bbox/mask-разметка. Обучение детектора и его mAP не выполнены. '
            'Метрики ниже относятся только к QC-классификаторам.</p>',
            '<p><a href="metrics.json">Все метрики JSON</a> · <a href="metrics.csv">Таблица CSV</a></p>']
    csv_rows = []
    for index, (path, (meta, target, scores, splits, groups)) in enumerate(zip(runs, loaded)):
        run = {'name': path.name, 'backbone': meta['backbone'], 'loss': meta['loss'],
               'epochs': meta['epochs'], 'split': meta['split'], 'model_sha256': meta['model_sha256'],
               'metrics': {}}
        body.append(f'<section><h2>{html.escape(path.name)}</h2><p>Эпох: {meta["epochs"]}. '
                    f'Область: {html.escape(meta.get("region_scope","all"))}; режим: {html.escape(meta.get("train_mode","head"))}; выбранная эпоха: {meta.get("selected_epoch",meta["epochs"])}. '
                    f'Разделение: {html.escape(json.dumps(meta["split"]))}.</p>')
        history_path = path/'history.json'
        if history_path.exists():
            history = json.loads(history_path.read_text())
            fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
            for key in ('train_loss', 'validation_loss'):
                ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
            ax.set(xlabel='Epoch', ylabel=meta['loss'].upper(), title='Loss (not comparable between BCE and ASL)')
            ax.legend(); name = f'run-{index}-loss.png'; fig.savefig(output/name, dpi=130); plt.close(fig)
            body.append(f'<p><img src="{name}" alt="Loss по эпохам: train и validation" style="max-width:100%"></p>')
        body.append('<table border="1" cellpadding="6"><caption>Test; — означает, что метрика не определена</caption>'
                    '<thead><tr><th scope="col">Задача</th><th scope="col">N / нарушений</th>'
                    '<th scope="col">ROC AUC</th><th scope="col">AP</th><th scope="col">Precision</th>'
                    '<th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Specificity</th></tr></thead><tbody>')
        charts = []
        for j, name in enumerate(OUTPUTS):
            threshold = meta['thresholds'][name]
            by_split = {}
            for split in ('train', 'validation', 'test'):
                idx = (splits == split) & np.isfinite(target[:, j])
                m = measures(target[idx, j], scores[idx, j], threshold)
                if split == 'test':
                    m['ci95'] = intervals(target[idx,j], scores[idx,j], threshold, groups[idx], resamples)
                by_split[split] = m
                csv_rows.append({'run': path.name, 'task': name, 'split': split,
                                 **{k: v for k, v in m.items() if k not in ('ci95', 'confusion_tn_fp_fn_tp')}})
            run['metrics'][name] = by_split
            m = by_split['test']
            cells = ''.join(f'<td>{m[k]:.3f}</td>' if m[k] is not None else '<td>—</td>'
                            for k in ('roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'specificity'))
            body.append(f'<tr><th scope="row">{LABELS[name]}</th><td>{m["n"]} / {m["positives"]}</td>{cells}</tr>')
            idx = (splits == 'test') & np.isfinite(target[:, j])
            y, s = target[idx,j], scores[idx,j]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout='constrained')
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y, s); axes[0].plot(fpr, tpr); axes[0].plot([0,1],[0,1], '--')
                precision, recall, _ = precision_recall_curve(y, s); axes[1].plot(recall, precision)
                axes[1].axhline(y.mean(), linestyle='--')
            else:
                for ax in axes[:2]: ax.text(.1,.5,'Undefined: missing class')
            axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC', xlim=(0,1), ylim=(0,1))
            axes[1].set(xlabel='Recall', ylabel='Precision', title='PR', xlim=(0,1), ylim=(0,1))
            cm = m['confusion_tn_fp_fn_tp']
            if cm is not None:
                a = np.array(cm).reshape(2,2); axes[2].imshow(a, cmap='Blues')
                for row in range(2):
                    for col in range(2): axes[2].text(col, row, str(a[row,col]), ha='center', va='center')
                axes[2].set(xticks=[0,1], yticks=[0,1], xlabel='Predicted', ylabel='True', title='Confusion')
            else:
                axes[2].text(.05,.5,'No calibrated threshold'); axes[2].set_axis_off()
            filename = f'run-{index}-{name}.png'; fig.savefig(output/filename, dpi=130); plt.close(fig)
            charts.append(f'<details><summary>{LABELS[name]}: ROC, PR, матрица ошибок</summary>'
                          f'<img src="{filename}" alt="Графики test для {LABELS[name]}" style="max-width:100%"></details>')
        body.append('</tbody></table><p>AP — average precision. 95% интервалы в JSON: bootstrap по исследованиям; '
                    'при малом числе примеров оценки нестабильны. Метрики train оптимистичны.</p>'+''.join(charts)+'</section>')
        summary['runs'].append(run)
    body.append('</main></body></html>')
    (output/'index.html').write_text('\n'.join(body))
    (output/'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', type=Path, default=Path(__file__).parent/'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100: parser.error('--bootstrap must be at least 100')
    build(args.runs, args.labels, args.output, args.bootstrap)
