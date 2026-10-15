"""Plot exploratory QC tradeoffs from the repository audit JSON."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('docs/competition/anatomy_upgrade_2026_10_15.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/competition/anatomy_upgrade_pareto_2026_10_15.svg'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = json.loads(args.report.read_text())
    rows = [r['metrics'] for r in source['trials']]
    baseline = source['baseline']
    best = max(rows, key=lambda r: (r['overall_quality']['f1'], r['overall_quality']['roc_auc']))
    typed = max(rows, key=lambda r: r['criterion_macro_f1'])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    points = ax.scatter([r['overall_quality']['roc_auc'] for r in rows], [r['overall_quality']['f1'] for r in rows],
                        c=[r['criterion_macro_f1'] for r in rows], cmap='viridis', s=40, alpha=.8)
    fig.colorbar(points, ax=ax, label='Средний F1 типов')
    ax.scatter(baseline['roc_auc'], baseline['f1'], s=130, marker='X', color='#b34136', label='Исходное сравнение')
    q = best['overall_quality']
    ax.scatter(q['roc_auc'], q['f1'], s=170, marker='*', color='#17659b', label=f"Наибольший F1: {q['f1']:.4f}; AUC: {q['roc_auc']:.4f}")
    ax.scatter(typed['overall_quality']['roc_auc'], typed['overall_quality']['f1'], s=110, facecolors='none', edgecolors='black',
               label=f"Наибольший средний F1 типов: {typed['criterion_macro_f1']:.4f}")
    ax.set(xlabel='ROC-AUC QC', ylabel='F1 QC', title=f'Поиск QC: {len(rows)} вариантов / 249 изображений / 100 групп')
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=8)
    fig.text(.02, -.04, 'Повторно исследованная когорта; разделение по пациентам не установлено.\nПоисковые результаты; независимое превосходство и глобальный максимум не доказаны.', fontsize=8)
    fig.savefig(args.output, bbox_inches='tight')
    args.output.write_text('\n'.join(line.rstrip() for line in args.output.read_text().splitlines()) + '\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
