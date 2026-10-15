"""Assemble a portable handoff from verified images and a clean Git snapshot.

Dataset directories and raw medical formats are excluded from the source
snapshot. Model weights are inside the images; included source files come
from exactly the recorded Git commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): raise ValueError('Choose a new handoff directory')
    audit = json.loads(args.archive_audit.read_text())
    release = json.loads((ROOT/'docs/CURRENT_RELEASE.json').read_text())
    if audit.get('docker_load_performed') is not True or audit.get('gzip_crc_valid') is not True:
        raise ValueError('A verified archive and a successful Docker load are required')
    expected = {release['image_id'], release['research_gpu_candidate']['image_id']}
    if set(audit.get('loaded_image_ids', [])) != expected:
        raise ValueError('Archive image identities differ from the current release')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('Commit source and documentation before creating the handoff')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    archive = (ROOT/audit['archive']).resolve()
    if digest(archive) != audit['sha256']:
        raise ValueError('Archive changed after verification')
    output.mkdir(parents=True)
    try:
        os.link(archive, output/archive.name)
    except OSError:
        shutil.copy2(archive, output/archive.name)
    source_name = f'source-{commit[:12]}.tar.gz'
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '-z', commit], cwd=ROOT).decode().split('\0')
    excluded_prefixes = ('datasets/', 'data/')
    excluded_paths = ('competition/reports/label_review_sheet.png',)
    raw_suffixes = ('.dcm', '.dicom', '.nrrd', '.nii', '.nii.gz')
    included = [name for name in tracked if name and name not in excluded_paths and not name.startswith(excluded_prefixes)
                and not name.lower().endswith(raw_suffixes)]
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(output/source_name), commit, '--', *included], cwd=ROOT, check=True)
    files = ['competition/run.sh', 'competition/run_anatomy.sh', 'competition/local_run.sh',
             'docs/CURRENT_RELEASE.json', 'docs/FINAL_DEMO_RU.md', 'docs/FINAL_PRESENTATION_RU.html',
             'docs/FINAL_PRESENTATION_RU.pdf', 'docs/AUTOMATIC_ANATOMY_1_12_0_RU.md',
             'docs/competition/anatomy_upgrade_2026_10_15.json',
             'docs/competition/anatomy_upgrade_pareto_2026_10_15.svg']
    for name in files:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    shutil.copy2(args.archive_audit, output/'archive-verification.json')
    readme = f'''# Локальная поставка DXA-QC {release['version']}

Исходный код: Git {commit}; архив {source_name}.
Из снимка исходников исключены папки datasets/ и data/ и сырые медицинские форматы.
В комплекте два образа с весами, скрипты Linux/Unix, отчёты и презентация.
Архивы исходных DICOM и внешние обучающие датасеты в комплект не добавлены.

Нужен работающий Docker. GPU-варианту требуется совместимый драйвер NVIDIA
и доступная GPU. Скрипт выбирает NVIDIA Toolkit либо подключение устройств.
Во время инференса веса и зависимости из интернета не скачиваются.

Из этой папки:

```sh
sha256sum -c SHA256SUMS
docker load -i {archive.name}
./competition/local_run.sh serve 8091
```

Открыть http://127.0.0.1:8091. Для пакетной обработки:

```sh
./competition/local_run.sh batch /absolute/input.zip /absolute/new-output
```

CPU-вариант: `./competition/run.sh serve 8090`. У него отдельные базовые
модели и другая область подтверждённых результатов. Локальное рабочее
состояние сохраняется в создаваемых скриптами папках `competition/out*`.

GPU: исследовательский F1 0,619 / ROC-AUC 0,812 на ранее изученной выборке;
это не оценка закрытого теста. `requirements_complete=false` сохранён.
Неопределённые анатомические находки не объявляются положительными проверками.
Подробности — docs/AUTOMATIC_ANATOMY_1_12_0_RU.md, паспорт — docs/CURRENT_RELEASE.json.
'''
    (output/'README_RU.md').write_text(readme)
    hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file() and p.name != archive.name}
    hashes[archive.name] = audit['sha256']
    manifest = {'version': release['version'], 'source_git_commit': commit, 'images': sorted(expected),
                'files_sha256': hashes, 'medical_datasets_copied': False,
                'source_excluded_prefixes': excluded_prefixes, 'source_excluded_suffixes': raw_suffixes,
                'source_excluded_paths': excluded_paths,
                'source_files': len(included),
                'packager_code_sha256': digest(Path(__file__))}
    (output/'delivery-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    hashes['delivery-manifest.json'] = digest(output/'delivery-manifest.json')
    (output/'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in sorted(hashes.items())))
    print(json.dumps({'output': str(output), 'source_git_commit': commit, 'files': len(hashes)}))


if __name__ == '__main__': main()
