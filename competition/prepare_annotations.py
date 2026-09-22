"""Prepare original-geometry DXA images and empty COCO tasks for independent expert review.

No model output is converted into a reference annotation. Review sheets are private
local artifacts and not training labels until expert completion/adjudication.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
from dxaqc.dicom_io import read_any
from dxaqc.model import CRITERIA, group_of
from dxaqc.specialist_qc import OUTPUTS, digest
from experiments.cnn_quality import valid_labels
from train_specialist import split_indices

KEYPOINTS = {
 'spine':[f'{v}_{point}' for v in ('Th12','L1','L2','L3','L4','L5')
          for point in ('center','upper_left','upper_right','lower_left','lower_right')],
 'hip':['head_center','neck_medial','neck_lateral','shaft_proximal','shaft_distal',
        'greater_trochanter','lesser_trochanter','ischium'],
}


def prepare(dataset, labels_path, output):
    if output.exists(): raise ValueError('Choose a new annotation directory')
    labels=valid_labels(labels_path)
    train,val,test=split_indices(labels)
    splits={int(i):part for part,idx in zip(('train','val','test'),(train,val,test)) for i in idx}
    output.mkdir(parents=True);(output/'images').mkdir()
    tasks={region:{'images':[],'annotations':[], 'categories':[
        {'id':1,'name':region,'keypoints':KEYPOINTS[region],'skeleton':[]}]} for region in KEYPOINTS}
    records=[];seen={}
    for i,row in enumerate(labels.itertuples()):
        path=(dataset/row.first_source_path).resolve()
        if not path.is_relative_to(dataset.resolve()): raise ValueError('Image path escapes dataset')
        image=read_any(path)
        if seen.setdefault(image.pixel_sha256,row.study_key) != row.study_key:
            raise ValueError('Decoded duplicate crosses study groups')
        ident=f'image-{i:04d}';name=f'images/{ident}.png'
        if not cv2.imwrite(str(output/name),image.pixels): raise ValueError('Could not save image')
        h,w=image.pixels.shape;region=group_of(row.region)
        tasks[region]['images'].append({'id':i+1,'file_name':name,'width':w,'height':h})
        records.append({'image_id':ident,'image':name,'label':f'labels/{ident}.txt',
                        'study':hashlib.sha256(str(row.study_key).encode()).hexdigest(),
                        'split':splits[i],'region':row.region,'reviewed':False,
                        'pixel_sha256':image.pixel_sha256,'png_sha256':digest(output/name),'label_row':i})
    for region,task in tasks.items():
        (output/f'{region}-keypoints.coco.json').write_text(json.dumps(task,indent=2)+'\n')
    object_task={'images':[image for t in tasks.values() for image in t['images']], 'annotations':[],
                 'categories':[{'id':1,'name':'implant'},{'id':2,'name':'removable_artifact'}]}
    (output/'objects.coco.json').write_text(json.dumps(object_task,indent=2)+'\n')
    manifest={'schema_version':1,'labels_sha256':digest(labels_path), 'samples':records,
              'keypoints':KEYPOINTS,'ground_truth_ready':False,
              'instructions':'Annotate independently; no coordinate or quality reference is supplied as an answer.'}
    (output/'review-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fields=['image_id','reviewer_id','reviewed',*OUTPUTS,'comment']
    for reviewer in ('A','B'):
        with (output/f'reviewer-{reviewer}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for record in records:
                writer.writerow({'image_id':record['image_id'],'reviewed':'false'})
    (output/'README.txt').write_text('Локальный пакет для двух независимых экспертов. Изображения могут содержать персональные надписи.\n'
      'Исходные оценки и предсказания ИИ намеренно не показаны в бланках.\n'
      'CSV: 1 = нарушение, 0 = нет нарушения, пусто = неизвестно/неприменимо. reviewed=true только после проверки.\n'
      'В COCO отсутствуют аннотации: заполнить в инструменте разметки. Пустой список не означает норму.\n'
      'Для передачи в обучение нужны завершённые аннотации, проверка соответствия изображений и согласование расхождений.\n'
      'Разбиение по исследованиям сохранено; новый независимый набор нужно получить отдельно.\n')
    print(json.dumps({'images':len(records),'studies':int(labels.study_key.nunique()),'ground_truth_ready':False}))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=Path(__file__).parent/'labels/image_labels.csv')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.dataset,a.labels,a.output)
