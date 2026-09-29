"""Pinned research-only 2D projections of independently marked human femur meshes.

These are orthographic silhouettes, NOT DXA images or clinical QC references.
Labels are projections of 3D landmarks, not independently reviewed 2D anatomy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

REPO = 'RWTHmediTEC/VSDFullBodyBoneModels'
COMMIT = '2674d76e247060301833b077f964832ad8dc84af'
LICENSE = 'CC-BY-NC-SA-4.0'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'DXA-QC-research'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def acquire(root):
    root.mkdir(parents=True, exist_ok=True)
    tree = json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    if tree.get('sha') != COMMIT or tree.get('truncated'):
        raise ValueError('Unpinned or incomplete source tree')
    indexed = {item['path']: item for item in tree['tree'] if item['type'] == 'blob'}
    def fetch(name):
        item = indexed[name]
        path = root/name
        content = path.read_bytes() if path.exists() else get(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}')
        git_hash = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if len(content) != item['size'] or git_hash != item['sha']:
            raise ValueError('Source asset hash mismatch: '+name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(content)
        return {'path': name, 'bytes': len(content), 'git_blob_sha1': git_hash,
                'sha256': hashlib.sha256(content).hexdigest()}
    records = [fetch('README.md')]
    if 'CC_BY--NC--SA_4.0' not in (root/'README.md').read_text():
        raise ValueError('Check upstream data license')
    for rater in range(1, 6):
        records.append(fetch(f'ManualLandmarks/Femur/Rater{rater}.mat'))
    annotations = loadmat(root/'ManualLandmarks/Femur/Rater1.mat', simplify_cells=True)['Results']
    names = sorted({f'Bones/{row[4]}.mat' for row in annotations})
    # Budget is deliberately limited to the 19 annotated human subjects.
    if len(names) != 19 or sum(indexed[name]['size'] for name in names) > 150_000_000:
        raise ValueError('Unexpected subject inventory or download budget')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch, names):
            records.append(record)
            print('Verified '+record['path'], flush=True)
    manifest = {'repository': 'https://github.com/'+REPO, 'commit': COMMIT,
                'data_license': LICENSE, 'files': records,
                'total_bytes': sum(r['bytes'] for r in records),
                'scope': '19 cadaveric human subjects; 3D surface models and five raters; not clinical DXA QC'}
    (root/'acquisition.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def sphere_center(points):
    points = np.asarray(points, float)
    centered = points-points.mean(0)
    a = np.c_[2*centered, np.ones(len(points))]
    solution, _, rank, _ = np.linalg.lstsq(a, (centered*centered).sum(1), rcond=None)
    if rank != 4:
        raise ValueError('Head landmarks cannot define a sphere')
    return solution[:3]+points.mean(0)


def project(root, output):
    if output.exists():
        raise ValueError('Choose a new projection output')
    acquisition = json.loads((root/'acquisition.json').read_text())
    if acquisition.get('commit') != COMMIT or acquisition.get('data_license') != LICENSE:
        raise ValueError('Source version or license differs from pinned protocol')
    for record in acquisition['files']:
        if hashlib.sha256((root/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Changed input data')
    raters = [loadmat(root/f'ManualLandmarks/Femur/Rater{r}.mat', simplify_cells=True)['Results'] for r in range(1,6)]
    subjects = sorted({(str(row[4]), str(row[5])) for row in raters[0]})
    output.mkdir(parents=True)
    records = []
    for subject, side in subjects:
        ratings = []
        expected_names = None
        for rater in raters:
            rows = [row for row in rater if (str(row[4]), str(row[5])) == (subject, side)]
            if len(rows) != 1:
                raise ValueError('Ambiguous rater subject identity')
            for cycle in rows[0][:4]:
                names = list(cycle[:,0])
                if expected_names is not None and names != expected_names:
                    raise ValueError('Landmark ordering differs between raters')
                expected_names = names
                ratings.append(np.vstack(cycle[:,1]).astype(float))
        ratings = np.asarray(ratings)
        points = np.median(ratings,axis=0)
        bones = loadmat(root/f'Bones/{subject}.mat',simplify_cells=True)['B']
        bone = next(b for b in bones if b['name'] == f'Femur_{side}')['mesh']
        vertices = np.asarray(bone['vertices'],float)
        faces = np.asarray(bone['faces'],int)-1
        if faces.min()<0 or faces.max()>=len(vertices):
            raise ValueError('Invalid mesh vertex index')
        distances = cKDTree(vertices).query(ratings.reshape(-1,3))[0]
        if not np.isfinite(ratings).all() or np.max(distances)>5:
            raise ValueError('3D landmarks and mesh coordinate frames do not match')
        head = sphere_center(points[:6])
        neck = points[6:10].mean(0)
        if np.linalg.norm(head-neck)<5:
            raise ValueError('Degenerate independent neck axis')
        # Native CT coordinates retain superior toward increasing Z. Verify it.
        if head[2] <= np.median(vertices[:,2]):
            raise ValueError('Unexpected superior direction')
        center = head + [0,0,-45]
        proximal_faces = faces[np.all(vertices[faces,2]>=head[2]-160,axis=1)]
        for angle in (-30,-15,0,15,30,60,90):
            theta = np.radians(angle)
            rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                                 [np.sin(theta),np.cos(theta),0],[0,0,1.]])
            def raster_coords(p):
                transformed = (np.asarray(p)-center) @ rotation.T
                return transformed[..., [0,2]]*[2,-2]+[160,160]
            projected = raster_coords(vertices)
            mask = np.zeros((320,320),np.uint8)
            for face in proximal_faces:
                cv2.fillConvexPoly(mask,np.rint(projected[face]).astype(np.int32),255)
            landmarks = {'greater_trochanter': raster_coords(points[10]).tolist(),
                         'lesser_trochanter': raster_coords(points[11]).tolist(),
                         'femoral_neck_axis': raster_coords([head,neck]).tolist()}
            all_points = np.vstack([landmarks['greater_trochanter'],landmarks['lesser_trochanter'],landmarks['femoral_neck_axis']])
            visible = ((all_points>=0)&(all_points<320)).all(axis=1)
            if not visible.all():
                raise ValueError('Annotated proximal anatomy lies outside research crop')
            filename=f'{subject}_{side}_{angle:+04d}.png'
            if not cv2.imwrite(str(output/filename),mask):
                raise OSError('Cannot write silhouette')
            records.append({'subject':subject,'side':side,'split':'test' if subject in {s[0] for s in subjects[-4:]} else 'train',
                            'image':filename,'rotation_about_native_z_deg':angle,
                            'pixel_mm_x':.5,'pixel_mm_y':.5,'landmarks':landmarks,
                            'rating_count':len(ratings),'maximum_rater_to_mesh_distance_mm':float(distances.max()),
                            'landmark_rater_dispersion_median_mm':float(np.median(np.linalg.norm(ratings-points,axis=2)))})
    report={'source_commit':COMMIT,'data_license':LICENSE,'subjects':len(subjects),'cases':len(records),
            'records':records,'clinical_validation':False,'requirements_complete':False,
            'limitations':['Orthographic silhouette; no DXA attenuation or pelvic superposition.',
                           'Head center fitted to six median surface landmarks; neck center from four isthmus landmarks.',
                           '3D consensus labels projected into 2D; not independent 2D expert annotation.',
                           'Native-coordinate rotation is not an expert insufficient/excessive rotation class.']}
    (output/'dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'subjects':len(subjects),'cases':len(records),'output':str(output)}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fetch','project'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='fetch': acquire(args.source)
    elif args.output: project(args.source,args.output)
    else: parser.error('project requires --output')
