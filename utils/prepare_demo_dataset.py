"""Consolidate collected crops and the user's full-image boxes into one CSV."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image
import yaml


def prepare(manifest, source_root, output, training_output, extend=False):
    records=[json.loads(line) for line in Path(manifest).read_text().splitlines()]
    output=Path(output).resolve()
    training_output=Path(training_output).resolve()
    for folder in [output,training_output]:
        if folder.exists() and any(folder.iterdir()) and not extend:
            raise ValueError(f'Refusing to overwrite nonempty folder: {folder}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    previous={}
    if extend and (output/'annotations.csv').exists():
        with (output/'annotations.csv').open(newline='') as f:
            previous={r['id']:r for r in csv.DictReader(f)}
        if set(previous)-{r['id'] for r in records}:
            raise ValueError('The extension would remove existing annotated images.')
    class_ids={r['class_name']:r['class_id'] for r in records}
    if sorted(class_ids.values())!=list(range(len(class_ids))):
        raise ValueError('Class IDs must be unique and contiguous.')
    classes=sorted(class_ids,key=class_ids.get)
    for split in ['train','val','test']:
        for category in classes:
            (training_output/'classification'/split/category).mkdir(parents=True,exist_ok=True)
        (training_output/'detection/images'/split).mkdir(parents=True,exist_ok=True)
        (training_output/'detection/labels'/split).mkdir(parents=True,exist_ok=True)
    columns=['image','split','category','class_id','xmin','ymin','xmax','ymax','width','height',
             'group','id','source','product_id','product_name','brand','barcode',
             'quantity_raw','amount','unit','pack_count','categories_original','source_url',
             'image_url','license','attribution','review_status','single_product_verified',
             'box_annotation_method','sha256','crop_metadata']
    temporary_csv=output/'.annotations.csv.tmp'
    with temporary_csv.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for r in records:
            if r['split'] not in ('train','val','test'):
                raise ValueError(f'Unresolved split: {r["id"]}')
            source=Path(source_root)/r['image_path']
            old=previous.get(r['id'])
            if old and any(str(old[key])!=str(value) for key,value in
                [('sha256',r['sha256']),('split',r['split']),('category',r['class_name']),
                 ('group',r['split_group']),('class_id',r['class_id'])]):
                raise ValueError(f'Existing annotation changed: {r["id"]}')
            with Image.open(source) as im:
                im.verify()
            with Image.open(source) as im:
                w,h=im.size
            dest=output/'images'/(r['id']+'.jpg')
            if not dest.exists():
                temporary_image=dest.with_suffix('.jpg.tmp')
                shutil.copyfile(source,temporary_image)
                temporary_image.replace(dest)
            digest=hashlib.sha256(dest.read_bytes()).hexdigest()
            if digest!=r['sha256']:
                raise ValueError(f'Checksum changed: {source}')
            for link in [training_output/'classification'/r['split']/r['class_name']/dest.name,
                         training_output/'detection/images'/r['split']/dest.name]:
                if link.is_symlink() or link.exists():
                    if not link.is_symlink() or link.resolve()!=dest:
                        raise ValueError(f'Unexpected existing training file: {link}')
                else:
                    link.symlink_to(dest)
            coords=[float(old[k]) for k in ('xmin','ymin','xmax','ymax')] if old else [0,0,w,h]
            x1,y1,x2,y2=coords
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):
                raise ValueError(f'Invalid retained box for {r["id"]}')
            label_text=f'0 {(x1+x2)/(2*w):g} {(y1+y2)/(2*h):g} {(x2-x1)/w:g} {(y2-y1)/h:g}\n'
            label_path=training_output/'detection/labels'/r['split']/(r['id']+'.txt')
            if not label_path.exists() or list(map(float,label_path.read_text().split()))!=list(map(float,label_text.split())):
                temporary_label=label_path.with_suffix('.txt.tmp')
                temporary_label.write_text(label_text)
                temporary_label.replace(label_path)
            row={key:r.get(key,'') for key in columns}
            row.update(image='images/'+dest.name,category=r['class_name'],xmin=0,ymin=0,xmax=w,ymax=h,
                width=w,height=h,group=r['split_group'],
                box_annotation_method='full_crop_extent_user_confirmed',
                categories_original=json.dumps(r['categories_original'],ensure_ascii=False),
                crop_metadata=json.dumps(r['crop'],ensure_ascii=False))
            row.update({key:r['quantity'].get(key) for key in ['quantity_raw','amount','unit','pack_count']})
            writer.writerow(old or row)
    temporary_csv.replace(output/'annotations.csv')
    class_mapping=training_output/'classification/classes.json'
    temporary_mapping=class_mapping.with_suffix('.json.tmp')
    temporary_mapping.write_text(json.dumps(class_ids,indent=2))
    temporary_mapping.replace(class_mapping)
    detection=training_output/'detection'
    (detection/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(detection),
        train='images/train',val='images/val',test='images/test',names={0:'product'})))
    archive=output.with_suffix('.zip')
    temporary_archive=archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(temporary_archive,'w') as z:
        z.write(output/'annotations.csv','annotations.csv',compress_type=zipfile.ZIP_DEFLATED)
        for p in sorted(output/'images'/(r['id']+'.jpg') for r in records):
            z.write(p,'images/'+p.name)
    with zipfile.ZipFile(temporary_archive) as z:
        assert z.testzip() is None
    temporary_archive.replace(archive)
    summary=dict(images=len(records),classes=len(classes),splits=dict(Counter(r['split'] for r in records)),
        sources=dict(Counter(r['source'] for r in records)),complete_dataset=str(output),archive=str(archive),
        box_method='Full crop extent, as explicitly confirmed by the user. No inferred boxes or augmentation.',
        detector_classes=['product'],classifier_classes=classes)
    (training_output/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/complete_dataset'))
    p.add_argument('--training-output',type=Path,default=Path('data/demo_training'))
    p.add_argument('--extend',action='store_true',help='Append images while preserving existing CSV annotations.')
    args=p.parse_args()
    prepare(args.manifest,args.source_root,args.output,args.training_output,args.extend)
