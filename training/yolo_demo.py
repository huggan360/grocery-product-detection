"""Fine-tune a product detector on user-confirmed full-image crop boxes."""
from pathlib import Path
import argparse
import json

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class NoAugmentationTrainer(DetectionTrainer):
    """Disable the augmentation pipeline itself, including Albumentations."""

    def build_dataset(self, img_path, mode='train', batch=None):
        dataset=super().build_dataset(img_path,mode,batch)
        dataset.augment=False
        dataset.transforms=dataset.build_transforms(self.args)
        if mode=='train':
            policy={'augmentation':False,'transforms':str(dataset.transforms),
                    'box_annotation_method':'full_crop_extent_user_confirmed'}
            (self.save_dir/'augmentation_policy.json').write_text(json.dumps(policy,indent=2))
        return dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch',type=int,default=8)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/'models/yolo/yolo_demo1'
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite {output}')
    pretrained=root/'models/yolo/pretrained'
    pretrained.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(pretrained/'yolo26m.pt'))
    model.train(trainer=NoAugmentationTrainer,data=str(args.data.resolve()),
        project=str(output),name='train',exist_ok=False,
        epochs=args.epochs,batch=args.batch,imgsz=640,device=0,workers=2,
        optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.001,cos_lr=True,
        warmup_epochs=2,warmup_bias_lr=.0003,patience=6,freeze=10,
        single_cls=True,seed=42,amp=True,augment=False,
        hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=0.,scale=0.,
        shear=0.,perspective=0.,flipud=0.,fliplr=0.,mosaic=0.,mixup=0.,
        copy_paste=0.,cutmix=0.,erasing=0.,auto_augment=None,multi_scale=False)
    best=Path(model.trainer.best)
    results=YOLO(str(best)).val(data=str(args.data.resolve()),split='test',
        imgsz=640,batch=args.batch,device=0,workers=2,
        project=str(output),name='catalogue_test',plots=True)
    report={'weights':str(best),'annotation_type':'full crop extent, user confirmed',
        'augmentation':False,'notice':'Held-out catalogue crop detection metrics do not measure multi-object fridge-video performance.',
        'metrics':{k:float(v) for k,v in results.results_dict.items()}}
    (output/'evaluation_catalogue_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
