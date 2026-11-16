import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
from utils.prepare_demo_dataset import prepare


class DemoDatasetTests(unittest.TestCase):
    def test_extension_keeps_annotations_and_portable_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            records=[]
            for i in range(3):
                path=source/f'{i}.jpg'
                Image.new('RGB',(40,80),(i*70,30,50)).save(path)
                records.append(dict(id=f'ica_{i}',class_name='milk',class_id=0,split='val' if i==1 else 'train',
                    split_group=f'group-{i}',image_path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    categories_original=['Milk'],quantity=dict(quantity_raw='1l',amount=1000,unit='ml',pack_count=1),
                    crop={},source='ica',product_name='Original product',source_url='https://example.org'))
            manifest=root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in records[:2]))
            output=root/'complete';training=root/'training'
            with contextlib.redirect_stdout(io.StringIO()):prepare(manifest,source,output,training)
            with (output/'annotations.csv').open() as f:old=list(csv.DictReader(f))
            old[0]['product_name']='Manually corrected product'
            with (output/'annotations.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=old[0].keys());writer.writeheader();writer.writerows(old)
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with contextlib.redirect_stdout(io.StringIO()):prepare(manifest,source,output,training,extend=True)
            with (output/'annotations.csv').open() as f:result=list(csv.DictReader(f))
            self.assertEqual(result[:2],old)
            self.assertEqual(len(result),3)
            self.assertEqual((result[2]['xmin'],result[2]['ymin'],result[2]['xmax'],result[2]['ymax']),('0','0','40','80'))
            self.assertFalse(any(p.is_symlink() for p in (output/'images').iterdir()))
            with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                self.assertEqual(len(z.namelist()),4)
                self.assertIsNone(z.testzip())
            with self.assertRaisesRegex(ValueError,'overwrite'):
                prepare(manifest,source,output,training)


if __name__=='__main__':unittest.main()
