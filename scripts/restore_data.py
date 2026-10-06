"""Reconstruct the saved image split from the pinned source, without resplitting."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import pandas as pd
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_config,path_for
from src.split_dataset import read_lock,verify_lock
from src.utils import sha256,checked_path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',required=True,type=Path)
 p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/plantvillage_approved.yaml'))
 a=p.parse_args();c=load_config(a.config);root=Path(c['root']);source=a.source.resolve()
 provenance=json.loads((root/'data/metadata/plantvillage_provenance.json').read_text())
 commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 if commit!=provenance['commit']:p.error('Checkout source commit '+provenance['commit'])
 lock=read_lock(c)
 df=pd.read_csv(path_for(c,'manifest_path'),keep_default_na=False)
 lookup=df.set_index('image_id').to_dict('index')
 if set(lookup)!={x['image_id'] for x in lock['all_assignments']}:p.error('Approved manifest/lock mismatch')
 verified={}
 for image_id,row in lookup.items():
  record=provenance['images'][row['relative_path']]
  src=checked_path(source,record['source_path'])
  if not src.is_file() or sha256(src)!=row['sha256_hash'] or row['sha256_hash']!=record['source_sha256']:
   p.error('Missing or changed source: '+str(src))
  verified[image_id]=src
 restored=0
 for item in lock['all_assignments']:
  row=lookup[item['image_id']];src=verified[item['image_id']]
  curated=checked_path(path_for(c,'raw_data_dir'),row['relative_path'])
  if curated.exists() and sha256(curated)!=row['sha256_hash']:p.error('Curated file differs; refusing overwrite')
  if not curated.exists():curated.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,curated)
  dst=checked_path(root,item['processed_path'])
  if dst.exists():
   if sha256(dst)!=item['processed_sha256']:p.error('Processed file differs; refusing overwrite: '+str(dst))
   continue
  dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.restore.png')
  try:
   with Image.open(src) as im:ImageOps.exif_transpose(im).convert('RGB').save(tmp,format='PNG')
   if sha256(tmp)!=item['processed_sha256']:p.error('Reconstructed PNG differs; use pinned Pillow. Never replace the lock to hide a mismatch.')
   tmp.replace(dst);restored+=1
  finally:
   if tmp.exists():tmp.unlink()
 print(json.dumps({'restored_images':restored,'verification':verify_lock(c)},indent=2))
if __name__=='__main__':main()
