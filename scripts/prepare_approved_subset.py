"""Restore exactly the user-approved image list from the pinned upstream source."""
import argparse,json,shutil,sys
from pathlib import Path
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import sha256,write_json,now

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 original=pd.read_csv(root/'data/metadata/image_manifest.csv',keep_default_na=False)
 review=pd.read_csv(root/'reports/proposed_subset_review.csv',keep_default_na=False)
 eligible=set(review.loc[review.proposed_action=='eligible_verified_subset','image_id'])
 selected=original[original.image_id.isin(eligible)].copy()
 if len(selected)!=2902 or not selected.source_type.eq('verified_group').all():raise ValueError('Approved selection mismatch')
 provenance=json.loads((root/'data/metadata/plantvillage_provenance.json').read_text())
 dstroot=root/'data/curated/plantvillage_verified'
 for row in selected.itertuples():
  record=provenance['images'][row.relative_path];src=a.source/record['source_path']
  if sha256(src)!=row.sha256_hash:raise ValueError('Source changed: '+str(src))
  dst=dstroot/row.relative_path;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists() and sha256(dst)!=row.sha256_hash:raise ValueError('Existing curated image differs')
  if not dst.exists():shutil.copy2(src,dst)
 selected.to_csv(root/'data/metadata/approved_manifest.csv',index=False)
 c=yaml.safe_load((root/'config/config.yaml').read_text())
 c['paths'].update(raw_data_dir='data/curated/plantvillage_verified',manifest_path='data/metadata/approved_manifest.csv',reports_dir='reports/approved_subset')
 (root/'config/plantvillage_approved.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
 write_json(root/'reports/approved_subset/curation_decision.json',dict(status='applied_with_user_approval',timestamp=now(),counts=selected.class_label.value_counts().to_dict(),images=2902,leaf_groups=int(selected.source_group.nunique()),policy='Exclude all source/duplicate components touching unverified metadata or cross-label candidates; original source unchanged.',recovery_note='Prior unsaved checkpoints lost in runtime reset. Same approved image list, seed, settings and split algorithm restored. No previous test evaluation occurred.'))
 print(selected.class_label.value_counts().to_dict())
if __name__=='__main__':main()
