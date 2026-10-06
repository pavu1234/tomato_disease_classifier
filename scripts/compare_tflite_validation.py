"""Compare frozen Keras and both TFLite exports on validation only."""
import argparse,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_config,path_for
from src.dataset_loader import load_dataset
from src.split_dataset import verify_lock
from src.evaluate import frozen_artifact,compute_metrics
from src.reporting import save_report
from src.utils import sha256

def main():
 import tensorflow as tf
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/plantvillage_approved.yaml');a=p.parse_args();c=load_config(a.config)
 verify_lock(c,scan_test=False);model_path,_=frozen_artifact(c);model=tf.keras.models.load_model(model_path)
 ds,names,_=load_dataset(c,'val');runtime={};preds={'keras':[]};labels=[]
 prefix='mobilenetv3' if c['model']['architecture']=='MobileNetV3Small' else 'mobilenetv2'
 for mode in ['float32','dynamic_range']:
  path=path_for(c,'models_dir')/f'tomato_disease_{prefix}_{mode}.tflite'
  it=tf.lite.Interpreter(model_path=str(path),num_threads=2);it.allocate_tensors()
  runtime[mode]=(it,it.get_input_details()[0]['index'],it.get_output_details()[0]['index']);preds[mode]=[]
 for x,y in ds:
  labels.extend(y.numpy().tolist());preds['keras'].extend(model(x,training=False).numpy())
  for sample in x.numpy():
   for mode,(it,i,o) in runtime.items():
    it.set_tensor(i,sample[None,...]);it.invoke();preds[mode].append(it.get_tensor(o)[0])
 y=np.array(labels);prob={k:np.array(v) for k,v in preds.items()}
 result={'scope':'validation only; NOT independent test performance','sample_count':len(y),'class_order':names,'keras_model_sha256':sha256(model_path),'results':{}}
 for mode,arr in prob.items():
  metrics,_,_=compute_metrics(y,arr,names)
  result['results'][mode]=metrics
  if mode!='keras':
   metrics['top1_agreement_with_keras']=float(np.mean(arr.argmax(1)==prob['keras'].argmax(1)))
   metrics['changed_predictions']=int(np.sum(arr.argmax(1)!=prob['keras'].argmax(1)))
   metrics['max_absolute_probability_difference']=float(abs(arr-prob['keras']).max())
 result['deployment_recommendation']='Use float32 TFLite for closest fidelity to the selected Keras model. Dynamic-range quantization changes predictions and probabilities; it is included for review, not recommended as a drop-in equivalent.'
 save_report(path_for(c,'reports_dir'),'tflite_full_validation_comparison',result)
 print(__import__('json').dumps(result,indent=2))
if __name__=='__main__':main()
