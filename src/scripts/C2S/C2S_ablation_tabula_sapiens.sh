#!/bin/bash


source ./C2S/bin/activate

python C2S_ablation.py --address b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad --context 1000
python C2S_ablation.py --address 40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad --context 1000
python C2S_ablation.py --address 95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad --context 1000
python C2S_ablation.py --address c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad --context 1000
python C2S_ablation.py --address da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad --context 1000


python C2S_ablation.py --address b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad --context 500
python C2S_ablation.py --address 40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad --context 500
python C2S_ablation.py --address 95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad --context 500
python C2S_ablation.py --address c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad --context 500
python C2S_ablation.py --address da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad --context 500

python C2S_ablation.py --address b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad --context 250
python C2S_ablation.py --address 40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad --context 250
python C2S_ablation.py --address 95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad --context 250
python C2S_ablation.py --address c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad --context 250
python C2S_ablation.py --address da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad --context 250


python C2S_ablation.py --address b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad --context 125
python C2S_ablation.py --address 40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad --context 125
python C2S_ablation.py --address 95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad --context 125
python C2S_ablation.py --address c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad --context 125
python C2S_ablation.py --address da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad --context 125
