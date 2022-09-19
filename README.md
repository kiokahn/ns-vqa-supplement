# [NS-VQA 실행 방법]


### Author  

최초 작성 : 임소현 연구원 / AI 응용 기술 연구팀  / GazziLabs Inc.     
수정        : Kiok Ahn / GazziLabs Inc. 


## 실행 환경

Windows 10   
Docker (Python 3.6)   


## NS-VQA GitHub code 저장
- https://github.com/kexinyi/ns-vqa 에서 code 전체 저장   
```
git clone https://github.com/kexinyi/ns-vqa.git
```


## Docker image pull
- https://hub.docker.com/layers/pytorch/pytorch/0.4_cuda9_cudnn7/images/sha256-2a25af68b37a33881e6e41dfa663291e39fac784d9453dfa26be918fb57d25e5
```
docker pull pytorch/pytorch:0.4_cuda9_cudnn7
```


## Docker container run
```
docker run -itd --shm-size=8G  --mount type=bind,src=<NS-VQA GitHub code 저장 위치>,dst=<Docker container path> --privileged --gpus all —name <Docker container name> pytorch/pytorch:0.4_cuda9_cudnn7
```

ex) 
```
docker run -itd --shm-size=8G --mount type=bind,src=D:/work/VQA,dst=/workspace --privileged --gpus all --name vqa_test pytorch/pytorch:0.4_cuda9_cudnn7
```


## Docker container attach 

```
docker attach <Docker container name>
```
ex) 
```
docker attach vqa_test
```

## Python pip install
- 아래 항목 순서대로 실행

```
apt-get update
apt-get install libgl1 libglib2.0-0
pip install --upgrade pip
pip install cython matplotlib numpy==1.15 scipy pyyaml packaging pycocotools tensorboardx h5py opencv-python
```

## download data and pretrained model
- https://github.com/kexinyi/ns-vqa에서 sh download.sh 실행이 안 되므로 아래 항목 링크를 통해 다운받아야 함    
    
CLEVR_v1.0 : https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip    
CLEVR_mini :　http://nsvqa.csail.mit.edu/assets/CLEVR_mini.zip    
pretrained : http://nsvqa.csail.mit.edu/assets/pretrained.zip    
backbones : http://nsvqa.csail.mit.edu/assets/backbones.zip    

## Original
======= 이하 https://github.com/kexinyi/ns-vqa의 README.md 설명과 동일 =======   
(각각 실행할 때 필요한 파일 위치 확인하여 파일 이동 필요)   

## compile CUDA code for Mask-RCNN

```
cd {repo_root}/scene_parse/mask_rcnn/lib  # change to this directory
sh make.sh
```

##  preprocess the CLEVR questions

```
cd {repo_root}/reason
```

### clevr-train
```
python tools/preprocess_questions.py \
    --input_questions_json ../data/raw/CLEVR_v1.0/questions/CLEVR_train_questions.json \
    --output_h5_file ../data/reason/clevr_h5/clevr_train_questions.h5 \
    --output_vocab_json ../data/reason/clevr_h5/clevr_vocab.json
```
### clevr-val
```
python tools/preprocess_questions.py \
    --input_questions_json ../data/raw/CLEVR_v1.0/questions/CLEVR_val_questions.json \
    --output_h5_file ../data/reason/clevr_h5/clevr_val_questions.h5 \
    --input_vocab_json ../data/reason/clevr_h5/clevr_vocab.json
```

## object detection
```
cd {repo_root}/scene_parse/mask_rcnn
```
```
python tools/test_net.py \
    --dataset clevr_original_val \
    --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --load_ckpt ../../data/pretrained/object_detector.pt \
    --output_dir ../../data/mask_rcnn/results/clevr_val_pretrained
```






##  attribute extraction
```
cd {repo_root}/scene_parse/attr_net
```
```
python tools/process_proposals.py \
    --dataset clevr \
    --proposal_path ../../data/mask_rcnn/results/clevr_val_pretrained/detections.pkl \
    --output_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json

python tools/run_test.py \
    --run_dir ../../data/attr_net/results \
    --dataset clevr \
    --load_checkpoint_path ../../data/pretrained/attribute_net.pt \
    --clevr_val_ann_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json \
    --output_path ../../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json
```

##  reasoning
```
cd {repo_root}/reason
```
```
python tools/run_test.py \
    --run_dir ../data/reason/results \
    --load_checkpoint_path ../data/pretrained/question_parser.pt \
    --clevr_val_scene_path ../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json \
    --save_result_path ../data/reason/results/result_pretrained.json
```