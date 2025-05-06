# GCA-DETR

## Install

```bash
pip install -r requirements.txt
```

## Training & Evaluation

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/gcadetr/gcadetr_r50vd_6x_coco.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/gcadetr/gcadetr_r50vd_6x_coco.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/gcadetr/gcadetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

## Train custom data

set `remap_mscoco_category: False`. This variable only works for ms-coco dataset. If you want to use `remap_mscoco_category` logic on your dataset, please modify variable [`mscoco_category2name`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/data/coco/coco_dataset.py#L154) based on your dataset.