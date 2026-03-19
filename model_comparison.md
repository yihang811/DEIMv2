# 模型性能对比

## 小目标检测模型性能对比表

| Model | P | R | mAP50 | mAP50:95 | Para | GFLOPs |
|:------|:-:|:--:|:-----:|:--------:|:----:|:------:|
| MASF-YOLO | 56.3 | 40.7 | 49.2 | 32.9 | 12.1 | - |
| SF-YOLO | 42.4 | 36.2 | 43.2 | 25.8 | - | 3.71 |
| VRF-DETR | - | - | 51.4 | 31.8 | 13.5 | 44.3 |
| CF-YOLO | 52.8 | 43.4 | 44.9 | 27.5 | 23.9 | 3.77 |
| DEIMv2-s (COCO预训练) | 55.2 | 61.6 | 39.3 | 23.3 | 9.67 | 25.6 |
| DEIMv2-s (直接训练) | 45.9 | 59.3 | 33.6 | 19.1 | 9.67 | 25.6 |
| **DEIMv2_x** | **48.9** | **42.2** | **44.1** | **26.8** | **50.3** | **151.6** |
| **DEIMv2_x（改进版）** | **52.6** | **46.2** | **47.5** | **29.7** | **58.7** | **228.8** |



## DEIMv2_x 详细评估指标

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.268
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.274
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.379
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.576
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.122
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.324
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.676
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.435
```

## 指标说明

- **P (Precision)**: mAP@0.5:0.95，平均精度
- **R (Recall)**: AR@0.5:0.95 (maxDets=100)，平均召回率
- **mAP50**: AP@IoU=0.50，IoU阈值为0.5时的平均精度
- **mAP50:95**: AP@IoU=0.50:0.95，IoU阈值从0.5到0.95的平均精度
- **Para**: 模型参数量 (M)
- **GFLOPs**: 计算量 (G)

## 小目标检测性能

| Model | AP@small | AR@small |
|:------|:--------:|:--------:|
| DEIMv2_x | 17.9 | 32.4 |
