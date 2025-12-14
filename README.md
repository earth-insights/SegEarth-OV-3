<div align="center">

<h1>SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images</h1>

<!-- <h3></h3> -->

<div>
    <strong>Adapting SAM 3 for remote sensing OVSS</strong>
</div>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://github.com/bavarianvilliager' target='_blank'>Shengqi Zhang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=Lmqy-D4AAAAJ&hl=zh-CN&oi=ao' target='_blank'>Yupeng Deng</a><sup>2</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Chinese Academy of Sciences&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://github.com/earth-insights/SegEarth-OV-3" target='_blank'>[Code]</a> • <a href="https://arxiv.org/abs/2512.08730" target='_blank'>[arXiv]</a> • <a href="https://github.com/earth-insights/SegEarth-OV-3/blob/main/demo.py" target='_blank'>[Demo]</a> •
    </h4>
</div>

</div>

<img src="resources/vis.png" width="100%"/>

> Inference results of SegEarth-OV3 on a remote sensing image exceeding 10k×10k resolution. The image originates from [OpenMapCD](https://zenodo.org/records/14028095).

<img src="https://github.com/user-attachments/assets/d17ce794-9cd8-47cc-8b2c-a9b3e4739a10" width="100%"/>

> The overall inference pipeline of SegEarth-OV3. Given an input image and a list of text prompts, we leverage SAM 3's decoupled outputs. The pipeline involves: (1) instance aggregation to consolidate sparse object predictions; (2) dual-head mask fusion to combine the fine-grained instance details with the global coverage of the semantic head; and (3) presence-guided filtering (using the presence score) to suppress false positives from absent categories. "MAX" denotes the element-wise maximum operation, and "×" denotes multiplication.

## Abstract
> *Most existing methods for training-free Open-Vocabulary Semantic Segmentation (OVSS) are based on CLIP. While these approaches have made progress, they often face challenges in precise localization or require complex pipelines to combine separate modules, especially in remote sensing scenarios where numerous dense and small targets are present. Recently, Segment Anything Model 3 (SAM 3) was proposed, unifying segmentation and recognition in a promptable framework. In this paper, we present a preliminary exploration of applying SAM 3 to the remote sensing OVSS task without any training. First, we implement a mask fusion strategy that combines the outputs from SAM 3's semantic segmentation head and the Transformer decoder (instance head). This allows us to leverage the strengths of both heads for better land coverage. Second, we utilize the presence score from the presence head to filter out categories that do not exist in the scene, reducing false positives caused by the vast vocabulary sizes and patch-level processing in geospatial scenes. We evaluate our method on extensive remote sensing datasets. Experiments show that this simple adaptation achieves promising performance, demonstrating the potential of SAM 3 for remote sensing OVSS.*

## Dependencies and Installation

You only need to focus on installing mmcv and mmsegmentation correctly; other dependencies are not strict.

## Datasets
We include the following dataset configurations in this repo: 
1) `Semantic Segmentation`: OpenEarthMap, LoveDA, iSAID, Potsdam, Vaihingen, UAVid<sup>img</sup>, UDD5, VDD
2) `Building Extraction`: WHU<sup>Aerial</sup>, WHU<sup>Sat.Ⅱ</sup>, Inria, xBD<sup>pre</sup>
4) `Road Extraction`: CHN6-CUG, DeepGlobe, Massachusetts, SpaceNet
5) `Water Extraction`: WBS-SI

Please refer to [dataset_prepare.md](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md) for dataset preparation.

## Download checkpoints of SAM 3

Download checkpoints from [HF](https://huggingface.co/facebook/sam3) or [ModelScope](https://modelscope.cn/models/facebook/sam3).

## Quick Inference
```
python demo.py
```

## Model evaluation

```
python eval.py ./configs/cfg_DATASET.py
```

## Results
<div>
<img src="https://github.com/user-attachments/assets/393b9b2f-ac81-417f-aa4a-a11f99b8fb87" width="80%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/64c581f2-70f7-4d6e-b59f-fc6484648501" width="80%"/>
</div>

## Citation

```
@article{li2025segearthov3,
  title={SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images},
  author={Li, Kaiyu and Zhang, Shengqi and Deng, Yupeng and Wang, Zhi and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.08730},
  year={2025}
}
```

## Acknowledgement
This implementation is based on [SAM 3](https://github.com/facebookresearch/sam3) and [SCLIP](https://github.com/wangf3014/SCLIP). Thanks for the awesome work.

