# VIPTR: A Vision Permutable Extractor for Fast and Efficient Scene Text Recognition

| [paper](https://arxiv.org/abs/2401.10110) | [English datasets](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) |[Chinese datasets](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)| **pretrained model:** [Google driver](https://drive.google.com/drive/folders/1ARBG3GqWjpBqdELvd4I60jLeDBV-UPyt?usp=drive_link) or [Baidu Netdisk (passwd:7npu)](https://pan.baidu.com/s/1N9tSWv2RdZ9peB9w8nr9IA?pwd=7npu) |

## Getting Started

### Dependency

- This work was tested with PyTorch 1.8.0, CUDA 10.1, python 3.6.13 and Ubuntu 18.04. 
- requirements : lmdb, Pillow, torchvision, nltk, natsort, timm, mmcv

```
pip install lmdb pillow torchvision nltk natsort timm mmcv
```

### Download lmdb dataset for training and evaluation from following

#### English datasets:

- Synthetic image datasets: [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) and [SynthAdd (password:627x)](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg);

- Real image datasets: the union of trainsets IIIT5K, SVT, IC03, IC13, IC15, COCO-Text, SVTP, CUTE80; ([baidu]()|[google]())

- Validation datasets : the union of the sets IC13 (857), SVT, IIIT5k (3000), IC15 (1811), SVTP, and CUTE80 ; ([baidu]()|[google]())
- Evaluation datasets : English benchmark datasets, consist of IIIT5k (3000), SVT, IC13 (857), IC15 (1811), SVTP, and CUTE80.

#### Chinese datasets:

- Download Chinese training sets, validation sets and evaluation sets from [here](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download) .

## Run benchmark with pretrained model

1. Download pretrained model from [Google driver](https://drive.google.com/drive/folders/1ARBG3GqWjpBqdELvd4I60jLeDBV-UPyt?usp=drive_link) or [Baidu Netdisk (passwd:7npu)](https://pan.baidu.com/s/1N9tSWv2RdZ9peB9w8nr9IA?pwd=7npu) ;

2. Set models path, testsets path and characters  list ;

3. Run **test_benchmark.py** ;

   ```
   CUDA_VISIBLE_DEVICES=0 python test_benchmark.py --benchmark_all_eval --Transformation TPS19 --FeatureExtraction VIPTRv1T --SequenceModeling None --Prediction CTC --batch_max_length 25 --imgW 96 --output_channel 192
   ```

4. Run **test_chn_benchmark.py**

   ```
   CUDA_VISIBLE_DEVICES=0 python test_chn_benchmark.py --benchmark_all_eval --Transformation TPS19 --FeatureExtraction VIPTRv1T --SequenceModeling None --Prediction CTC --batch_max_length 64 --imgW 320 --output_channel 192
   ```

## Results on benchmark datasets and comparison with SOTA

![image-20240201134742483](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20240201134742483.png)

## Acknowledgements
