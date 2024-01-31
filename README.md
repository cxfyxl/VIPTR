# VIPTR: A Vision Permutable Extractor for Fast and Efficient Scene Text Recognition

| [paper](https://arxiv.org/abs/1904.01906) | [training and evaluation data](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [pretrained model (Google driver)](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0) | [pretrained model (Baidu Netdisk)(passwd:rryk)](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) |

## Getting Started

### Dependency

- This work was tested with PyTorch 1.8.0, CUDA 10.1, python 3.6.13 and Ubuntu 18.04. 
- requirements : lmdb, Pillow, torchvision, nltk, natsort, timm, mmcv

```
pip install lmdb pillow torchvision nltk natsort timm mmcv
```

### Download lmdb dataset for traininig and evaluation from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)

Training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) ;\
Validation datasets : the union of the sets IC13 (857), SVT, IIIT5k (3000), IC15 (1811), SVTP, and CUTE80 ;\
Evaluation datasets : English benchmark datasets, consist of IIIT5k (3000), SVT, IC13 (857), IC15 (1811), SVTP, and CUTE80.

### Run benchmark with pretrained model

1. Download pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) ;

2. Set models path, testsets path and characters  list ;

3. Run **test_benchmark.py** ;

   ```
   CUDA_VISIBLE_DEVICES=0 python test_benchmark.py --benchmark_all_eval --Transformation TPS19 --FeatureExtraction VIPTRv1T --SequenceModeling None --Prediction CTC --batch_max_length 25 --imgW 96 --output_channel 192
   ```

4. Run **test_chn_benchmark.py**

   ```
   CUDA_VISIBLE_DEVICES=0 python test_chn_benchmark.py --benchmark_all_eval --Transformation TPS19 --FeatureExtraction VIPTRv1T --SequenceModeling None --Prediction CTC --batch_max_length 64 --imgW 320 --output_channel 192
   ```

### Results on benchmark datasets and comparison with SOTA

