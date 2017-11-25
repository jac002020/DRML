# Continuous Kendall–Tau branch

[中文](README_zh.md)

This branch is extra module which is based on [master branch](https://github.com/Lmy0217/DRML), in which FullScale model use modified [Continuous Kendall–Tau](http://www.sciencedirect.com/science/article/pii/S0165168415002686) (CKT) loss function to train.

Clone this branch in master branch home folder `DRML` as

```shell
git clone -b fullscale https://github.com/Lmy0217/DRML.git FullScale
cd ./FullScale
```
If you need trained models and results, please recursively clone as (after install [Git LFS](https://git-lfs.github.com/))

```shell
git clone -b fullscale --recursive https://github.com/Lmy0217/DRML.git FullScale
cd ./FullScale
```

## pre-training
* Run `cnn.lua` to create ours pre-training model `../results/cnn_0.t7` as

```shell
th cnn.lua
```
The next steps are the same (except the first step) with master branch (in this branch home folder `DRML`).

## Fine-tuning
The fine-tuning steps are the same with master branch (in this branch home folder `FullScale`).

different models P-R curves on CUHK01 testset

![](./pr.eps)

FullScale could get higher P-R curve with more time.

## Citation
If you find this branch useful in your research, please consider citing:
```
@article{luo2017deep,
  title={Deep Residual Metric Learning for Human Re-identification in Video Surveillance-based Affective Computing},
  author={Mingyuan Luo, Wei Huang, Peng Zhang, Jing Li, Min Wan, Huijun Ding, Guang Chen},
  journal={Affective Social Multimedia Computing (ASMMC)},
  year={2017}
}
```

## License
Inherit from [master branch license](https://github.com/Lmy0217/DRML/blob/master/LICENSE)。
