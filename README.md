# Continuous Kendall–Tau branch

[中文](README_zh.md)

This branch is extra module which is based on [master branch](), in which DRML model use modified [Continuous Kendall–Tau](http://www.sciencedirect.com/science/article/pii/S0165168415002686) (CKT) loss function to train. **If you want to use this branch, please pre-training in master branch before fine-tuning in this branch**.

Clone this branch in master branch home folder `DRML` as

```shell
git clone -b ckt https://github.com/Lmy0217/DRML.git CKT
cd ./CKT
```
If you need trained models and results, please recursively clone as (after install [Git LFS](https://git-lfs.github.com/))

```shell
git clone -b ckt --recursive https://github.com/Lmy0217/DRML.git CKT
cd ./CKT
```

## Fine-tuning
The fine-tuning steps are the same with master branch (in this branch home folder `CKT`).

different models P-R curves on CUHK01 testset

![](./pr.jpg)

DRML could get higher P-R curve with more time.

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
Inherit from [master branch license]()。