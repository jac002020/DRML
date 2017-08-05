# Continuous Kendall–Tau 分支

[English](README.md)

本分支是 [master 分支]()的额外模块，其中 DRML 模型使用改进的 [Continuous Kendall–Tau](http://www.sciencedirect.com/science/article/pii/S0165168415002686) (CKT) 损失函数进行训练。**如果需要使用本分支，请先在 master 分支内完成预训练后，再在本分支内完成微调**。

在 master 分支主文件夹 `DRML` 内克隆本分支

```shell
git clone -b ckt https://github.com/Lmy0217/DRML.git CKT
cd ./CKT
```
如果需要训练好的模型和结果，请递归克隆（先安装 [Git LFS](https://git-lfs.github.com/)）

```shell
git clone -b ckt --recursive https://github.com/Lmy0217/DRML.git CKT
cd ./CKT
```

## 微调
微调步骤与 master 分支相同 （在本分支主文件夹 `CKT` 内）。

不同模型在 CUHK01 测试集上的 P-R 曲线

![](./pr.jpg)

DRML_CKT 模型增加实验时间可以获得更高的 P-R 曲线。

## 引用
如果发现本分支对你的研究有帮助，请考虑引用：
```
@article{luo2017deep,
  title={Deep Residual Metric Learning for Human Re-identification in Video Surveillance-based Affective Computing},
  author={Mingyuan Luo, Wei Huang, Peng Zhang, Jing Li, Min Wan, Huijun Ding, Guang Chen},
  journal={Affective Social Multimedia Computing (ASMMC)},
  year={2017}
}
```

## 许可证
继承 [master 分支许可证]()。