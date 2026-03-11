
# BGREX

## 理论基础
1. 普通 REX的理论依据
判断高温的构象比低温的构象能量小就交换
2. HREX理论





## Question

## Outline
关键词：Normalizing Flow (NF) , MD


### Double-Well

### Ala2

### chignolin









## Methods



## Results
1. Double-Well
2. ala
3. chignolin
4. FGFR-ploop


## Introduction





## Discussion


## Abstract









# GaMD + NF

## 前期调研
有一篇[DBMD](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c00926)利用概率贝叶斯深度神经网络模型生成高斯分布的增强势能，以减少非谐性并提高分子模拟的准确性和采样效率。

GaMD添加的V比较平缓会不会收敛比较慢，可以好好看看这一篇：
- [Asghar 等 - 2024 - Efficient rare event sampling with unsupervised normalizing flows](Asghar%20等%20-%202024%20-%20Efficient%20rare%20event%20sampling%20with%20unsupervised%20normalizing%20flows.pdf)

[GaMD-Openmm-examples](https://github.com/MiaoLab20/gamd-openmm-examples/tree/main)
[GaMD-Openmm](https://github.com/MiaoLab20/gamd-openmm)




# ideas
1. 实现生成模型生成出的<font color="#ff0000">原子坐标进行重定位</font>
生成分子构象存在的问题：
- 维度太大
- 因为是全原子做生成，如果有一个原子的位置有偏差就会使得整体的能量过大。能不能开发一种算法/模型，可以实现生成模型生成出的<font color="#ff0000">原子坐标进行重定位</font>，可以提高精度
	- 能量最小化
	- 

2. 蛋白质由20种氨基酸构成，能不能对20种氨基酸单体训练一个生成模型，能单独对20个氨基酸做构象采样（AI2BMD）

3. 构象编辑器：模型生成出的构象不合理，可以通过“构象编辑器”修饰。但修饰完之后不能改变原有的分布














