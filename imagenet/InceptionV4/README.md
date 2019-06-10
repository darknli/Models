# Inception V4 结构
## 整体结构
![inception v4](imgs/model_structure.png "Inception V4 整体结构")<br>
Inception v4主要由1个stem，4个Inception A 模块，7个Inception B模块和3个Inception C模块组成。<br>
- Stem模块作为inception v4的主干，起到提取基本特征的作用。
- Inception模块主要负责按尺寸提取特征。
- Reduction模块主要负责减小尺寸、加大深度。
## Stem模块
![stem](imgs/stem.png "stem模块")
## Inception A 模块
![Inception_A](imgs/Inception_A.png "Inception A模块")
## Reduction A 模块
![Reduction_A](imgs/Reduction_A.png "Reduction A模块")
## Inception B 模块
![Inception_B](imgs/Inception_B.png "Inception B模块")
## Reduction B 模块
![Reduction_B](imgs/Reduction_B.png "Reduction B模块")
## Inception C 模块
![Inception_C](imgs/Inception_C.png "Inception C模块")
