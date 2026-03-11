# 参考的文献
1. partial replica exchange molecular dynamics (PREMD) and local replica exchange molecular dynamics (LREMD)------[局部副本交换文献](https://pubs.acs.org/doi/10.1021/jp045437y)


系统的Hamiltonian分开，仅针对涉及关注部分进行复制交换，而系统的其余部分则保持在单个温度下。<font color="#ff0000">我们想做的跟这个不同的是我们只有在低温下模拟，高温的是通过BG生成的</font>





在这种应用中，通常合理地假设较大区域对较小感兴趣区域的瞬时构象的结构较弱。

想用隐式水来跑结果发现在500K的时候蛋白会散开



用高温数据训练出来的Ploop没有限制，现用筛选的方法试试能不能走通


condition生成



成功之后的好处：
- 可以只模拟一段loop
- 有一个loop编辑器的脚本