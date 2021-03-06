---
title: A*算法理解及代码实现
date: 2019-12-28 12:12:07
categories: 
- 机器学习
tags:
- 算法
---
A*寻路算法要解决的问题就是在有障碍物的情况下，如何快速找到一条到达目的节点的最短路径。

把问题抽象成以下场景：在一个由M×N的方块组成的区域中，绿色代表起始点，蓝色代表无法越过的障碍物，红色代表终点。需要注意的是，我们在寻路的时候无法越过“墙角”，对照到下图就是不能走红色路线，必须走蓝色路线。这是因为在抽象场景下，移动物体是无体积所以可以直接沿着红线穿过去，但在实际情况下，比如无人车寻路，考虑到体积因素是无法行进红色路线的，所以在建模的时候需要加上这样的约束条件。当然在不涉及到墙角的情况下是可以走斜线的，这是毋庸置疑的。
<!-- more -->
<div style="margin: auto">![不能直接越过墙角](http://pic.logicjake.xyz/a_star1.jpg)</div>

# 算法流程
首先定义open list和close list，open list存放已知但还没有探索过的区块，close list存放已经探索过的区块。

最短路径肯定涉及到距离度量，在A*算法中距离分为两个部分：G 和H，总距离F=G + H。

G等于从起点移动到指定方格的移动代价。在本例中，相邻节点间，横向和纵向的移动代价为 10 ，对角线的移动代价为 14 （10×根号2的近似）。为了方便计算和寻路，我们为每个节点设置一个父节点。父节点可以这样理解，在目前已知条件下，存在一条从起点到当前指定方格的最优路径，而父亲节点就是这条路径上的指定方格的上一个节点，计算当前方格的 G 值的方法就是找出其父亲的 G 值，然后按在父亲节点直线方向还是斜线方向加上 10 或 14。

H为从当前节点到终点的估计距离，是对剩余距离的估算值，而不是实际值。它是一种理想值，忽略了障碍物的影响。在本例中使用曼哈顿距离（街区距离）来度量剩余距离。

整个算法流程为：
* 把起点加入open list，重复以下流程
  * 如果open list为空，寻路失败，找不到到达终点的路径。遍历 open list ，查找 F 值最小的节点，把它作为当前要处理的节点。
  * 把这个节点移到 close list
  * 对当前方格的 8 个相邻方格的每一个方格
    * 如果它是不可抵达的或者它在 close list 中，忽略。
    * 如果它不在 open list 中，把它加入 open list ，并且把当前方格设置为它的父亲，计算该方格的 F ， G 和 H 值。
    * 如果它已经在 open list 中，检查通过当前方格到达该方格是否代价更小，即G值更小。如果是这样，把它的父亲设置为当前方格，并重新计算它的 G 和 F 值。
* 如果终点加入到了open list中，此时路径已经找到，从终点开始，每个方格沿着父节点移动直至起点，这就是最优路径。

由算法可以看出通过总距离F选出当前处理节点，通过G来更新路径（改变节点的父节点就是改变了路径）。

另外需要注意：在寻找F值最小的时候可能会出现不止一个节点的情况，此时处于节省寻路时间的考虑，选择最后放入open list的节点。因为最后放入open list的节点是上一个处理节点的邻居节点，从而保证寻路时的连贯性，不会出现在寻路过程中突然跳到另外的地方重新开辟一条新路径。

# 流程解释
解释流程前，先说明图例：

* 绿色填充方块：起点
* 蓝色填充方块：障碍
* 红色填充方块：终点
* 绿色边的方块：open list中的方块
* 黄色边框方块：close list中的方块
* 方块中白色箭头指向父亲节点
* 方块中左上角数字代表F值，左下角G值，右下角H值*

![第1次搜索](http://pic.logicjake.xyz/a_star2.jpg)


## 开始搜索
从open list中取出起始点，将起点加入close list。起点周围8个方格都可到达所以都加入到open list中，设置父节点为起点，并计算各自的F，G，H值。结果如上图所示。

## 继续搜索
从open list中找出F值最小的方格，起点右边的方格F值为40最小，暂且称该节点为A。将A从open list剔除，加入到close list。A右边为障碍物，忽略；其余方向的方格都已经在open list中且加入A并没有减小他们的G值，所以维持原样不变。

结果如下图所示，可见起点右边的方格加上了黄色框，代表进入close list，其余不变。

![第2次搜索结果](http://pic.logicjake.xyz/a_star3.jpg)

重复以上步骤，值得注意的是在第5次搜索，随着起点正下方方格（称其为B）加入到close list，处于B下方的方格（称其为C）因为B的加入，起点到C的距离缩短到80，所以C的父节点跟新为B，并相应跟新F，G，H的值。

![第4次搜索结果](http://pic.logicjake.xyz/a_star4.jpg)
![第5次搜索结果](http://pic.logicjake.xyz/a_star5.jpg)

不断重复上述步骤，最后终点被加入到open list中，从终点开始，每个方格沿着父节点移动直至起点，就是最优路径。

![最终结果](http://pic.logicjake.xyz/a_star6.jpg)
![全部过程](http://pic.logicjake.xyz/a_star7.webp)

# 代码实现
https://github.com/LogicJake/A-star-search