---
title: 'KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing TOP13赛后总结'
date: 2020-06-16 14:18:17
categories: 
- 比赛总结
- 数据挖掘
tags:
- 推荐系统
- 召回
- 排序
---
最终成绩：Full榜15，Half榜13  
比赛地址：https://tianchi.aliyun.com/competition/entrance/231785/  
源码：https://github.com/LogicJake/2020_KDD_Debiasing_TOP13  
# 赛题背景
该挑战着重于曝光的公平性，即推荐过去很少曝光的项目，以对抗在推荐系统中经常遇到的 Matthew 效应。特别是，在对点击数据进行训练时执行偏差减少对该任务的成功至关重要。就像现代推荐系统中记录的点击数据与实际的在线环境有差距一样，训练数据与测试数据也会有差距，主要体现在趋势和物品的受欢迎程度上。获胜的解决方案需要在历史上很少暴露的项目上表现良好。训练数据和测试数据是在许多时期收集的，其中包括大规模的销售活动。由于趋势的变化，为了进行可靠的预测，减少偏差是不可避免的。我们提供物品的多模式特性以及一些(匿名的)关键用户特性，以帮助参与者探索解决方案，以调整数据偏差，并能够很好地处理未探索过的物品。

# 建模
比赛给出的数据如下所示，××_click_× 是用户在各个阶段的点击记录，××_qtime_× 是需要预测的用户相对的最后一次点击。数据按划窗的方式逐阶段（phase）给出，各阶段的时间分布如图1所示。各阶段大概有4天的数据，相邻阶段数据时间有交叉。且单阶段中，训练集用户和测试集用户不交叉。原始数据集只给出了测试集的 qtime ，为了获取稳定的线下验证，需要我们自己构造有标签的训练集 qtime。基于题设，我们抽出每个阶段训练集用户的最后一条 click 数据作为训练集 qtime 数据，用于线下验证，形式化表示如图2所示。
```
|-- data
	|-- underexpose_train
		|-- underexpose_user_feat.csv
		|-- underexpose_item_feat.csv
		|-- underexpose_train_click-0.csv
		|-- ...
		|-- underexpose_train_click-9.csv
	|-- underexpose_test
		|-- underexpose_test_click-0
			|-- underexpose_test_qtime-0.csv
			|-- underexpose_test_click-0.csv
		|-- ...
		|-- underexpose_test_click-9
			|-- underexpose_test_qtime-9.csv
			|-- underexpose_test_click-9.csv
```
![图1 各阶段时间分布](http://pic.logicjake.xyz/time.png)
![图2 训练集 qtime 构造方式](http://pic.logicjake.xyz/data_construct.png)

问题被切分为召回和排序两个部分，总共六路召回，最后合并删除重复召回的商品。排序阶段建模为二分类问题，采用了两个深度模型和一个LGB模型。整体流程如图3所示。

![图3 整体流程图](http://pic.logicjake.xyz/structure.png)

# 召回
在召回阶段，我们采用了6路召回，通过不同策略的差异性提高整体商品的召回率。这6种方法或利用点击序列信息提高了热门商品的召回率，或采用商品的属性信息提高冷门商品的召回率，或二者兼顾。在所有召回策略中，均是分 phase 召回。

## 改进版 itemcf
原始的 itemcf 将用户点击过的商品看做一个无序的集合，但在实际应用中，应该考虑到点击次序和时间带来的影响。在计算同一序列中两个商品的相似度时，不仅需要考虑其共现次数，也需要考虑两个商品之间的次序关系和时间关系。同一点击序列中两个商品位置越远或者点击时间间隔越大，相关性应该减小。正因为存在时间和距离衰减，我们跳过了点击距离过远（大于5）或点击时间过长（大于0.000003）的商品对的相似度计算。商品对顺序和逆序的权重也不同，在点击序列A，B，C中，"BC"这样的正序权重应该大于"BA"这样的逆序权重。为了解决偏差问题，除了从序列上建立相似性关系，也可以将商品之间的文本相似度和图像相似度纳入相似性因素，引入这二者可以极大地提高冷门商品的召回率。

建立商品的相似度关系后，进入到给用户召回商品阶段，根据用户的交互商品，结合商品相似度选择 TOP100 关联商品。选取关联商品时，除了考虑和历史交互商品的相似度，还要加入时间衰减，位置距离衰减和候选商品热度。

我们总共有两版 itemcf，二者或在上述策略的部分选用上有所差异或者衰减函数有所差异。我们保存了本阶段得到的商品相似度，供后续特征工程进一步使用。

## item2vec
除了使用人工规则从序列中提取相似度，我们还可以使用序列学习模型 Word2Vec 为商品学习向量表示。将用户的商品点击序列作为句子喂到 Word2Vec 模型，然后选取和用户最近交互商品最相似的关联商品。向量学习和寻找相似全部使用 gensim 库的 Word2Vec 实现，向量维度和迭代次数对最终结果影响较大。该方法能够极好地召回冷门商品。
```
# 模型训练
model = Word2Vec(sentences, size=256, window=5, min_count=1,
                         sg=1, hs=0, seed=seed, iter=300, negative=5, workers=6)

# 寻找关联商品
interacted_items = user_item[user_id]
sim_items = model.wv.most_similar(positive=[str(x) for x in interacted_items[-2:]], topn=100)
```

## 基于商品属性的 ANN 召回
基于交互序列的 itemcf 倾向于召回热门商品，为了进一步提高冷门商品的召回，我们采用了基于商品属性信息的 ANN 召回方法。数据集本身给出了商品的文本表示和图像表示，所以可以分别针对商品的文本 embedding 和图像 embedding 进行向量召回。我们对用户历史交互商品的向量表示进行加权平均得到用户的表示向量。在相似度计算召回阶段，如果使用矩阵运算空间复杂度会非常高，如果采用双层 for 循环则时间复杂度又太高，所以我们使用 Annoy 工具。Annoy 是 Spotify 开源的一个用于近似最近邻查询的 C++/Python 工具，对内存使用进行了优化，索引可以在硬盘保存或者加载。简单来说，先将商品的向量表示建立索引，然后用用户的向量表示寻找近似的最相似商品。Annoy 稍微牺牲了相似度计算的准确度，却极大地提高了查找速度。

对比基于商品文本信息和图像信息的召回，基于文本的召回方式要胜于基于图像的召回方式，可能是因为文本能更好地描述商品，而图片信息噪声相对较大。

## 基于网络的召回
该方法源自"Bipartite network projection and personal recommendation"，代码采用自论坛开源。该方法也分为两个阶段，商品相似度计算和基于用户交互历史的商品召回。在相似度计算阶段，通过用户将两个商品连接起来。计算相似度时考虑两种因素：1)两个商品的共同被点击用户过多，则相似度减少；2)共同被点击用户的交互商品过多，相似度也要减少。基于用户交互历史的商品召回和 itemcf 类似，不再赘述。

我们设置每路都召回100个商品，但这之间肯定存在重复召回的情况，这也是为什么召回策略讲究差异性，其实就是为了减少重复召回的数量。去重之后，平均每个用户召回到440个商品。因为每个用户在 qtime 中只会点击一个商品，所以召回构造的样本正负比也会比较大，所以进一步地我们删除了召回未命中的用户样本，减少了无用负样本的数量，也提高了后续排序模型的效果。

假设训练集用户A 在阶段0的 qtime 真实点击了商品i，基于用户A的阶段0历史交互记录，我们为其召回到商品i，j，k，l，那么在召回阶段结束我们可以得到如下样本。
| user_id | phase |  query_time  | item_id | label |
| :-----: | :---: | :----------: | :-----: | :---: |
|    A    |   0   | 0.9839420823 |    i    |   1   |
|    A    |   0   | 0.9839420823 |    j    |   0   |
|    A    |   0   | 0.9839420823 |    k    |   0   |
|    A    |   0   | 0.9839420823 |    l    |   0   |

# 排序
我们把排序建模为二分类任务，分为特征工程和二分类模型预测两部分。特征工程主要围绕召回策略进行，大量使用用户历史交互商品和待预测商品的相似度特征。模型包括两个基于 LSTM 的深度模型和一个 LGB 模型，两个深度模型的预测概率作为 LGB 模型的特征。

## 特征工程
特征工程分为三大类：商品属性，用户属性，用户-商品交互属性。数据集本身给出的属性信息较少，所以特征工程主要围绕交互属性展开。商品信息给出了128维的图像向量和128维的文本向量，原始向量维度较大，所以我们分别对其使用 PCA 降维，降维后反而提高了模型效果。除此以外统计了商品被点击次数，平均点击间隔，点击用户的年龄统计和性别统计。

用户特征包括：
* user_age_level：用户所属的年龄段（原始属性）  
* user_gender：用户的性别，可以为空（原始属性）  
* user_city_level：用户所在城市的等级（原始属性）  
* 历史点击商品次数
* 历史点击时间的统计特征（min,max,std）
* 预测时间点距离用户最进一次点击的时间差
* 用户历史点击时间跨度（max-min）

交互特征主要基于之前的召回策略进行，通过保存召回阶段的商品相似度信息或向量，我们能够间接或直接得到用户对待预测商品的评分。基于商品属性的ANN召回已经得到用户和商品的向量表示，通过余弦相似度可以直接得到用户对待预测商品的评分。基于 itemcf， Bipartite network 和 item2vec 的召回得到的只是商品之间的相似度，需要和用户的历史交互商品计算间接得到用户-商品评分，采用如下方式：
* 待预测商品和用户所有历史交互商品最大相似度
* 待预测商品和用户所有历史交互商品相似度按次序加权求和
* 待预测商品和用户最近一次交互商品相似度
* 待预测商品和用户最近k次交互商品相似度之和
* 待预测商品和用户最近k次交互商品相似度平均

## 模型
两个深度模型的结构分别如图4和图5所示，LSTM 能够很好地学习用户交互序列，所以两个模型都用到了 LSTM 层。图4的深度模型除了使用 LSTM 学习序列特征，还使用了 Attention 层进一步捕捉序列关系。除了学习序列特征，其还使用了特征工程部分输出的特征，将两类特征 concat 送入两层全连接层。

![图4 深度模型1结构图](http://pic.logicjake.xyz/nn1.png)
![图5 深度模型2结构图](http://pic.logicjake.xyz/nn2.png)

图5模型与图4的主要不同点在于，将阶段 0-6 的序列输入到 Word2Vec 模型，预训练得到商品的 embedding，然后冻结模型中的 embedding layer。

特征工程得到的特征加上两个深度模型的预测概率特征输入到 LGB 模型，输出每个用户点击某商品的最终概率。


# 后处理
我们进一步对预测概率值进行后处理，减轻商品推荐的偏差问题。统计每个商品历史被点击次数，适当让点击次数较少的商品被推荐概率放大。最开始我们让概率除以商品点击次数，但这种方式较为极端，极大提高 half 指标的同时，full 指标下降严重。因为复赛要保证 full 指标在前65成绩才有效，为了稳妥起见，我们选取了另外一种后处理方式：概率除以商品点击次数的开方，提高 half 的同时平衡 full 指标的下降。

# 参考资料
[改进青禹小生baseline，phase3线上：0.2](https://tianchi.aliyun.com/forum/postDetail?postId=105787)  
[A simple itemCF Baseline, score:0.1169(phase0-2)
](https://tianchi.aliyun.com/forum/postDetail?postId=103530)  
[A Simple Recall Method based on Network-based Inference，score:0.18 (phase0-3)](https://tianchi.aliyun.com/forum/postDetail?postId=104936)  
[天池-安泰杯跨境电商智能算法大赛分享（冠军）](https://zhuanlan.zhihu.com/p/100827940)