---
title: Kaggle Riiid! Answer Correctness Prediction-赛后总结
date: 2021-01-09 18:57:00
categories: 
- 比赛总结
- 数据挖掘
tags:
- 机器学习
- 教育
- 知识追踪
---
最终成绩：74 / 3395 银牌  
比赛地址：https://www.kaggle.com/c/riiid-test-answer-prediction

这比赛线上测评采用 API 方式逐步给出测试数据，模拟真实场景下的在线预测，所以基本避免了穿越特征的出现。此外由于在线测评的原因，要兼顾线上特征构造和模型预测的时间复杂度和空间复杂度，一个不满足都会导致线上提交失败。

赛题提供的数据量很大且是典型的序列数据，所以前排有许多基于 transformer 的 NN 方案，这一点是我们队伍的欠缺。本方案的线下验证集划分采用 tito 开源的 [CV Strategy](https://www.kaggle.com/its7171/cv-strategy)，特征构造框架也是采用 tito 大佬开源的 [LGBM with Loop Feature Engineering](https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering)，这里就不赘述了，直接进入特征工程部分。

# 特征工程
我们将构造的特征分为静态特征和动态特征，静态特征直接由训练集统计得到，不需要随着数据的增加逐步更新。赛题的线上测试集中是不包含新题目的，所以可以认为题目处于一种比较稳定的状态，不需要迭代更新。动态特征主要与用户相关，进来一条新数据，就需要更新与之相关的状态字典。

## 静态特征
* 题目的 tags 提取出来的第一个标签 tag1
* 题目被回答次数，回答正确次数，正确率
* 题目所属 bundle_id 有多少题目
* 题目所属 bundle_id 的被回答次数，正确回答次数，正确率
* 题目所属 part 的被回答次数，正确回答次数，正确率
* 对 part 进行分桶，以 part=5 为分界划分是听力测试还是阅读测试
* 题目所属 tag1 所有题目的正确率的平均值和方差
* 对题目的 tags 做 embedding(dimension=2)，题目所有的 tag embedding 的平均值
* 题目出现次数，题目 part 出现次数，题目 tag1 出现次数在训练集中的占比
* 题目被多少个不同的人回答过

## 动态特征
动态特征需要根据新数据不断更新，为了避免线上线下特征构造方式的差异性带来的潜在问题，我们在线下也使用了迭代式的特征构造方式。主要逻辑如下面的伪代码所示，核心在于3个函数，add_user_feats 用于线下数据的特征构造和状态字典更新，线上推理的时候，一批数据的真实标签需要在下批数据中给出，所以特征构造和状态字典更新需要在函数 add_user_feats_without_update 和函数 update_user_feats 分别进行。

```
# 线下特征构造
特征状态字典 = {}

def add_user_feats(df, 特征状态字典):
    for ... in df.values:
        根据 key 查询特征状态字典，构造本样本的特征
        更新特征状态字典

add_user_feats(训练集数据, 特征状态字典)

保存特征状态字典

# 线上特征构造
加载线下保存的特征状态字典

def update_user_feats(df, 特征状态字典):
    for ... in df.values:
        更新特征状态字典

def add_user_feats_without_update(df, 特征状态字典):
    for ... in df.values:
        根据 key 查询特征状态字典，构造本样本的特征

previous_test_df = None

for test_df in 线上测试集:
    if previous_test_df is not None:
        update_user_feats(previous_test_df, 特征状态字典)

    previous_test_df = test_df.copy()
    add_user_feats_without_update(test_df, 特征状态字典)
```
动态特征选取了不同的组合维度，包括用户，用户和题目 part 组合，用户和题目 tag1 组合，用户和题目组合。时间属性能够反应很多信息，构造时间差特征能有效反应学生的做题效率，做题热情等信息。下面斜体部分的特征都是围绕时间差展开构造的，总共大概有1.7个百分位的提升。

* 用户侧特征
  * 用户答题次数，回答正确的次数，答题正确率
  * *用户距离上次，上上次，上上上次答对题目的时间差*
  * *用户距离上次，上上次，上上上次，上上上上次答题的时间差和时间差之间的差值*
  * *用户距离上次，上上次听讲座的时间差*
  * *用户距离上次，上上次，上上次答错题目的时间差*
  * *用户距离上次，上上次学习（包括做题和听讲座）的时间差*
  * **用户这次题目所属 container_id 和上个题目所属 container_id 的差值**
  * 用户之前查看题目解答的次数
  * 用户之前答题正确查看题目解答的次数
  * 用户听力测试题目和阅读测试题目答题次数，回答正确的次数，答题正确率
  * 用户听讲座的次数
  
* 用户和题目 part 组合特征
  * 用户对同类型的题目回答次数，回答正确次数，正确率
  * *用户距离上次同类型的题目回答正确和回答错误的时间差*

* 用户和题目 tag1 组合特征
  * 用户对同标签的题目回答次数，回答正确次数，正确率

* 用户和题目组合
  * **用户之前是否回答过这道题**
  * ***用户距离上次回答该问题的时间差***

### 关于 task_container_id 的强特
Kaggle 上关于 task_container_id 的讨论不少，[这则帖子](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/189465)指出了 timestamp 和 task_container_id 不连续的问题，如图1所示。  

![图1](https://pic.logicjake.xyz/kaggle_riiid_1.png)  


根据 Kaggle 工作人员在其他 discussion 里的答疑，timestamp 代表的是用户结束回答题目的时间。task_container_id 来源于这样一个场景：用户每次做题会得到一批题目，task_container_id 代表的是这批题目的编号，编号从0开始自增，所以 task_container_id 的大小反应了用户开始做题的顺序。如果用户都是做完一批题目再做下一批题目，那么 task_container_id 应该是随着 timestamp 依次增大的。图2代表的就是做完一批题目再做下一批题目的场景（方框中编号代表 task_container_id）。  

![图2](https://pic.logicjake.xyz/kaggle_riiid_2.png)  


但是在上学的时候，老师应该教过我们：遇到不会的题目先跳过，过一会再做。task_container_id 不连续的原因就在于此，以图3为例，用户做到 task_container_id 为1的题目时发现题目不会做，所以先行跳过做 task_container_id 为2的题目。task_container_id 为2的题目比较简单，做完之后再返回去攻克 task_container_id 为1的题目。这就导致了 task_container_id 不随着 timestamp 自增的现象。根据这个现象我们可以构造一个强特：当前题目所属 task_container_id 和之前一批题目的 task_container_id 的差值，差值的绝对值越大，代表回答这次题目时跳过的越多。**这个特征提升约1.5k**。

```
# 求和上批题目 task_container_id 的差值
utci[cnt] = task_container_id - u_task_container_id_dict[user_id]

# 更新最新批题目的 task_container_id 
if task_container_id != u_task_container_id_dict[user_id]:
    u_task_container_id_dict[user_id] = task_container_id

```

![图3](https://pic.logicjake.xyz/kaggle_riiid_3.png)

### 用户和题目组合特征
用户和题目的交叉特征非常有用，但受限于二者交叉之后巨大维度带来的内存问题，只构造了2个特征，且都需要特殊的数据结构减少内存占用。

[这则帖子](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/194266)发现用户会重复回答某些问题，所以可以构造特征：用户第几次回答这个问题。如果采用之前的字典方式，user_id 和 content_id 交互后 key 的数量十分巨大，十分占内存。经原贴评论区大佬指导，可以采用 bitarray 这个数据结构退而求其次记录用户之前是否做过这个题目。每个用户维护长度为14000的 bitarray，content_id 因为已经是从0开始的连续值，所以可以用下标查询的方式确定 bitarry 对应位置是否为1。**该特征提升幅度4个k。**

之前也说过时间差是个很重要的特征，所以继续构造用户距离上次回答该问题的时间差，如果时间差比较短，那么用户可能记忆比较深刻，做错的概率也较小。构造这个特征就没办法使用 bitarray 了，还是得延续之前的字典做法。之前字典爆内存的原因在于 key 过多，所以我们采用 LRU 缓存的方式，只维护指定长度的字典，一旦新加入记录时字典满了，直接删除最近最少使用的记录。python 的数据结构 OrderedDict 有个特性，他会按照 key 加入的顺序排序。所以我们每次查询一条记录，如果该记录存在，则把他从 OrderedDict 中删除，再重新添加到 OrderedDict 中去，这样 OrderedDict 的末尾保存的就是最近使用的记录。同样当字典满了，直接删除第一条记录，就可以起到删除最近最少使用的记录的作用。**该特征提升幅度2个k。**

```
class LRUCache(OrderedDict):
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
        else:
            value = -1

        return value

    def set(self, key, value):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
        else:
            if len(self.cache) == self.capacity:
                self.cache.popitem(last=False)
                self.cache[key] = value
            else:
                self.cache[key] = value
```

# 模型融合
最终模型选用 LGB 融合[开源](https://www.kaggle.com/gilfernandes/riiid-self-attention-transformer)的0.77的 transformer 模型，0.75 * LGB 预测概率 + 0.25 * NN 预测概率，融合大概能带来3k的提升。