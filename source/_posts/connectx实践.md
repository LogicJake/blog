---
title: Connect X 实践
date: 2021-10-08 20:55:47
tags: 
- 强化学习 
categories: 
- 强化学习
---
游戏目标为在你的对手之前，在游戏板上水平、垂直或对角地放置连续4个跳棋。轮到你时，把你的一个跳棋“投”到棋盘一列中，棋子会掉落在这一列最底部的空位置。

### random
随机选取尚有空位置的一列进行投放。

```python
def my_agent(obs, config):
    from random import choice
    return choice([c for c in range(config.columns) if obs.board[c] == 0])
```
### minimax
适用于零和博弈场景，每次操作即搜索选择对自己有利的情况，如果我们令甲胜的局面值为1，乙胜的局面值为-1，而和局的值为0。当轮到甲走时，甲定会选择子节点值最大的走法；而轮到乙时，乙则会选择子节点值最小的走法。所以对于中间节点的值有如下计算方法：如果该节点所对应的局面轮到甲走棋，则该节点的值是其所有子节点中值最大的一个的值。而如果该节点所对应的局面轮到乙走棋，则该节点的值是其所有子节点中值最小的一个的值，这就是minimax的搜索思想。minimax算法本质还是穷尽，解空间大的时候不适用。此时可以约束game tree的深度，另外不一定需要知道终局分数，也能对当前棋面做出大致评估。在下面的代码中，棋面的评估仅被分为输、赢、平3种状态。

![minmax反向](https://raw.githubusercontent.com/LogicJake/imghub/master/minmax反向.png)

```python
def minimax_agent(obs, config):
    from math import inf as infinity
    from random import choice
    # 电脑
    COMP = 2
    # 玩家
    HUMAN = 1
    
    columns = config.columns
    rows = config.rows
    # 因为是提前一个落子检查，所以只需要满足inarow - 1个连续
    inarow = config.inarow - 1
    size = rows * columns

    def is_win(board, player, column):
        # 找到当前列的落子位置
        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                # 停止条件
                if (
                    r < 0
                    or r >= rows
                    or c < 0
                    or c >= columns
                    or board[c + (r * columns)] != player
                ):
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # 垂直方向，向下搜
            or (count(0, 1) + count(0, -1)) >= inarow  # 水平方向，左右两边搜
            or (count(-1, -1) + count(1, 1)) >= inarow  # 主对角线方向
            or (count(-1, 1) + count(1, -1)) >= inarow  # 次对角线方向
        )
    
    def play(board, column, player):
        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])
        board[column + (row * columns)] = player
    
    def minimax(board, player, depth):
        if player == HUMAN:
            best_score = -infinity
            best_column = None
        else:
            best_score = infinity
            best_column = None
        
        # 递归终止条件1 深度达到设定值
        if depth == 0:
            return [0, None]
        
        # 递归终止条件2 一方获胜
        for column in range(columns):
            # 遍历可选列，检查是否可以获胜
            if board[column] == 0:
                if is_win(board, player, column):
                    ## 玩家获胜
                    if player == HUMAN:
                        return [1, column]
                    ## 电脑获胜
                    else:
                        return [-1, column]
            
        for column in range(columns):
            if board[column] == 0:
                next_board = board[:]
                play(next_board, column, player)
                # 向后看，计算分数
                score, _ = minimax(next_board, player % 2 + 1, depth - 1)
                
                if player == HUMAN:
                    if score > best_score:
                        best_score = score
                        best_column = column
                else:
                    if score < best_score:
                        best_score = score
                        best_column = column
                        
        return [best_score, best_column]
    
    max_depth = 4
    _, column = minimax(obs.board[:], HUMAN, max_depth)
    # 兜底策略，如果minimax没找到解，则使用random算法
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == 0])
    return column
```

[1] https://www.bilibili.co7m/video/BV1Eb411a7Vb  
[2] https://github.com/Cledersonbc/tic-tac-toe-minimax  

### negamax
negamax是minimax的改进版，在效果上没有改进，仅仅进行了视角的统一。在minimax算法中，我们和对手之间是对立的视角，所以一个是最大化，一个是最小化。但是在negamax进行了视角的统一，大家追求的都是最大化。因此就有两个改进点，第一个就是当有一方胜利是，可以不加角色区分都返回1；第二个就在于对子节点的score进行选择时，统一最大化返回。

```python
def negamax_agent(obs, config):
    from math import inf as infinity
    from random import choice
    # 电脑
    COMP = 2
    # 玩家
    HUMAN = 1
    
    columns = config.columns
    rows = config.rows
    # 因为是提前一个落子检查，所以只需要满足inarow - 1个连续
    inarow = config.inarow - 1
    size = rows * columns

    def is_win(board, player, column):
        # 找到当前列的落子位置
        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                # 停止条件
                if (
                    r < 0
                    or r >= rows
                    or c < 0
                    or c >= columns
                    or board[c + (r * columns)] != player
                ):
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # 垂直方向，向下搜
            or (count(0, 1) + count(0, -1)) >= inarow  # 水平方向，左右两边搜
            or (count(-1, -1) + count(1, 1)) >= inarow  # 主对角线方向
            or (count(-1, 1) + count(1, -1)) >= inarow  # 次对角线方向
        )
    
    def play(board, column, player):
        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])
        board[column + (row * columns)] = player
    
    def negamax(board, player, depth):
        # 递归终止条件1 深度达到设定值
        if depth == 0:
            return [0, None]
        
        # 递归终止条件2 一方获胜
        for column in range(columns):
            # 遍历可选列，检查是否可以获胜
            if board[column] == 0:
                if is_win(board, player, column):
                    return [1, column]

            
        best_score = -infinity
        best_column = None

        for column in range(columns):
            if board[column] == 0:
                next_board = board[:]
                play(next_board, column, player)
                # 向后看，计算分数
                score, _ = negamax(next_board, player % 2 + 1, depth - 1)
                score = -score
                
                if score > best_score:
                    best_score = score
                    best_column = column

        return [best_score, best_column]
    
    max_depth = 4
    _, column = negamax(obs.board[:], HUMAN, max_depth)
    # 兜底策略，如果minimax没找到解，则使用random算法
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == 0])
    return column
```