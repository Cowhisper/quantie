# 策略调优

请按照以下步骤尝试优化现有策略

## 准备工作
1. 重置实验环境，切换到主分支(branch:main)
2. 新建一个分支 `git checkout -b autostrategy/<tag>`, <tag>取名逻辑: Exp-策略名-date-timestamp
3. 如不存在新建一个文件夹`exp/`，新建实验结果文件`exp/experiments.csv`
4. 确认待优化策略，阅读`docs/` 和 `lib/strategies/`目录下对应策略的文档和实现代码
5. baseline初始化:
   - 确认在目录`exp/`下存在当前策略的回测配置`baseline.yaml`, 保证`engine.serve=false`, 且`trades_xlsx=exp/trades.xlsx`, 且同时只回测一个策略, 如果不满足上述条件直接返回问题，停止下述流程
   - 确认`.gitignore`文件，保证git 只会记录`exp/`文件夹下的`baseline.yaml`, 同时删除 `exp/`路径下的其他文件（上次实验的残留文件）
   - 执行`python quantie.py backtest -c exp/baseline.yaml`生成交易信息文件`exp/trades.xlsx`
   - 把当前 git commit hash, 主要改动说明，胜率，总收益率，是否采用改动(可以是keep, discard, or crash)，写入实验记录文件`exp/experiments.tsv`

下面进入本次优化的工作循环
## 循环优化
1. 结合SQLite3数据库文件和交易信息文件分析当前策略存在的问题，列出所有负收益交易，总结评分系统有什么问题，需注意：
   - 重点关注评分收益预期不符的交易，一个良好的策略评分应该和收益正相关
   - 优化目标是：1.提高收益率。2.提高胜率。提收益率更优先
   - 策略优化优先级：修改超参（买卖条件超参）> 修改评分公式（引入新的要求，例如股价要求，高阶指标要求等）
2. 修改对应策略代码和文档，还有对应的配置文件`exp/baseline.yaml`
3. git commit
4. 执行`python quantie.py backtest -c exp/baseline.yaml`生成交易信息文件`exp/trades.xlsx`
5. 把当前 git commit hash, 主要改动说明，胜率，总收益率，是否采用改动，写入实验记录文件`exp/experiments.tsv`
6. 如果胜率有提升则保留当前改动继续优化，如果胜率没有变化或变差，则 git reset 回本次改动前

**注意**
- 你**只可以**修改`docs/` 和 `lib/strategies/`目录下对应策略的文档和实现代码，不要改回测回测框架
- 对于`exp/experiments.tsv`中的”是否采用改动“依赖
  - keep: 采用
  - discard: 不采用
  - crash: 修改代码报错，或者结果出现明显问题
  - timeout: 如果某次改动超过 5min 未改好, 则停止修改，git commit 后这一栏填timeout
