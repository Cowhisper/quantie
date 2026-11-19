# 每日任务

你需要按照这个流程执行每日任务

1. 数据更新
   - 每天凌晨 2:00 开始
   - 使用`download_data.py`拉取所有增量的行情数据，对于历史上已拉取过的数据跳过，只拉取最新数据。如果出现下载失败，则重试至所有股票都成功更新

2. 预测和结果推送
   - 使用 `forcast.py` 预测当天开市后需要买入的股票
   - 使用企业微信推送，webhook 消息推送链接 `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=8c9ca1ca-3f20-489c-969b-e60bfdf239cf`
   - 推送消息格式：
```
# <date> 股票买入卖出操作
股票\t股票代码\t操作
xxx\tsh.xxxxx\t买入

```

# 执行方式（已实现脚本）

1. 完整流程（数据更新 → 预测 → 推送）：`daily_task.py`
   - 定时（crontab，每天凌晨 02:00）：
     ```
     0 2 * * * cd /Users/kurt/Repo/quantie && /Users/kurt/miniconda3/bin/python daily_task.py -c configs/daily.yaml >> daily.log 2>&1
     ```
     凌晨 02:00 昨日行情已完整，`daily.end` 默认为昨天；下载失败自动重试（`daily.max_retries` 轮，TASKS.md 要求重试至全部成功）。
   - 手动跑完整流程：`python daily_task.py -c configs/daily.yaml`
     （跑完自动推送；`daily.push=false` 可只跑不推；盘后补跑纳入当日行情用 `daily.end=<当天>`）

2. 手动触发推送（不更新数据、不重新预测，重发最新预测结果）：
   - `python daily_task.py -c configs/daily.yaml daily.mode=push`
   - 直接读取 `daily.output_dir`（默认 `forecast/`）中最新买入日的预测 CSV 推送。

3. 单独的预测脚本 `forecast.py` 保持不变（`python forecast.py -c configs/forecast.yaml`），
   与每日任务共用同一套扫描逻辑（`forecast.run_scan`），产出同样的 CSV。

注：当前仓库无真实持仓记录，推送消息只含策略预测的「买入」部分；无买入信号时
推送「（今日无买入信号）」。
