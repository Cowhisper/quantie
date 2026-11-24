# 每日任务

你需要按照这个流程执行每日任务

1. 数据更新
   - 每天凌晨 2:00 开始
   - 使用`quantie.py download`拉取所有增量的行情数据，对于历史上已拉取过的数据跳过，只拉取最新数据。如果出现下载失败，则重试至所有股票都成功更新

2. 预测和结果推送
   - 使用 `quantie.py forecast` 预测当天开市后需要买入的股票
   - 使用企业微信推送，webhook 消息推送链接 `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=8c9ca1ca-3f20-489c-969b-e60bfdf239cf`
   - 推送为两条消息：
     1. 文字（markdown）：买入清单，格式如下
     2. 图片（image）：持仓明细 + 本周期卖出 + 模拟账户概览的表格图片
        （企业微信 markdown 不支持表格，由 lib/report_img.py 渲染成 PNG
        后发送；PIL/字体不可用时退回纯文本 markdown）
   - 文字消息格式：
```
# <date> 股票买入卖出操作
股票\t股票代码\t操作
xxx\tsh.xxxxx\t买入
```
   - 图片表格列（与 engine server 模式交易明细同列）：
```
持仓明细（N 只）  /  本周期卖出（N 只）
股票代码	名称	买入日期	卖出日期	买入价格/股数	卖出价格/股数	开盘价格	收盘价格	当日收益率	持有收益率	卖出原因	买入原因
模拟账户：本金 / 总资产 / 现金 / 持仓市值 / 总交易次数 / 盈利次数 / 亏损次数 / 胜率 / 总收益额 / 总收益率
```
   （每日预测后本地模拟账户自动同步——执行昨日挂单、检查持仓卖出、按日
   盯市，详见 lib/paper_account.py）

# 执行方式（已实现脚本，统一入口 quantie.py，配置 configs/quantie.yaml）

1. 完整流程（数据更新 → 预测 → 推送）：`python quantie.py daily`
   - 定时（crontab，每天凌晨 02:00）：
     ```
     0 2 * * * cd /Users/kurt/Repo/quantie && /Users/kurt/miniconda3/bin/python quantie.py daily -c configs/quantie.yaml >> daily.log 2>&1
     ```
     凌晨 02:00 昨日行情已完整，`daily.end` 默认为昨天；下载失败自动重试（`daily.max_retries` 轮，TASKS.md 要求重试至全部成功）。
   - 手动跑完整流程：`python quantie.py daily`
     （跑完自动推送；`daily.push=false` 可只跑不推；盘后补跑纳入当日行情用 `daily.end=<当天>`）

2. 手动触发推送（不更新数据、不重新预测，重发最新预测结果）：
   - `python quantie.py daily daily.mode=push`
   - 直接读取 `daily.output_dir`（默认 `forecast/`）中最新买入日的预测 CSV 推送。

3. 单独的预测命令 `python quantie.py forecast`（配置 `forecast:` 段），
   与每日任务共用同一套扫描逻辑（`lib/forecast.run_scan`），产出同样的 CSV。

注：卖出记录与账户概览来自本地模拟账户（`lib/paper_account.py`，本金 10 万，
`paper_account.db`），按真实行情撮合昨日预测挂单、由策略信号触发卖出；无买入
信号时操作表推送「（今日无买入信号）」。买入日为周末（周六/周日）休市时
自动跳过推送，周末的定时任务只更新数据、不发消息。
