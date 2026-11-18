"""企业微信机器人 webhook 消息推送（TASKS.md 每日任务的推送通道）。

群机器人 webhook 只支持 json 协议：POST {"msgtype": "markdown", ...}，
这里用标准库 urllib 实现，不引入额外依赖。推送失败（网络错误 / errcode
非 0）抛异常，由调用方决定如何处理（daily_task 会打印错误并以非零退出）。
"""

import json
import urllib.request

# 默认机器人地址（来自 TASKS.md）；调用方应优先从配置传入（configs/daily.yaml
# 的 daily.webhook），本默认值仅供未配置时兜底
DEFAULT_WEBHOOK = ("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?"
                   "key=8c9ca1ca-3f20-489c-969b-e60bfdf239cf")


def send_markdown(content: str, webhook: str = DEFAULT_WEBHOOK,
                  timeout: float = 10.0) -> dict:
    """推送 markdown 消息到企业微信群机器人，返回服务端响应
    （{"errcode": 0, "errmsg": "ok"} 即成功），失败抛 RuntimeError。"""
    payload = json.dumps(
        {"msgtype": "markdown", "markdown": {"content": content}},
        ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if result.get("errcode") != 0:
        raise RuntimeError(f"企业微信推送失败: {result}")
    return result
