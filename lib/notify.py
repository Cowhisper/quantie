"""企业微信机器人 webhook 消息推送（TASKS.md 每日任务的推送通道）。

群机器人 webhook 只支持 json 协议：POST {"msgtype": "markdown", ...}，
这里用标准库 urllib 实现，不引入额外依赖。推送失败（网络错误 / errcode
非 0）抛异常，由调用方决定如何处理（daily 任务会打印错误并以非零退出）。

markdown 语法是企业微信的有限子集（标题/加粗/链接/字体颜色/引用），
不支持表格——多列明细（如持仓表格）用 PIL 渲染成图片后以 msgtype=image
发送（base64 + md5，2MB 以内免上传）。
"""

import base64
import hashlib
import json
import os
import urllib.request

# 默认机器人地址（来自 TASKS.md）；调用方应优先从配置传入（configs/quantie.yaml
# 的 daily.webhook），本默认值仅供未配置时兜底
DEFAULT_WEBHOOK = ("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?"
                   "key=8c9ca1ca-3f20-489c-969b-e60bfdf239cf")


def _post(payload: dict, webhook: str, timeout: float = 10.0) -> dict:
    """发送群机器人 json 消息，返回服务端响应（errcode=0 即成功）。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if result.get("errcode") != 0:
        raise RuntimeError(f"企业微信推送失败: {result}")
    return result


def send_markdown(content: str, webhook: str = DEFAULT_WEBHOOK,
                  timeout: float = 10.0) -> dict:
    """推送 markdown 消息到企业微信群机器人，返回服务端响应
    （{"errcode": 0, "errmsg": "ok"} 即成功），失败抛 RuntimeError。"""
    return _post({"msgtype": "markdown", "markdown": {"content": content}},
                 webhook, timeout)


def send_image(png_path: str, webhook: str = DEFAULT_WEBHOOK,
               timeout: float = 10.0) -> dict:
    """推送图片消息（png/jpg，企业微信限制 base64 后 2MB 以内）。"""
    with open(png_path, "rb") as f:
        raw = f.read()
    if len(raw) > 2 * 1024 * 1024:
        raise RuntimeError(f"图片过大（{len(raw) / 1024 / 1024:.1f}MB > 2MB）: {png_path}")
    payload = {
        "msgtype": "image",
        "image": {
            "base64": base64.b64encode(raw).decode("ascii"),
            "md5": hashlib.md5(raw).hexdigest(),
        },
    }
    return _post(payload, webhook, timeout)


def send_report_image(png_path: str, webhook: str = DEFAULT_WEBHOOK) -> bool:
    """发送报告图片；PIL/字体不可用或图片生成失败时返回 False（调用方
    退回纯文本 markdown），发送本身失败仍抛异常。"""
    if not png_path or not os.path.exists(png_path):
        return False
    send_image(png_path, webhook=webhook)
    return True
