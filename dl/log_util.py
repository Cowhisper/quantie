"""日志工具: 同时输出到控制台和文件"""
import logging
import os


def fmt_eta(secs):
    """秒数格式化为 1h05m / 12m30s / 45s"""
    secs = max(int(secs), 0)
    if secs >= 3600:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    if secs >= 60:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs}s"


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_file:
        log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger
