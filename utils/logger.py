import logging
import os
import sys
import os.path as osp
from datetime import datetime


# def setup_logger(name, save_dir, if_train):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     if save_dir:
#         if not osp.exists(save_dir):
#             os.makedirs(save_dir)
#         if if_train:
#             fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
#         else:
#             fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#
#     return logger

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 创建一个 StreamHandler 用于输出到控制台
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if save_dir:
        # 如果 save_dir 不存在，创建它
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 根据 if_train 参数决定日志文件名
        if if_train:
            log_filename = f"train_log_{timestamp}.txt"
        else:
            log_filename = f"test_log_{timestamp}.txt"

        # 创建一个 FileHandler 用于写入日志文件
        fh = logging.FileHandler(os.path.join(save_dir, log_filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger