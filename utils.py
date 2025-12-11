import random
import numpy as np
import os


def set_random_seed(enable=False, seed=42):
    """
    设置随机种子以确保实验的可重复性（支持多进程）
    
    Args:
        enable (bool): 是否启用固定随机种子
        seed (int): 当 enable 为 True 时使用的随机种子（主进程）
    
    注意：
        - 在多进程环境下，每个子进程会根据进程ID和主种子生成独立的种子
        - 这样既保证了可重复性，又避免了多进程间的随机数冲突
    """
    if enable:
        # 获取当前进程ID，用于生成进程特定的种子
        pid = os.getpid()
        # 使用主种子和进程ID生成唯一的进程种子
        # 使用简单的哈希避免种子溢出（保持在32位整数范围内）
        process_seed = (seed + pid) % (2**31)
        
        random.seed(process_seed)
        np.random.seed(process_seed)
        
        # 打印当前进程的种子信息
        if pid == os.getppid() or 'MainProcess' in str(os.getpid()):
            print(f"[主进程 PID={pid}] 随机种子已设置为: {process_seed} (base_seed={seed})")
        else:
            print(f"[子进程 PID={pid}] 随机种子已设置为: {process_seed} (base_seed={seed})")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        # numpy 的随机状态重置
        np.random.seed(None)
        pid = os.getpid()
        print(f"[进程 PID={pid}] 随机种子已禁用，使用完全随机模式")
