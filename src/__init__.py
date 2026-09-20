"""Onmyoji Vision Agent —— 基于计算机视觉的 GUI 自主交互 Agent 框架。

包结构::

    src.core          基础设施：配置、路径、异常
    src.utils         通用工具：日志
    src.vision        视觉理解层：capture（采集）/ detector（检测）/ ocr（文字识别）
    src.action        输入控制层：键鼠等 GUI 操作封装
    src.state         状态管理层：界面状态建模与转移
    src.task          任务调度层：任务定义、编排与执行

本项目只做屏幕视觉理解与 GUI 自动化，不读取游戏内部数据、不修改内存、
不做网络协议分析或程序注入。
"""
