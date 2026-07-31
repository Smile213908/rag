"""引擎层共享路径常量(模块化后集中管理,避免各模块各自拼相对路径)。

目录约定(2026-07-30 前后端分离:backend/frontend/ 两入口):
  PROJECT_ROOT  后端根 backend/(engine/ 的上一级)
  MODEL_DIR     模型仓库根(项目两级上级 model/ 即 rag/model,可用 MODEL_DIR 覆盖)
  CACHE_DIR     BM25 索引缓存(backend/.cache/)
  LOG_DIR       运行日志(backend/logs/,含拒答/反馈日志)
  DATA_DIR      知识库源文档目录(backend/docs_data/,可用 DATA_DIR 覆盖)
  DEVICE        计算设备(嵌入/精排模型):默认 cpu,GPU 机器设 RAG_DEVICE=cuda
"""
from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "..", "..", "model"))

CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "docs_data"))

# 计算设备:bge 嵌入/精排模型的运行设备(cpu / cuda),全链路单点收口
DEVICE = os.environ.get("RAG_DEVICE", "cpu")
