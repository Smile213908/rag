"""引擎层:RAG 核心(分块/检索/精排/两层拒答/生成/编排/模型注册)。

模块职责(docs/02 §3.1):
  chunking        解析+清洗+结构分块
  retriever       混合检索 + Chroma 交互
  reranker        CrossEncoder 精排
  refuser         拒答第一层(分数阈值)+ 拒答日志
  checker         拒答第二层(CRAG 质检)
  generator       LLM 调用 + prompt 组装(含流式)
  model_registry  模型注册表(F7)
  pipeline        编排主流程(CLI: python -m engine.pipeline)
  paths           共享路径常量
"""
