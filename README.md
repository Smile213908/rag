# 企业知识库智能问答系统(RAG)

> 基于 bge-m3 + bge-reranker + Chroma + 云端 LLM 的 RAG 系统(全 CPU,云端 LLM 生成)。
> M1 已验收(有条件通过,见 `backend/docs/05-M1验收报告.md`);M2 开发项全部完成:模型路由/多轮对话/运营看板/admin 账号/性能优化(答案缓存+快模型封顶+召回候选收敛,2026-07-31);待 PRD 延迟口径拍板后做 M2 整体验收。

## 目录结构(前后端分离)

```text
rag/
├── backend/    后端(FastAPI + RAG 引擎,Python 3.12 conda env `rag`)
│   ├── engine/       引擎层:分块/检索/精排/两层拒答/生成/反馈/知识库/会话/改写/提问日志/统计/编排(paths.py 集中路径常量)
│   ├── api/          服务层:server.py(问答 SSE/反馈/文档管理)+ tasks.py(异步任务)
│   ├── eval/         离线评估:evalset_v1.jsonl 50 条 + eval.py(检索层)+ acceptance_run.py(验收)
│   ├── tests/        端到端实测脚本(e2e_api.py)
│   ├── docs/         产品文档(00 索引 / 01 PRD / 02 架构 / 03 API / 04 路线图 / 05 M1验收报告)
│   ├── docs_data/    知识库源文档(.md/.txt/.pdf)
│   ├── docker-compose.yml  Chroma 容器编排
│   └── requirements.txt
├── frontend/   前端(React + TS + Vite + Tailwind + shadcn/ui,问答/知识库两页)
└── README.md
```

## 架构

```text
【离线】PDF → PyMuPDF 解析+清洗页眉页脚 → 结构递归分块 → bge-m3 稠密向量 → Chroma 容器(持久化)
【在线】问题 → 指代消解(多轮) → 答案缓存(命中直接回放) → bge-m3 稠密 + BM25 混合检索 Top-N(默认 50,本机已收敛 20) → bge-reranker 精排 Top-5 → 拒答第一层(阈值 0.7) → CRAG 质检(第二层,LLM 判断资料是否支撑) → 模型路由(简单问题走快模型) → Kimi(k3) 生成带引用答案
```

- **bge-m3**:Embedding(稠密向量),1024 维;**BM25**:混合检索第二路;**bge-reranker-base**:Cross-Encoder 精排
- **模型存放**:`../model/`(即 `D:\蓝卓\Agent\开发环境\rag\model\<name>\snapshots\master`),经 `MODEL_DIR` 常量加载,可用环境变量覆盖
- **Chroma**:docker 容器(localhost:8000,数据卷持久化);**云端 LLM**:Kimi k3(**只允许 temperature=1**,不传)

## 使用

```bash
PY="/c/Users/墨染/.conda/envs/rag/python.exe"
cd backend
docker compose up -d                              # 起 Chroma
$PY -m engine.pipeline                            # 首次建索引
$PY -m engine.pipeline "员工出差住宿补贴标准是多少?"  # CLI 问答
$PY -m engine.pipeline --rebuild                  # 强制重建索引
$PY -m uvicorn api.server:app --port 8080         # HTTP 服务(启动加载模型 30~60s,单 worker)
# POST /api/v1/chat/ask(SSE) · /chat/ask/sync · /chat/feedback · /documents/* · GET /health
# 可选:AUTH_TOKEN=xxx 启用 Bearer 简单认证;评估:python eval/eval.py;验收:python eval/acceptance_run.py

cd ../frontend && npm install && npm run dev      # 前端 http://localhost:3000,vite 代理 /api → 8080
```

云端 LLM 配置在 `backend/.env`(含密钥,已 gitignore)。

> **知识库源文档需本地自备**:`backend/docs_data/` 为公司内部资料,未纳入本仓库。clone 后请自行放入制度文档(.md/.txt/.pdf),再执行 `python -m engine.pipeline` 建索引;`backend/.env` 参照 `backend/.env.example` 配置。

## 环境关键坑

- conda/pip 官方源被网络拦截,**须用清华镜像**;huggingface.co 被墙,**bge 模型走 ModelScope**(已迁至 `../model/`)
- **FlagEmbedding 1.4 与 transformers 冲突**,改用 sentence-transformers 驱动 bge 模型
- Windows git-bash 下 curl 中文 body 变 GBK 致 FastAPI 400,须 `--data @UTF-8文件`;node 只有 npm.cmd,需自建 npm shim

## 验证过的能力

- 跨文档准确分流(考勤 vs 差旅);多片段综合;诚实拒答库外问题(评估集拒答率 100%);答案带引用溯源(100%);
  事实型 Top-1 正确率 95.7%(人工抽检);知识库在线增删重建;👍👎反馈与 bad case 池;
  多轮指代消解;运营看板(命中率/拒答率/👍率/P95);重复问题答案缓存命中 0ms 回放。
- 已知挂账:端到端延迟 P95 超标——模型路由已落地(k3 40~100s → kimi-for-coding 10~16s,简单问题走快模型),
  剩余为口径问题:流式 TTFT(2~3s)才是用户体感指标,PRD 延迟口径修订(TTFT vs 端到端)待拍板,随后做 M2 整体验收。
