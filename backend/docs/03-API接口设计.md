# API 接口设计:企业知识库智能问答系统

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.6 |
| 创建日期 | 2026-07-22 |
| 上游文档 | [[01-PRD-产品需求文档]] / [[02-架构设计文档]] |
| 实现约定 | 后端 FastAPI,Python 3.12,conda env `rag`;REST + SSE;JSON 交换 |

---

## 1. 通用约定

### 1.1 Base URL
```
开发:  http://localhost:8080/api/v1
```

### 1.2 认证
- 除健康检查外,接口需携带 `Authorization: Bearer <token>`(JWT,一期可为简单 token)。
- 管理类接口(`/documents`、`/admin/*`)需管理员角色。

### 1.3 统一响应结构
```json
{ "code": 0, "message": "ok", "data": { } }
```
错误时 `code != 0`,`message` 为可读原因。统一错误码:

| code | 含义 |
|------|------|
| 0 | 成功 |
| 400 | 参数错误(空问题/超长/格式错误) |
| 401 | 未认证 / token 失效 |
| 403 | 无权限(越权访问文档) |
| 404 | 资源不存在 |
| 429 | 触发限流 |
| 500 | 服务内部错误 |
| 503 | 依赖不可用(云端 LLM / Chroma 异常) |

---

## 2. 问答接口(核心)

### 2.1 提问(流式)

`POST /chat/ask`(SSE 流式返回)

**请求**
```json
{
  "question": "员工出差住宿补贴标准是多少?",
  "session_id": "uuid-or-null",        // 多轮对话标识;null=新会话
  "top_k": 5,                           // 可选,精排块数,默认 5
  "doc_filter": ["国内差旅费支出管理办法"]  // 可选,限定文档范围(权限内)
}
```

**响应(SSE 事件流)**
```
event: meta          // 首帧:命中来源(引用溯源)
data: {"sources":[{"n":1,"doc":"国内差旅费支出管理办法.pdf","chunk_id":"chunk_11","score":0.993,"snippet":"..."}, ...], "refused": false}

event: delta         // 增量答案 token(流式)
data: {"text": "- 员工出差住宿补贴为 **80 元/人/晚**"}
data: {"text": ",适用于对方免费接待..."}

event: done          // 末帧:完成,附完整答案与统计
data: {"finish": true, "answer": "...(完整 markdown)...", "latency_ms": 5200, "tokens": 380}
```

**拒答场景**(`meta` 帧 `refused: true`)
```
event: meta
data: {"sources": [], "refused": true, "refuse_reason": "根据现有资料无法回答该问题"}
event: done
data: {"finish": true, "answer": "根据现有资料无法回答该问题(未检索到相关内容)。建议换个问法或联系 HR。"}
```

**字段说明**
| 字段 | 说明 |
|------|------|
| `sources[].n` | 引用编号,与答案中 `[来源N]` 对应 |
| `sources[].score` | reranker 相关性分数(供前端展示置信度) |
| `refused` | 是否触发拒答(置信度低于阈值) |
| `cache_hit` | 是否命中答案缓存(M2 P0);命中时 delta 一次性回放完整答案,done 帧 `tokens: 0`、`latency_ms`≈0 |

> **已实现(v1.6,M2 性能优化)**:① **答案缓存**(`engine/anscache.py`,sqlite `backend/.cache/answers.db`):非拒答答案按「指代消解后的独立问题」归一化(去全部空白 + 去结尾句读)做精确匹配缓存,一期不做相似问题模糊缓存;② **失效**:知识库任何变更(上传/删除/重建)经 `retriever._after_mutation()` 统一全清,宁可全清不漏清;③ 读写失败静默放行(缓存是加速层,不是正确性依赖);CLI `ask()` 有意不走缓存,留作调试对照通道;④ **快模型输出封顶** `FAST_MAX_TOKENS`(env,默认 2048,仅 `FAST_LLM_MODEL` 生效;注意 kimi-for-coding 也是思考型——512 曾致空答案,故封顶放宽 + **空输出自动去封顶重试一次**,宁慢不空);空答案不写入缓存(防毒化);⑤ **召回候选收敛** `RECALL_TOP_N` 默认 50,经候选集保持率实验(eval v1 库内 39 题 top-50/30/20 均 100%)本机 `.env` 已收敛至 20,rerank 对数减 60%。

### 2.2 提问(非流式,可选)

`POST /chat/ask/sync` — 参数同上,一次性返回完整 JSON(供脚本/测试用)。

### 2.3 答案反馈

`POST /chat/feedback`
```json
{
  "qa_id": "uuid",            // 一次问答的唯一标识(ask 响应中返回)
  "rating": 1,                // 1=👍, -1=👎
  "issue_type": "not_found",  // 👎时可选:not_found/wrong_answer/wrong_source/bad_refuse/other
  "comment": "可选的补充说明"
}
```
响应:`{"code":0,"data":{"received":true,"bad_case":false,"ts":"..."}}`。👎 样本进入 bad case 池供运营。

> **已实现(v1.3,`engine/feedback.py`)**:全部反馈追加 `logs/feedback.jsonl`;`rating=-1` 同时追加 `logs/bad_cases.jsonl`(bad case 池,带 `status: open` 供 M2 闭环)。M1 阶段问答日志未落盘,不校验 `qa_id` 存在性。

### 2.4 会话管理(多轮,Should)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/chat/sessions` | 当前用户的会话列表 |
| GET | `/chat/sessions/{id}` | 某会话的历史消息 |
| DELETE | `/chat/sessions/{id}` | 删除会话(开启新话题) |

> **已实现(v1.4,多轮对话)**:① `POST /chat/ask` 不带 `session_id` 时自动建会话并在 meta 返回;带则续聊——追问先经指代消解改写为独立问题(`engine/rewriter.py`,快模型,失败退回原问题)再检索,meta 增 `standalone_question`(与原文不同时下发);② GET `/chat/sessions`(列表,含轮数/最近问题)、DELETE `/chat/sessions/{id}`(清空上下文)已落地;GET `…/{id}` 历史消息未做(M2 后续);③ 会话存储为单进程内存版(`engine/session.py`,500 会话/10 轮双封顶),迁多实例换 Redis/业务库。

---

## 3. 知识库管理接口

### 3.1 上传文档并建索引

`POST /documents`(`multipart/form-data`)
```
file: <二进制 PDF/Word/MD/TXT>
```
**响应**(异步,索引为耗时操作)
```json
{"code":0,"data":{"doc_id":"国内差旅费支出管理办法","status":"indexing","task_id":"uuid"}}
```

### 3.2 文档列表

`GET /documents?page=1&size=20`
```json
{"code":0,"data":{"total":2,"items":[
  {"doc_id":"考勤与假期管理办法","filename":"考勤与假期管理办法.pdf",
   "chunks":22,"status":"done","uploaded_at":"2026-07-22T21:00:00",
   "uploader":"admin"}
]}}
```
`status`:`indexing`(索引中)/ `done`(已完成)/ `failed`(失败)。

### 3.3 索引进度查询

`GET /documents/{doc_id}/status`
```json
{"code":0,"data":{"doc_id":"...","status":"indexing","progress":0.6,"chunks_done":13,"chunks_total":22}}
```

### 3.4 删除文档

`DELETE /documents/{doc_id}` — 级联删除其所有块、向量、BM25 项。
```json
{"code":0,"data":{"deleted_chunks":22}}
```

### 3.5 重建索引

`POST /documents/{doc_id}/rebuild`(文档内容更新后)
```json
{"code":0,"data":{"doc_id":"...","status":"indexing","task_id":"uuid"}}
```

---

## 4. 运营/统计接口(Should,管理侧)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/stats/overview` | 核心指标:总提问数、命中率、拒答率、👍率、P95 延迟、token 消耗 |
| GET | `/admin/stats/hot-questions` | 高频问题 Top-N(发现制度盲区) |
| GET | `/admin/bad-cases?status=open` | bad case 列表(👎样本),供标注与修复 |
| POST | `/admin/bad-cases/{id}/resolve` | 标记 bad case 已处理,附处理动作(补文档/调参数) |

> **已实现(v1.5,运营看板)**:四个接口全部落地(`engine/stats.py` + `engine/asklog.py`)。数据基础:每次问答落 `logs/asks.jsonl`(M1 遗留的提问日志缺口已补);bad case 按 qa_id 折叠到最新状态(resolve 为追加写);命中率为作答占比口径。前端「看板」页签:指标卡 + 高频问题 + bad case 闭环(标记已处理附动作)。

**`/admin/stats/overview` 响应示例**
```json
{"code":0,"data":{
  "total_queries": 1280, "hit_rate": 0.93, "refuse_rate": 0.06,
  "thumbs_up_rate": 0.87, "p95_latency_ms": 7600, "tokens_today": 452000
}}
```

---

## 5. 模型管理接口(新增,管理侧)

> 对应 PRD F7。把模型作为平台资产:上传、解析校验、注册、启停切换、删除。

### 5.1 上传模型(异步解析校验)

`POST /models`(`multipart/form-data`)
```
file:        <zip/tar.gz 打包的 HF/sentence-transformers 模型目录>
model_type:  embedding | reranker
name:        自定义模型名(可选,默认取目录名)
```
**响应**(解析校验为异步,先返回受理)
```json
{"code":0,"data":{"model_id":"bge-m3-domain","type":"embedding","status":"validating","task_id":"uuid"}}
```

### 5.2 解析校验结果查询

`GET /models/{model_id}/status`
```json
{"code":0,"data":{
  "model_id":"bge-m3-domain","type":"embedding","status":"ready",
  "metadata":{"dimension":1024,"max_seq_len":8192,"size_mb":2270},
  "progress":1.0
}}
```
校验失败(`status:"failed"`):
```json
{"code":0,"data":{"model_id":"...","status":"failed",
  "error":"缺少必需文件: tokenizer.json;或未通过试加载: <报错摘要>"}}
```

### 5.3 模型列表

`GET /models?type=embedding`
```json
{"code":0,"data":{"items":[
  {"model_id":"bge-m3","type":"embedding","source":"builtin","status":"active",
   "metadata":{"dimension":1024},"is_active":true,"uploaded_at":"2026-07-22T21:19:00"},
  {"model_id":"bge-m3-domain","type":"embedding","source":"uploaded","status":"ready",
   "metadata":{"dimension":1024},"is_active":false,"uploaded_at":"2026-07-23T10:00:00"}
]}}
```
`is_active` 标识当前生效模型。

### 5.4 启用/切换模型

`POST /models/{model_id}/activate`
```json
{"confirm_rebuild": true}   // 嵌入模型切换需确认重建;重排模型可省略
```
**响应**
- 重排模型:直接热切换 `{"code":0,"data":{"model_id":"...","status":"active","rebuild_required":false}}`
- 嵌入模型:返回重建任务 `{"code":0,"data":{"model_id":"...","status":"active","rebuild_required":true,"rebuild_task_id":"uuid"}}`

> 未确认 `confirm_rebuild` 时切换嵌入模型,返回 `409` 提示「切换嵌入模型将重建全库索引,需确认」。

### 5.5 重建索引进度(嵌入模型切换后)

`GET /models/{model_id}/rebuild-status`
```json
{"code":0,"data":{"status":"rebuilding","progress":0.4,"chunks_done":17,"chunks_total":42}}
```

### 5.6 删除模型

`DELETE /models/{model_id}` — 仅允许删除非 active 模型;清理文件与注册表。
```json
{"code":0,"data":{"deleted":true}}
```
删除当前生效模型返回 `409`「模型正在使用中,请先切换到其他模型」。

---



| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查:Chroma 连通、当前生效模型加载、LLM 连通 |

**`/health` 响应**
```json
{"code":0,"data":{
  "chroma": "ok", "chroma_chunks": 42,
  "embed_model": "bge-m3(active, ok)", "rerank_model": "bge-reranker-base(active, ok)",
  "llm": "ok(k3)"
}}
```

---

## 6. 内部模块接口(引擎层,当前代码)

供服务层调用,与现有代码对齐:

```python
# 检索(engine/retriever.py)
retriever.search(query, top_k=50, dense_weight=0.7, where=None) -> list[(Chunk, score)]
# 重排(engine/reranker.py)
reranker.rerank(query, candidates, top_k=5) -> list[(Chunk, score)]
# 生成(engine/generator.py)
generate_answer(query, ranked, model=None, temperature=None) -> str
generate_answer_stream(query, ranked, model=None) -> 迭代器(增量文本,StopIteration.value=usage)
# 编排(engine/pipeline.py)
RAGPipeline().ask(query) -> (answer, ranked)
RAGPipeline().ask_stream(query, top_k=None, doc_filter=None, session_id=None) -> 迭代器((event, data))
```

> **拒答判断**与**多轮查询理解**为新增模块,服务层在 `reranker.rerank` 后依据最高分做拒答分流。

---

## 变更记录

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-07-22 | 首版 | — |
| v1.1 | 2026-07-30 | 已实现:`/chat/ask`(SSE)、`/chat/ask/sync`、`/health`;`meta` 帧补 `qa_id`;统一响应结构经 `_ok()` 落地 | — |
| v1.2 | 2026-07-30 | 已实现:§3 知识库管理五接口(`/documents` 上传/列表/状态/删除/重建);删除含源文件级联(防全量重建复活);任务跟踪为单进程内存版(`api/tasks.py`) | — |
| v1.3 | 2026-07-30 | 已实现:§2.3 `/chat/feedback`(`engine/feedback.py`,全量落 `logs/feedback.jsonl`,👎 同时落 `logs/bad_cases.jsonl` 即 bad case 池);rebuild 接口实测通过(块数幂等) | — |
| v1.4 | 2026-07-30 | 已实现:§2.4 多轮对话——指代消解改写(`engine/rewriter.py`)+ 会话存储(`engine/session.py` 内存版)+ GET/DELETE `/chat/sessions`;meta 增 `model`/`standalone_question` 字段;`/chat/ask/sync` 上游异常 500→503 | — |
| v1.5 | 2026-07-30 | 已实现:§4 运营统计四接口(`engine/stats.py`);补提问日志 `logs/asks.jsonl`(`engine/asklog.py`,M1 遗留缺口);bad case 状态折叠(qa_id 最新记录为准) | — |
| v1.6 | 2026-07-31 | 已实现:§2.1 答案缓存(`engine/anscache.py`,meta 帧增 `cache_hit`,命中回放 `tokens: 0`,日志 `layer="cache"`;失效挂 `_after_mutation()` 全清;CLI 通道不缓存);快模型封顶 `FAST_MAX_TOKENS`;`RECALL_TOP_N` 经保持率实验收敛(本机 .env=20) | — |
| v1.7 | 2026-07-31 | M2 验收修复(见 [[06-M2验收报告]] §3):kimi-for-coding 实为思考型,512 封顶致空答案——封顶放宽至 2048 + 空输出自动去封顶重试一次 + 空答案不写入缓存 | — |
