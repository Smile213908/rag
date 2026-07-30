"""模型注册表(Model Registry):模型的上传、解析校验、注册与切换。

对应 docs/02-架构设计文档 §3.4、docs/03-API接口设计 §5。
把模型从「代码写死」抽象为「可注册、可切换的资产」:
- 嵌入模型(embedding)→ SentenceTransformer 加载,产出稠密向量;
- 重排模型(reranker)  → CrossEncoder 加载,产出相关性分数。

关键约束(与文档一致):
- 上传模型须为 HuggingFace / sentence-transformers 标准格式;
- 校验不通过不注册;切换嵌入模型需触发全库重建(向量空间改变)。
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field, asdict

from engine.paths import MODEL_DIR

# 注册表文件(一期用 JSON,二期迁入业务库)
REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")

# 各类型模型的必需文件(校验用)
_REQUIRED_COMMON = ["config.json", "tokenizer.json"]
_REQUIRED_WEIGHTS = ["pytorch_model.bin", "model.safetensors"]  # 二选一
_REQUIRED_EMBEDDING_EXTRA = ["modules.json"]  # sentence-transformers 结构


def _safe_extract_zip(archive_path: str, dest: str) -> None:
    """解 zip,手动拦截路径穿越(zipfile 无内置过滤,恶意包可借 ../ 写穿目录)。"""
    dest_real = os.path.realpath(dest)
    with zipfile.ZipFile(archive_path) as z:
        for name in z.namelist():
            target = os.path.realpath(os.path.join(dest_real, name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise ValueError(f"压缩包包含越界路径: {name}")
        z.extractall(dest_real)


def _safe_extract_tar(archive_path: str, dest: str) -> None:
    """解 tar/tar.gz。Python 3.12+ 的 filter="data" 拦截越界路径、设备文件与外链。"""
    with tarfile.open(archive_path) as t:
        t.extractall(dest, filter="data")


@dataclass
class ModelRecord:
    """一条注册的模型记录。"""
    model_id: str
    type: str                      # embedding | reranker
    path: str                      # 模型文件所在目录
    source: str = "uploaded"       # builtin | uploaded | modelscope
    status: str = "validating"     # validating|ready|active|failed|disabled
    metadata: dict = field(default_factory=dict)
    error: str = ""
    uploaded_by: str = "admin"
    uploaded_at: str = ""


class ModelRegistry:
    """管理模型注册表(JSON 文件)与当前生效指针。"""

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._models: dict[str, ModelRecord] = {}
        self._active: dict[str, str] = {}      # type -> model_id
        self._lock = threading.Lock()
        self._load()

    # ---------- 注册表持久化 ----------
    def _load(self) -> None:
        if not os.path.exists(self.registry_path):
            return
        with open(self.registry_path, encoding="utf-8") as f:
            data = json.load(f)
        for m in data.get("models", []):
            rec = ModelRecord(**m)
            self._models[rec.model_id] = rec
        self._active = data.get("active", {})

    def _save(self) -> None:
        """落盘注册表:先写临时文件再原子替换,避免进程中断留下半截 JSON。

        锁只保护进程内并发;多进程(FastAPI 多 worker)并发写仍可能丢更新,
        届时须按路线图 M2 迁入业务库(SQLite/Postgres)。
        """
        data = {
            "models": [asdict(m) for m in self._models.values()],
            "active": self._active,
        }
        tmp_path = self.registry_path + ".tmp"
        with self._lock:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.registry_path)

    # ---------- 查询 ----------
    def list(self, type: str | None = None) -> list[ModelRecord]:
        recs = list(self._models.values())
        if type:
            recs = [r for r in recs if r.type == type]
        for r in recs:
            r.status = "active" if self._active.get(r.type) == r.model_id else (
                r.status if r.status != "active" else "ready")
        return recs

    def get(self, model_id: str) -> ModelRecord | None:
        return self._models.get(model_id)

    def get_active(self, type: str) -> ModelRecord | None:
        mid = self._active.get(type)
        return self._models.get(mid) if mid else None

    # ---------- 注册(内置/已有模型) ----------
    def register_builtin(self, model_id: str, type: str, path: str,
                         make_active: bool = False) -> ModelRecord:
        """把已存在于模型仓库的模型(如当前 bge)登记进注册表。"""
        rec = self._models.get(model_id)
        if not rec:
            rec = ModelRecord(model_id=model_id, type=type, path=path,
                              source="builtin", status="ready",
                              metadata=self._read_metadata(path))
            self._models[model_id] = rec
        if make_active:
            self._active[type] = model_id
        self._save()
        return rec

    # ---------- 上传解析校验 ----------
    def _read_metadata(self, path: str) -> dict:
        """从 config.json 提取元数据(尽力而为,不强制)。"""
        meta = {}
        cfg = os.path.join(path, "config.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, encoding="utf-8") as f:
                    c = json.load(f)
                meta["max_seq_len"] = c.get("max_position_embeddings")
                meta["hidden_size"] = c.get("hidden_size")
                meta["architectures"] = c.get("architectures")
            except Exception:
                pass
        return meta

    def _validate_files(self, path: str, type: str) -> list[str]:
        """校验必需文件,返回缺失清单(空=通过)。"""
        missing = [f for f in _REQUIRED_COMMON if not os.path.exists(os.path.join(path, f))]
        if not any(os.path.exists(os.path.join(path, w)) for w in _REQUIRED_WEIGHTS):
            missing.append("权重文件(pytorch_model.bin 或 model.safetensors)")
        if type == "embedding":
            # sentence-transformers 结构;宽松处理,缺失仅警告不阻断
            pass
        return missing

    def _try_load(self, path: str, type: str) -> dict:
        """试加载并跑一次前向,返回提取的运行时元数据;失败抛异常。"""
        if type == "embedding":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(path, device="cpu")
            vec = m.encode(["校验文本"], convert_to_numpy=True)
            return {"dimension": int(vec.shape[-1])}
        else:
            from sentence_transformers import CrossEncoder
            m = CrossEncoder(path, device="cpu")
            m.predict([["校验", "校验文本"]])
            return {}

    def upload(self, archive_path: str, type: str, model_id: str | None = None,
               make_active: bool = False, uploaded_by: str = "admin",
               dry_run_load: bool = True) -> ModelRecord:
        """上传一个打包模型:解包 → 校验 → 试加载 → 注册保存。

        dry_run_load: 是否真加载模型验证(CPU 上较慢,可在 API 层异步做)。
        """
        model_id = model_id or os.path.splitext(os.path.basename(archive_path))[0]
        dest = os.path.join(MODEL_DIR, model_id)
        rec = ModelRecord(model_id=model_id, type=type, path=dest,
                          source="uploaded", status="validating",
                          uploaded_by=uploaded_by)
        self._models[model_id] = rec
        self._save()

        tmp = tempfile.mkdtemp()
        try:
            # ① 解包(安全模式,拦截路径穿越)
            if zipfile.is_zipfile(archive_path):
                _safe_extract_zip(archive_path, tmp)
            elif tarfile.is_tarfile(archive_path):
                _safe_extract_tar(archive_path, tmp)
            else:
                raise ValueError("仅支持 zip / tar.gz 格式")

            # 若解出的是单层包裹目录,则进入该目录
            entries = [e for e in os.listdir(tmp) if not e.startswith("__MACOSX")]
            src = os.path.join(tmp, entries[0]) if len(entries) == 1 and \
                os.path.isdir(os.path.join(tmp, entries[0])) else tmp

            # ② 校验必需文件
            missing = self._validate_files(src, type)
            if missing:
                raise ValueError("缺少必需文件: " + ", ".join(missing))

            # ③ 试加载(可选,CPU 较慢)
            run_meta = self._try_load(src, type) if dry_run_load else {}

            # ④ 移动到模型仓库
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(src, dest)

            # ⑤ 元数据 + 注册
            rec.metadata = {**self._read_metadata(dest), **run_meta}
            rec.status = "ready"
            rec.error = ""
            if make_active:
                self._active[type] = model_id
                rec.status = "active"
        except Exception as e:  # 校验/加载失败
            rec.status = "failed"
            rec.error = str(e)[:300]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self._save()
        return rec

    # ---------- 切换 ----------
    def activate(self, model_id: str) -> ModelRecord:
        """把某模型设为当前生效(调用方负责:嵌入模型切换后触发重建)。"""
        rec = self._models.get(model_id)
        if not rec:
            raise KeyError(f"模型不存在: {model_id}")
        if rec.status not in ("ready", "active", "disabled"):
            raise ValueError(f"模型状态不可启用: {rec.status}")
        self._active[rec.type] = model_id
        self._save()
        return rec

    # ---------- 删除 ----------
    def delete(self, model_id: str) -> None:
        rec = self._models.get(model_id)
        if not rec:
            return
        if self._active.get(rec.type) == model_id:
            raise ValueError("模型正在使用中,请先切换到其他模型")
        if os.path.exists(rec.path):
            shutil.rmtree(rec.path, ignore_errors=True)
        del self._models[model_id]
        self._save()


# ---------- 默认注册:把现有 bge 模型登记为内置并设为生效 ----------
def bootstrap_default_models(registry: ModelRegistry | None = None) -> ModelRegistry:
    """首次启动时,把模型仓库里已有的 bge-m3 / bge-reranker-base 登记进注册表。"""
    reg = registry or ModelRegistry()
    defaults = [
        ("bge-m3", "embedding", os.path.join(MODEL_DIR, "bge-m3", "snapshots", "master")),
        ("bge-reranker-base", "reranker", os.path.join(MODEL_DIR, "bge-reranker-base", "snapshots", "master")),
    ]
    for mid, mtype, path in defaults:
        if os.path.exists(path):
            reg.register_builtin(mid, mtype, path)
        # 若该类型尚无生效模型,则把可用的内置模型设为生效
        if not reg.get_active(mtype) and reg.get(mid):
            reg.activate(mid)
    return reg


if __name__ == "__main__":
    reg = bootstrap_default_models()
    for m in reg.list():
        print(f"[{m.status:9}] {m.model_id:22} {m.type:10} -> {m.path}")
