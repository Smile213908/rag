"""分块:按结构递归切分 + overlap。

原则(见大脑笔记「RAG 离线索引工程」):
- 分块是**程序逻辑**,不用模型。bge-m3 的上下文长度只决定单块上限。
- 优先按文档结构(标题>段落>句子>字符)递归切,语义完整度远好于硬切。
- 黄金区间 256~512 token,overlap 10~15%。中文场景按字符近似(1 token≈1.5~2 字符)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个文本块,带元数据(血缘溯源用)。"""
    text: str
    doc_id: str
    chunk_id: str
    meta: dict = field(default_factory=dict)

    @property
    def cid(self) -> str:  # 确定性 ID:doc#chunk
        return f"{self.doc_id}#{self.chunk_id}"


# 结构分隔符,按优先级从高到低(标题 > 段落 > 句子 > 字符)
_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", "!", "?", ";", " "]


def _split_recursive(text: str, max_size: int, seps: list[str]) -> list[str]:
    """递归地按分隔符把 text 切到 max_size 以内。"""
    if len(text) <= max_size:
        return [text] if text.strip() else []
    if not seps:  # 没有可用分隔符了,硬切
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep)
    chunks, current = [], ""
    for piece in pieces:
        piece = (piece + sep) if sep.strip() else piece
        if len(piece) > max_size:  # 单块仍超,递归用更细分隔符
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_recursive(piece, max_size, rest))
        elif len(current) + len(piece) <= max_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    max_size: int = 400,
    overlap: int = 50,
    meta: dict | None = None,
) -> list[Chunk]:
    """把一篇文档切成带 overlap 的块。

    max_size/overlap 单位为字符(中文近似)。默认 400 字≈260 token,落黄金区间。
    """
    meta = meta or {}
    base = _split_recursive(text, max_size, _SEPARATORS)

    # 加 overlap:每块末尾拼上下一块的开头,防止语义在边界被切断
    out: list[Chunk] = []
    for i, b in enumerate(base):
        overlapped = b
        if i + 1 < len(base) and overlap > 0:
            overlapped = b + base[i + 1][:overlap]
        out.append(Chunk(text=overlapped.strip(), doc_id=doc_id,
                         chunk_id=f"chunk_{i}", meta=dict(meta)))
    return out


def chunk_markdown_file(path: str, max_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """读取一个 markdown/txt 文件并分块,doc_id 用文件名。"""
    import os
    with open(path, encoding="utf-8") as f:
        text = f.read()
    doc_id = os.path.splitext(os.path.basename(path))[0]
    return chunk_text(text, doc_id, max_size, overlap,
                      meta={"source": os.path.basename(path)})


def _clean_pdf_text(text: str) -> str:
    """清洗 PDF 提取文本中的页眉页脚噪声(如「XX字(2022)第001号,第2页共8页」)。

    见大脑笔记「RAG 离线索引工程」:页眉页脚混入会污染每个块,是解析质量分水岭。
    """
    # 匹配「第N页共M页」及其同行的文件字号信息
    text = re.sub(r"[^\n]*第\s*\d+\s*页\s*共\s*\d+\s*页[^\n]*", "", text)
    # 匹配独立的文件字号行,如「蓝卓办字(2022)第001 号」
    text = re.sub(r"^[^\n]*[（(]\s*\d{4}\s*[)）]\s*第?\s*\d+\s*号[^\n]*$", "", text, flags=re.M)
    # 收起多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_pdf_text(path: str) -> str:
    """从 PDF 提取纯文本并清洗页眉页脚。

    用 PyMuPDF(fitz):对纯文字 PDF(如制度办法类)效果好、速度快。
    复杂版面(双栏/大量表格/扫描件)需换版面分析模型(MinerU 等),此处从简。
    """
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return _clean_pdf_text("\n\n".join(pages))


def chunk_pdf_file(path: str, max_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """读取一个 PDF 并分块,doc_id 用文件名,meta 记录页数信息。"""
    import os
    text = extract_pdf_text(path)
    doc_id = os.path.splitext(os.path.basename(path))[0]
    return chunk_text(text, doc_id, max_size, overlap,
                      meta={"source": os.path.basename(path), "type": "pdf"})
