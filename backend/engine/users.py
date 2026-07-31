"""用户账号存储(M2 权限控制一期:仅播种 admin;部门角色字段预留,暂不使用)。

- sqlite 单表 users(username 主键, pass_hash, salt, role, created_at);
  库文件 backend/users.db(含密码哈希,已 gitignore,不入库)。
- 密码 PBKDF2-HMAC-SHA256 20 万次迭代 + 16B 随机盐,纯 stdlib 零新依赖。
- 播种 admin:密码取环境变量 ADMIN_PASSWORD;未设置时随机生成并只打印一次。

CLI:
  python -m engine.users init                 # 建表 + 播种 admin(幂等)
  python -m engine.users passwd admin <新密码> # 改密
  python -m engine.users verify admin <密码>   # 校验(调试用)
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import sys
import time

from engine.paths import PROJECT_ROOT

DB_PATH = os.path.join(PROJECT_ROOT, "users.db")

_ITER = 200_000


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, "
        "pass_hash TEXT NOT NULL, "
        "salt TEXT NOT NULL, "
        "role TEXT NOT NULL DEFAULT 'admin', "
        "created_at REAL NOT NULL)"
    )
    return conn


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITER
    ).hex()


def create_user(username: str, password: str, role: str = "admin") -> bool:
    """新建用户;已存在返回 False 不动原账号。"""
    salt = secrets.token_hex(16)
    with _conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?)",
                (username, _hash(password, salt), salt, role, time.time()),
            )
        except sqlite3.IntegrityError:
            return False
    return True


def verify(username: str, password: str) -> bool:
    """校验密码;用户不存在即 False。"""
    with _conn() as conn:
        row = conn.execute(
            "SELECT pass_hash, salt FROM users WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return False
    return secrets.compare_digest(row[0], _hash(password, row[1]))


def get_role(username: str) -> str | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username=?", (username,)
        ).fetchone()
    return row[0] if row else None


def change_password(username: str, new_password: str) -> bool:
    salt = secrets.token_hex(16)
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE users SET pass_hash=?, salt=? WHERE username=?",
            (_hash(new_password, salt), salt, username),
        )
    return cur.rowcount > 0


def seed_admin() -> None:
    """播种 admin(幂等)。密码来源:ADMIN_PASSWORD 环境变量 > 随机生成。"""
    with _conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username='admin'"
        ).fetchone()
    if exists:
        print("admin 已存在,跳过播种(改密用 passwd 命令)。")
        return
    pw = os.environ.get("ADMIN_PASSWORD")
    if pw:
        print("admin 已播种(密码来自 ADMIN_PASSWORD 环境变量,未回显)。")
    else:
        pw = secrets.token_urlsafe(9)
        print(f"admin 已播种,初始密码(只显示这一次,请立即记录并改密): {pw}")
    create_user("admin", pw, role="admin")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        seed_admin()
    elif cmd == "passwd" and len(sys.argv) == 4:
        ok = change_password(sys.argv[2], sys.argv[3])
        print("改密成功" if ok else "用户不存在")
    elif cmd == "verify" and len(sys.argv) == 4:
        print("校验通过" if verify(sys.argv[2], sys.argv[3]) else "校验失败")
    else:
        print(__doc__)
