"""采集来源模型（Phase 11.1）：``CaptureSource`` / ``SourceRegistry``。

```text
capture_id   唯一采集记录（一次"按下截图"）
session_id   一次连续采集会话（同一次实机联调 / 同一次连续截图）
source_id    稳定原图身份（= 内容身份，默认由 sha256 派生）
source_sha256 字节级完整性校验
```

**历史数据（178 张 raw）的现实**：

* 现有素材由 `tools/build_dataset.py` 从旧目录整理而来，**没有**采集会话记录；
* 因此历史条目的 ``session_id`` / ``captured_at`` / 几何字段一律**显式置空**
  （``None`` / 空串），**不得**根据目录名、文件名或时间顺序**凭空补造**；
* 新采集数据必须在**采集当时**登记来源，而不是事后推测。

依赖边界：只 import 标准库 + ``src.core`` / ``src.utils``。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

__all__ = [
    "CAPTURE_ROLE_CANDIDATE",
    "CAPTURE_ROLE_DIAGNOSTIC",
    "CAPTURE_ROLE_HOLDOUT",
    "CAPTURE_ROLE_UNKNOWN",
    "CAPTURE_ROLES",
    "CAPTURE_SOURCE_FIELDS",
    "CaptureSource",
    "SourceError",
    "SourceRegistry",
    "TEMPLATE_ELIGIBLE_ROLES",
    "UNKNOWN_VALUE",
    "compute_sha256",
    "image_size",
    "local_now_iso",
    "source_id_for",
]

logger = get_logger("vision.dataset.sources")

#: 历史数据中"来源不明"字段的占位（显式未知，不允许编造）
UNKNOWN_VALUE = "unknown"

#: 采集角色（Phase 11.9-A）：一张 raw 在 **candidate / holdout / diagnostic** 中的身份。
#:
#: 关键约束：历史素材与任何"未显式标注"的新图一律为 :data:`CAPTURE_ROLE_UNKNOWN`，
#: **不得**默认归入 ``candidate`` 或 ``holdout`` —— 否则会在模板选择阶段造成自匹配泄漏。
CAPTURE_ROLE_UNKNOWN = "unknown"
CAPTURE_ROLE_CANDIDATE = "candidate"
CAPTURE_ROLE_HOLDOUT = "holdout"
CAPTURE_ROLE_DIAGNOSTIC = "diagnostic"
CAPTURE_ROLES: frozenset[str] = frozenset(
    {
        CAPTURE_ROLE_CANDIDATE,
        CAPTURE_ROLE_DIAGNOSTIC,
        CAPTURE_ROLE_HOLDOUT,
        CAPTURE_ROLE_UNKNOWN,
    }
)

#: 仅"可用于模板生成 / 模板选择"的角色（holdout / diagnostic / unknown **不得**入选）。
TEMPLATE_ELIGIBLE_ROLES: frozenset[str] = frozenset({CAPTURE_ROLE_CANDIDATE})

#: 采集来源允许的字段
CAPTURE_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "source_id",
        "capture_id",
        "session_id",
        "relative_path",
        "captured_at",
        "window_width",
        "window_height",
        "client_width",
        "client_height",
        "display_scale",
        "capture_backend",
        "game_scene",
        "game_skin",
        "source_sha256",
        "origin",
        "note",
        "capture_role",
        "theme_or_event",
        "yard_variant",
    }
)


class SourceError(ValueError):
    """来源记录非法。"""


def local_now_iso() -> str:
    """本地 ISO 8601 时间戳。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def compute_sha256(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """计算文件 SHA256（字节级完整性）。

    Args:
        path: 文件路径。
        chunk_size: 分块大小。

    Returns:
        十六进制摘要（小写）。

    Raises:
        SourceError: 文件不存在或读取失败。
    """
    target = Path(path)
    if not target.is_file():
        raise SourceError(f"文件不存在: {target}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SourceError(f"读取失败: {target}: {exc}") from exc
    return digest.hexdigest()


def source_id_for(path: Path | str, *, prefix: str = "src") -> str:
    """由**内容**派生稳定 ``source_id``（``<prefix>_<sha256 前 16 位>``）。

    用内容而非文件名，保证"同一张原图被改名 / 被复制"时身份不变。

    Args:
        path: 原图路径。
        prefix: 前缀。

    Returns:
        稳定身份字符串。
    """
    return f"{str(prefix)}_{compute_sha256(path)[:16]}"


def image_size(path: Path | str) -> tuple[int, int]:
    """读取图片尺寸（**兼容非 ASCII 路径**：``np.fromfile`` + ``cv2.imdecode``）。

    Args:
        path: 图片路径。

    Returns:
        ``(width, height)``；读取失败返回 ``(0, 0)``。
    """
    import cv2

    target = Path(path)
    try:
        buffer = np.fromfile(str(target), dtype=np.uint8)
        if buffer.size == 0:
            return (0, 0)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - 读取失败不应中断调用方
        logger.warning("图片读取失败: %s: %s", target, exc)
        return (0, 0)
    if image is None:
        return (0, 0)
    height, width = image.shape[:2]
    return (int(width), int(height))


@dataclass(slots=True)
class CaptureSource:
    """一条采集来源记录（Phase 11.1）。

    Attributes:
        source_id: 稳定原图身份（``src_<sha256前16>``）。
        relative_path: 相对 ``dataset/`` 的路径（例如 ``raw/yard/yard_cap_001.png``）。
        capture_id / session_id: 采集记录 / 会话（**历史数据为 ``None``**）。
        captured_at: 采集时间（历史数据为 ``None``）。
        window_width / window_height: 窗口尺寸（未知为 ``0``）。
        client_width / client_height: **客户区**尺寸（bbox 坐标所在的帧尺寸）。
        display_scale: 显示缩放（例如 ``1.5``；未知为 ``0.0``）。
        capture_backend: 采集后端（``dxcam`` / ``mss`` / ``unknown``）。
        game_scene / game_skin: 游戏场景 / 皮肤（未知为 ``unknown``）。
        source_sha256: 原图 SHA256。
        origin: 来源说明（``legacy-import`` / ``live-capture`` …）。
        note: 备注。
        capture_role: 采集角色（``candidate`` / ``holdout`` / ``diagnostic`` / ``unknown``）。
            历史素材与未标注新图一律 ``unknown``（**不**默认成 candidate/holdout）。
        theme_or_event: 主题 / 活动（无法可靠识别时 ``unknown``，不猜）。
        yard_variant: 庭院变体（无法可靠识别时 ``unknown``，不猜）。
    """

    source_id: str
    relative_path: str
    capture_id: str | None = None
    session_id: str | None = None
    captured_at: str | None = None
    window_width: int = 0
    window_height: int = 0
    client_width: int = 0
    client_height: int = 0
    display_scale: float = 0.0
    capture_backend: str = UNKNOWN_VALUE
    game_scene: str = UNKNOWN_VALUE
    game_skin: str = UNKNOWN_VALUE
    source_sha256: str = ""
    origin: str = UNKNOWN_VALUE
    note: str = ""
    capture_role: str = CAPTURE_ROLE_UNKNOWN
    theme_or_event: str = UNKNOWN_VALUE
    yard_variant: str = UNKNOWN_VALUE

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id or "").strip()
        if not self.source_id:
            raise SourceError("来源记录必须携带 source_id")
        self.relative_path = str(self.relative_path or "").strip()
        if not self.relative_path:
            raise SourceError("来源记录必须携带 relative_path")
        self.capture_id = _optional(self.capture_id)
        self.session_id = _optional(self.session_id)
        self.captured_at = _optional(self.captured_at)
        self.capture_role = normalize_capture_role(self.capture_role)
        self.theme_or_event = _text_or_unknown(self.theme_or_event)
        self.yard_variant = _text_or_unknown(self.yard_variant)

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def has_session(self) -> bool:
        """是否具备真实采集会话信息（历史数据为 ``False``）。"""
        return bool(self.session_id)

    @property
    def has_capture(self) -> bool:
        """是否具备真实采集记录。"""
        return bool(self.capture_id)

    @property
    def client_size(self) -> tuple[int, int]:
        """客户区尺寸 ``(width, height)``（未知为 ``(0, 0)``）。"""
        return (int(self.client_width), int(self.client_height))

    def group_key(self) -> str:
        """分组键（泄漏防护用）：``session_id → capture_id → source_id``。"""
        return self.session_id or self.capture_id or self.source_id

    def group_level(self) -> str:
        """分组键的层级（``session`` / ``capture`` / ``source``）。"""
        if self.session_id:
            return "session"
        if self.capture_id:
            return "capture"
        return "source"

    @property
    def has_role(self) -> bool:
        """是否已显式确定采集角色（历史素材与未标注新图为 ``False``）。"""
        return self.capture_role != CAPTURE_ROLE_UNKNOWN

    @property
    def is_candidate(self) -> bool:
        """是否属于 candidate 池（**唯一**可用于模板生成 / 模板选择的角色）。"""
        return self.capture_role == CAPTURE_ROLE_CANDIDATE

    @property
    def is_holdout(self) -> bool:
        """是否属于 holdout 池（禁止参与模板生成 / 选择 / 调参）。"""
        return self.capture_role == CAPTURE_ROLE_HOLDOUT

    @property
    def is_diagnostic(self) -> bool:
        """是否属于 diagnostic 池（只用于定位问题，不选模板）。"""
        return self.capture_role == CAPTURE_ROLE_DIAGNOSTIC

    @property
    def is_template_eligible(self) -> bool:
        """是否允许进入模板生成 / 模板选择（仅 ``candidate``）。"""
        return self.capture_role in TEMPLATE_ELIGIBLE_ROLES

    @classmethod
    def legacy(
        cls,
        *,
        relative_path: str,
        source_id: str = "",
        digest: str = "",
        origin: str = "legacy-import",
        note: str = "历史素材：无采集会话信息（不得凭空补造）",
    ) -> CaptureSource:
        """构造**历史素材**来源记录（会话 / 时间 / 几何一律显式为空）。

        Args:
            relative_path: 相对 ``dataset/`` 路径。
            source_id: 稳定身份（为空时由 ``digest`` 派生）。
            digest: 已计算的 SHA256。
            origin: 来源说明。
            note: 备注。

        Returns:
            来源记录。
        """
        identity = source_id or (f"src_{digest[:16]}" if digest else "")
        if not identity:
            raise SourceError("legacy 来源需要 source_id 或 digest")
        return cls(
            source_id=identity,
            relative_path=relative_path,
            source_sha256=digest,
            origin=origin,
            note=note,
        )

    # ---- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（未知字段保持 ``null`` / ``unknown``）。"""
        return {
            "source_id": self.source_id,
            "capture_id": self.capture_id,
            "session_id": self.session_id,
            "relative_path": self.relative_path,
            "captured_at": self.captured_at,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "client_width": self.client_width,
            "client_height": self.client_height,
            "display_scale": self.display_scale,
            "capture_backend": self.capture_backend,
            "game_scene": self.game_scene,
            "game_skin": self.game_skin,
            "source_sha256": self.source_sha256,
            "origin": self.origin,
            "note": self.note,
            "capture_role": self.capture_role,
            "theme_or_event": self.theme_or_event,
            "yard_variant": self.yard_variant,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaptureSource:
        """从字典构造（未知字段报错）。

        Raises:
            SourceError: 出现未知字段。
        """
        unknown = sorted(set(map(str, data)) - CAPTURE_SOURCE_FIELDS)
        if unknown:
            raise SourceError(f"来源记录出现未知字段: {unknown}")
        return cls(
            source_id=str(data.get("source_id", "")),
            relative_path=str(data.get("relative_path", "")),
            capture_id=data.get("capture_id"),
            session_id=data.get("session_id"),
            captured_at=data.get("captured_at"),
            window_width=int(data.get("window_width", 0) or 0),
            window_height=int(data.get("window_height", 0) or 0),
            client_width=int(data.get("client_width", 0) or 0),
            client_height=int(data.get("client_height", 0) or 0),
            display_scale=float(data.get("display_scale", 0.0) or 0.0),
            capture_backend=str(data.get("capture_backend", UNKNOWN_VALUE) or UNKNOWN_VALUE),
            game_scene=str(data.get("game_scene", UNKNOWN_VALUE) or UNKNOWN_VALUE),
            game_skin=str(data.get("game_skin", UNKNOWN_VALUE) or UNKNOWN_VALUE),
            source_sha256=str(data.get("source_sha256", "") or ""),
            origin=str(data.get("origin", UNKNOWN_VALUE) or UNKNOWN_VALUE),
            note=str(data.get("note", "") or ""),
            capture_role=data.get("capture_role", CAPTURE_ROLE_UNKNOWN),
            theme_or_event=data.get("theme_or_event", UNKNOWN_VALUE),
            yard_variant=data.get("yard_variant", UNKNOWN_VALUE),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        session = self.session_id or "no-session"
        size = f"{self.client_width}x{self.client_height}" if self.client_width else "size-unknown"
        return f"CaptureSource({self.source_id} {self.relative_path} {session} {size})"


def normalize_capture_role(value: Any) -> str:
    """把任意输入归一为**合法**采集角色。

    规则：空值 ⇒ :data:`CAPTURE_ROLE_UNKNOWN`；大小写与首尾空白被归一；
    其余非法取值**报错**（不静默降级为 candidate/holdout）。

    Args:
        value: 原始取值。

    Returns:
        合法角色字符串。

    Raises:
        SourceError: 取值不在 :data:`CAPTURE_ROLES` 中。
    """
    text = str(value or "").strip().lower() or CAPTURE_ROLE_UNKNOWN
    if text not in CAPTURE_ROLES:
        allowed = ", ".join(sorted(CAPTURE_ROLES))
        raise SourceError(f"capture_role 非法: {text!r}（允许: {allowed}）")
    return text


def _text_or_unknown(value: Any) -> str:
    """把空值归一为 ``unknown``（显式未知），其余转去空白字符串。"""
    text = str(value or "").strip()
    return text or UNKNOWN_VALUE


def _optional(value: Any) -> str | None:
    """把空值归一为 ``None``（显式未知），其余转字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class SourceRegistry:
    """来源登记表（Phase 11.1）。

    Args:
        sources: 预载记录。
        path: JSONL 路径（``load`` / ``save`` 用）。

    Notes:
        ``by_sha256`` / ``by_path`` 提供两种查找口径；
        历史条目缺 ``session_id`` 时 :meth:`assert_independent` 会**保守拒绝**。
    """

    def __init__(
        self,
        sources: Iterable[CaptureSource] = (),
        *,
        path: Path | str | None = None,
    ) -> None:
        self._items: dict[str, CaptureSource] = {}
        self._errors: list[str] = []
        self._path = Path(path) if path is not None else None
        for item in sources:
            self.add(item)

    # ---- 只读 ---------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """JSONL 路径。"""
        return self._path

    @property
    def errors(self) -> tuple[str, ...]:
        """加载期错误。"""
        return tuple(self._errors)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def add(self, source: CaptureSource) -> CaptureSource:
        """登记一条来源（``source_id`` 重复时报错，避免身份冲突）。"""
        if source.source_id in self._items:
            raise SourceError(f"source_id 重复: {source.source_id}")
        self._items[source.source_id] = source
        return source

    def get(self, source_id: str) -> CaptureSource | None:
        """按身份取记录。"""
        return self._items.get(str(source_id))

    def by_path(self, relative_path: str) -> CaptureSource | None:
        """按相对路径取记录（大小写不敏感）。"""
        key = str(relative_path).replace("\\", "/").casefold()
        for item in self._items.values():
            if item.relative_path.replace("\\", "/").casefold() == key:
                return item
        return None

    def by_sha256(self, digest: str) -> CaptureSource | None:
        """按 SHA256 取记录。"""
        key = str(digest).lower()
        for item in self._items.values():
            if item.source_sha256.lower() == key:
                return item
        return None

    def sessions(self) -> tuple[str, ...]:
        """出现过的会话 ID（不含历史 ``None``）。"""
        return tuple(dict.fromkeys(item.session_id for item in self._items.values() if item.session_id))

    def legacy_count(self) -> int:
        """缺采集会话信息的记录数（历史素材）。"""
        return sum(1 for item in self._items.values() if not item.has_session)

    def duplicate_paths(self) -> tuple[str, ...]:
        """同一路径被登记多次（应为 0）。"""
        seen: dict[str, int] = {}
        for item in self._items.values():
            key = item.relative_path.replace("\\", "/").casefold()
            seen[key] = seen.get(key, 0) + 1
        return tuple(sorted(key for key, count in seen.items() if count > 1))

    def duplicate_sha256(self) -> tuple[str, ...]:
        """内容完全相同的原图（**完全重复**检测）。"""
        seen: dict[str, list[str]] = {}
        for item in self._items.values():
            if not item.source_sha256:
                continue
            seen.setdefault(item.source_sha256.lower(), []).append(item.source_id)
        return tuple(
            sorted(f"{digest}: {sorted(ids)}" for digest, ids in seen.items() if len(ids) > 1)
        )

    def assert_independent(self, source_ids: Iterable[str], *, require_session: bool = True) -> None:
        """断言给定原图属于**不同**采集会话（泄漏防护的保守检查）。

        Args:
            source_ids: 待检查的原图身份。
            require_session: 是否要求必须存在会话信息（``True`` = 缺失即拒绝）。

        Raises:
            SourceError: 存在同会话原图，或（``require_session``）会话信息缺失。
        """
        keys: dict[str, list[str]] = {}
        for source_id in source_ids:
            source = self.get(str(source_id))
            if source is None:
                raise SourceError(f"未登记的原图身份: {source_id}")
            if require_session and not source.has_session:
                raise SourceError(
                    f"原图缺少采集会话信息（不得假定独立采集）: {source_id} "
                    f"（{source.relative_path}）"
                )
            keys.setdefault(source.group_key(), []).append(source.source_id)
        shared = {key: value for key, value in keys.items() if len(value) > 1}
        if shared:
            raise SourceError(f"同源原图不得跨集合: {shared}")

    # ---- 构造 / 写入 --------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> SourceRegistry:
        """从 JSONL 加载（缺失文件视为空表）。"""
        target = Path(path)
        registry = cls(path=target)
        if not target.is_file():
            logger.debug("来源文件不存在（视为空表）: %s", target)
            return registry
        for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise SourceError(f"第 {number} 行不是 JSON 对象")
                registry.add(CaptureSource.from_dict(payload))
            except Exception as exc:  # noqa: BLE001
                registry._errors.append(f"{target.name}:{number}: {exc}")  # noqa: SLF001
                logger.warning("来源行解析失败（已跳过）: %s:%d: %s", target.name, number, exc)
        return registry

    def to_jsonl(self) -> str:
        """序列化为 JSONL。"""
        return "\n".join(
            json.dumps(item.to_dict(), ensure_ascii=False) for item in self._items.values()
        ) + ("\n" if self._items else "")

    def save(self, path: Path | str | None = None) -> Path:
        """写入 JSONL（**只写登记文件，绝不触碰 raw 图片**）。"""
        target = Path(path) if path is not None else self._path
        if target is None:
            raise SourceError("未提供来源输出路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl(), encoding="utf-8")
        logger.info("已写入 %d 条来源记录: %s", len(self._items), target)
        return target

    def stats(self) -> dict[str, Any]:
        """统计（会话数 / 历史缺信息数 / 完全重复数 —— 分开计数）。"""
        return {
            "sources": len(self._items),
            "sessions": len(self.sessions()),
            "legacy_without_session": self.legacy_count(),
            "duplicate_sha256": len(self.duplicate_sha256()),
            "duplicate_paths": len(self.duplicate_paths()),
            "errors": len(self._errors),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        stats = self.stats()
        return (
            f"SourceRegistry(sources={stats['sources']}, sessions={stats['sessions']}, "
            f"legacy_without_session={stats['legacy_without_session']}, "
            f"duplicate_sha256={stats['duplicate_sha256']})"
        )
