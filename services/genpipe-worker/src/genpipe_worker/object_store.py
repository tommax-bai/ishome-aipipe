"""出站边缘：读用户上传的户型图、写解析派生物——**私有对象存储**（阿里云 OSS 私有桶）。

上传入口就绪之后（2026-09-04 接线），activity 拿到的是**对象键**不是本地路径：
这正是当初把 `floorplan-parse` 后置的理由——"提前接线等于接一遍再改一遍"。

**本模块只读源图、只写派生物、不签名**。签名是"给谁看、看多久"的事，属业务侧
（生成侧不知用户是谁）。派生物与源图**同前缀**（`uploads/{content_sha256}/…`），
形态照 render2d `plan_store` 与 imagegen `image_store` 的既有默认；注册进 contracts
`registries/object_keys.md` 的时点写死＝中控仓那侧统一改表那一次（本仓不动那个文件）。

依赖方向（import-linter 锁定）：本模块只依赖运行库（oss2），不感知上层——
拿图的那只手不知道图会被拿去算什么。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import oss2

UPLOAD_ORIGINAL_KEY_TEMPLATE = "uploads/{content_sha256}/original.{ext}"
"""源户型图的对象键模板。**唯一真源在 contracts `registries/object_keys.md`**，本行是逐字副本
（写的一侧 channel-svc，Java；读的一侧本仓，Python——两边只能靠同一条键接头）。"""

UPLOAD_EXTENSIONS = ("jpg", "png", "webp", "gif", "bmp")
"""`{ext}` 的闭集，同上逐字副本。"""

DERIVED_KEY_TEMPLATE = "uploads/{content_sha256}/{artifact}"
"""派生物键模板：与源图同前缀（确定性派生、不铸新流水号、键里没有身份与渠道方言）。"""

GEOMETRY_ARTIFACT = "floorplan-geometry.json"
"""几何提取存档：勘测 + 几何（同 `floorplan-geometry` CLI 的存档形态）。
母版改走键的触发条件＝它进 contracts 对象键表。"""

READING_ARTIFACT = "floorplan-reading.json"
"""户型特征解析存档：勘测 / 图例 / 朝向 / 逐条判定 / 下发的特征标记（同 `floorplan-parse` CLI）。"""

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"

_UPLOAD_ORIGINAL_KEY_PATTERN = re.compile(
    r"^uploads/(?P<content_sha256>[0-9a-f]{64})/original\.(?:" + "|".join(UPLOAD_EXTENSIONS) + r")$"
)

_ENDPOINT_ENV = "ISHOME_OSS_ENDPOINT"
_BUCKET_ENV = "ISHOME_OSS_BUCKET_PRIVATE"
_ACCESS_KEY_ID_ENV = "ISHOME_OSS_ACCESS_KEY_ID"
_ACCESS_KEY_SECRET_ENV = "ISHOME_OSS_ACCESS_KEY_SECRET"


class ObjectStoreError(Exception):
    """键不合形态 / 取不到 / 写不进——响亮失败，不给一个指向空气的键。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def content_sha256_of(floorplan_object_key: str) -> str:
    """从源图键里取内容哈希；键不合约定形态即失败——派生物键由它推，键错一次就写到别人的前缀底下。"""
    matched = _UPLOAD_ORIGINAL_KEY_PATTERN.match(floorplan_object_key)
    if matched is None:
        raise ObjectStoreError(
            [f"源图键 `{floorplan_object_key}` 不合约定形态：要的是 {UPLOAD_ORIGINAL_KEY_TEMPLATE}"]
        )
    return matched.group("content_sha256")


def derived_key_of(floorplan_object_key: str, artifact: str) -> str:
    """派生物的对象键：与源图同前缀。"""
    return DERIVED_KEY_TEMPLATE.format(
        content_sha256=content_sha256_of(floorplan_object_key), artifact=artifact
    )


@dataclass(frozen=True)
class OssSettings:
    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str

    @staticmethod
    def from_env() -> OssSettings:
        """四个环境变量缺一即失败——缺凭证要在 worker 起不来的时候就知道，
        不是等第一张图来了才发现。"""
        values = {
            name: os.environ.get(name, "").strip()
            for name in (_ENDPOINT_ENV, _BUCKET_ENV, _ACCESS_KEY_ID_ENV, _ACCESS_KEY_SECRET_ENV)
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ObjectStoreError(
                [f"私有桶凭证缺 {'、'.join(missing)}（本机 `source ~/.ishome/oss-local.env`）"]
            )
        return OssSettings(
            endpoint=values[_ENDPOINT_ENV],
            bucket=values[_BUCKET_ENV],
            access_key_id=values[_ACCESS_KEY_ID_ENV],
            access_key_secret=values[_ACCESS_KEY_SECRET_ENV],
        )


class OssUploadStore:
    """阿里云 OSS 私有桶：读源图、写派生物。同步调用，activity 里经 `asyncio.to_thread` 使用。"""

    def __init__(self, settings: OssSettings) -> None:
        auth = oss2.Auth(settings.access_key_id, settings.access_key_secret)
        self._bucket = oss2.Bucket(auth, settings.endpoint, settings.bucket)
        self._bucket_name = settings.bucket

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    def get_original(self, floorplan_object_key: str) -> bytes:
        """取源图字节。键先过形态校验，再去桶里取；取不到即失败。"""
        content_sha256_of(floorplan_object_key)
        try:
            payload = self._bucket.get_object(floorplan_object_key).read()
        except oss2.exceptions.NoSuchKey as e:
            raise ObjectStoreError(
                [
                    f"私有桶 `{self._bucket_name}` 里没有 `{floorplan_object_key}`："
                    "渠道侧没落桶就派过来了"
                ]
            ) from e
        except oss2.exceptions.OssError as e:
            raise ObjectStoreError(
                [f"从私有桶 `{self._bucket_name}` 取 `{floorplan_object_key}` 失败：{e}"]
            ) from e
        if not isinstance(payload, bytes) or not payload:
            raise ObjectStoreError([f"`{floorplan_object_key}` 取回来是空的"])
        return payload

    def put_derived_json(self, floorplan_object_key: str, artifact: str, payload: bytes) -> str:
        """写一件 JSON 派生物，返回对象键。写失败即上抛。"""
        key = derived_key_of(floorplan_object_key, artifact)
        try:
            self._bucket.put_object(key, payload, headers={"Content-Type": _JSON_CONTENT_TYPE})
        except oss2.exceptions.OssError as e:
            raise ObjectStoreError(
                [f"派生物写不进私有桶 `{self._bucket_name}`（键 {key}）：{e}"]
            ) from e
        return key
