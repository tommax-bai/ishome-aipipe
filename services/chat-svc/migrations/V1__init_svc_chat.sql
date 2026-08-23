-- V1__init_svc_chat.sql —— svc_chat 首批表（对齐文档 §5.1）。
--
-- 执行契约：本文件表名不带 schema 前缀；chat-migrate 先 CREATE SCHEMA 并
-- SET search_path 再执行本文件——同一组迁移可原样应用到测试 schema
-- （如 svc_chat_test，集成测试用，跑完即删）。
--
-- 纪律（技术架构 6.4 继承）：主键 ULID（text，26 字符，应用侧生成）、时间一律
-- UTC timestamptz、软删（deleted_at，不物理删）、金额无涉。
--
-- 红线：槽位真相唯一在 svc_project.slots（project-svc 属主）——本 schema 永不建
-- 槽位/项目真相表；会话态（阶段/情绪轨迹/槽位缓存）归 Redis，不落表；
-- 禁止跨 schema join。
--
-- 【本批不建 episodic_memories】情景记忆表需要 pgvector（embedding 向量列 +
-- 向量索引），本地 postgres:16 官方镜像无该扩展，且该 PG 实例与其他服务共享、
-- 不重建容器。待基础镜像切换到带 pgvector 的镜像（如 pgvector/pgvector:pg16）
-- 后，在 V2__episodic_memories.sql 建表（结构化摘要列 + embedding，
-- 会话结束触发摘要写入；不用原始聊天记录做 RAG）。

-- 会话：渠道三元组定位一个会话线程（v1 一人一会话一项目）。
-- TODO(identity)：identity 归一后补渠道无关 user_id 回填与键控迁移（对齐 §6.5）。
CREATE TABLE conversations (
    id               text        PRIMARY KEY,  -- ULID
    channel_type     integer     NOT NULL,     -- contracts ChannelType 枚举数值
    channel_instance text        NOT NULL,     -- 渠道实例（如 mock:local / feishu:xxx）
    external_user_id text        NOT NULL,     -- 渠道侧用户 id
    user_id          text,                     -- 归一 user_id（identity 就绪前为空）
    project_id       text,                     -- 项目引用（真相在 svc_project，仅存 id）
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    deleted_at       timestamptz,              -- 软删；唯一键含软删行（历史线程不复活）
    CONSTRAINT conversations_channel_key UNIQUE (channel_type, channel_instance, external_user_id)
);

-- 消息原文：入站/出站都存；幂等键防重存（渠道重投/回话重试不产生重复行）。
CREATE TABLE messages (
    id                  text        PRIMARY KEY,  -- ULID（本服务生成）
    conversation_id     text        NOT NULL REFERENCES conversations (id),
    direction           text        NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    content_type        text        NOT NULL,     -- text/quick_reply/image/audio/card/unknown
    text                text        NOT NULL DEFAULT '',  -- 原文文本（非文本内容存归一化占位）
    external_message_id text        NOT NULL,     -- UnifiedMessage.message_id
    idempotency_key     text        NOT NULL,     -- 入站=渠道消息 id；出站=reply-{入站id}-{seq}
    occurred_at         timestamptz,              -- 渠道侧发生时间（可空）
    created_at          timestamptz NOT NULL DEFAULT now(),
    deleted_at          timestamptz,
    CONSTRAINT messages_idempotency_key UNIQUE (conversation_id, idempotency_key)
);

CREATE INDEX messages_conversation_order ON messages (conversation_id, created_at);

-- 画像五区（对齐 §5.1）：事实/偏好/承诺/沟通风格/情绪模式，各一 JSONB 区。
-- 存模式不存状态——例：情绪模式区存"进度延迟易焦虑"这类稳定模式，不存当下情绪
-- （当下情绪属会话态，归 Redis）。只存跨项目个人特征，项目档案归 svc_project。
CREATE TABLE user_profiles (
    id                  text        PRIMARY KEY,  -- ULID
    user_id             text        NOT NULL,     -- 归一 user_id；identity 就绪前存会话键
    facts               jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- 事实区
    preferences         jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- 偏好区
    commitments         jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- 承诺区（画像沉淀；运行态见 commitments 表）
    communication_style jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- 沟通风格区
    emotion_patterns    jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- 情绪模式区
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    deleted_at          timestamptz,
    CONSTRAINT user_profiles_user_key UNIQUE (user_id)
);

-- 承诺区（对齐 §5.1）：承诺白名单=自动化能力清单，兑现率恒 100%。
-- 挂任务承诺以任务事件核销（task_id 关联），不自扫 deadline——到期真相在任务层。
CREATE TABLE commitments (
    id                 text        PRIMARY KEY,  -- ULID
    user_id            text        NOT NULL,     -- 归一 user_id；identity 就绪前存会话键
    conversation_id    text        REFERENCES conversations (id),
    kind               text        NOT NULL,     -- 承诺类型（白名单键）
    content            text        NOT NULL,     -- 承诺内容原文
    deadline           timestamptz,              -- 承诺期限（可空；不自扫）
    fulfillment_action text,                     -- 履约动作（自动化能力清单项）
    status             text        NOT NULL DEFAULT 'open'
                       CHECK (status IN ('open', 'fulfilled', 'canceled')),
    task_id            text,                     -- 关联任务 id（任务事件核销）
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    deleted_at         timestamptz
);

CREATE INDEX commitments_user_status ON commitments (user_id, status);
