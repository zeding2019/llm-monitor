-- vllm-monitor SQLite schema. WAL + NORMAL 由 sqlite.py 在连接时设置。

CREATE TABLE IF NOT EXISTS minute_agg (
    metric_type TEXT NOT NULL,       -- 'transaction' | 'metric' | 'event'
    name        TEXT NOT NULL,
    minute_ts   INTEGER NOT NULL,    -- 秒对齐分钟
    count       INTEGER NOT NULL,
    sum         REAL    NOT NULL,
    max         REAL    NOT NULL,
    p50         REAL    NOT NULL,
    p95         REAL    NOT NULL,
    p99         REAL    NOT NULL,
    tags_json   TEXT    NOT NULL DEFAULT '{}',
    PRIMARY KEY (metric_type, name, minute_ts)
);
CREATE INDEX IF NOT EXISTS idx_minute_agg_ts ON minute_agg(minute_ts);

CREATE TABLE IF NOT EXISTS transactions (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    name         TEXT NOT NULL,
    start_ts     INTEGER NOT NULL,   -- ns
    duration_ns  INTEGER NOT NULL,
    status       TEXT NOT NULL,
    tree_json    TEXT NOT NULL,
    sampled_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_tx_start ON transactions(start_ts);
CREATE INDEX IF NOT EXISTS idx_tx_type_start ON transactions(type, start_ts);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT NOT NULL,
    name      TEXT NOT NULL,
    ts        INTEGER NOT NULL,
    status    TEXT NOT NULL,
    data_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ev_ts ON events(ts);

CREATE TABLE IF NOT EXISTS heartbeat (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    values_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hb_ts ON heartbeat(ts);
CREATE INDEX IF NOT EXISTS idx_hb_source_ts ON heartbeat(source, ts);

-- 单请求生命周期(按 SGLang rid 聚合一行)。跨进程:各 Scheduler 进程各写各的 rid,
-- rid 作 PK + INSERT OR REPLACE,天然不互相覆盖。
CREATE TABLE IF NOT EXISTS request_trace (
    rid            TEXT PRIMARY KEY,
    arrival_ts     INTEGER NOT NULL,   -- 首次进 prefill 的 wall ns
    first_token_ts INTEGER,            -- 首个 decode 产出 ns(算 TTFT)
    finish_ts      INTEGER,            -- 完成 ns
    prefill_ms     REAL DEFAULT 0,
    decode_ms      REAL DEFAULT 0,
    decode_steps   INTEGER DEFAULT 0,
    prompt_tokens  INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    ttft_ms        REAL DEFAULT 0,
    tpot_ms        REAL DEFAULT 0,
    e2e_ms         REAL DEFAULT 0,
    status         TEXT DEFAULT '0',
    pid            INTEGER DEFAULT 0,
    tags_json      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rt_arrival ON request_trace(arrival_ts);
CREATE INDEX IF NOT EXISTS idx_rt_finish ON request_trace(finish_ts);

-- Scheduler 逐 batch 执行记录(引擎事务明细)。每进程由 DbWriter 追加。
CREATE TABLE IF NOT EXISTS scheduler_step (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,     -- wall ns
    pid         INTEGER NOT NULL,
    mode        TEXT NOT NULL,        -- prefill / extend / decode / ...
    batch_reqs  INTEGER DEFAULT 0,
    batch_tokens INTEGER DEFAULT 0,
    dur_ms      REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ss_ts ON scheduler_step(ts);
CREATE INDEX IF NOT EXISTS idx_ss_pid_ts ON scheduler_step(pid, ts);
