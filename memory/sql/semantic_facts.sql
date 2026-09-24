CREATE TABLE IF NOT EXISTS semantic_facts (
    fact_id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    fact_key TEXT,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    version INTEGER NOT NULL DEFAULT 1,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1,
    category TEXT,
    supersedes_id TEXT,
    superseded_by TEXT,
    source_episode_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT 'promotion'
);

CREATE UNIQUE INDEX IF NOT EXISTS semantic_facts_one_current
    ON semantic_facts (owner_type, owner_id, fact_key)
    WHERE status = 'current'
      AND fact_key IS NOT NULL
      AND fact_key <> '';
