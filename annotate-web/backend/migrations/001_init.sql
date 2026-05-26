CREATE TABLE IF NOT EXISTS submissions (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  ip_hash      TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'base',
  metadata     TEXT NOT NULL,
  storage_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS submissions_created_at_idx ON submissions(created_at DESC);
CREATE INDEX IF NOT EXISTS submissions_ip_hash_idx ON submissions(ip_hash);
