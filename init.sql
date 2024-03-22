CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_stat_statements;

CREATE TABLE IF NOT EXISTS embeddings (
  id SERIAL PRIMARY KEY,
  embedding vector,
  text text,
  created_at timestamptz DEFAULT now()
);

ALTER system SET shared_preload_libraries='pg_stat_statements';
