CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    query TEXT NOT NULL,
    all_results JSONB NOT NULL,
    selected_result TEXT NOT NULL,
    similarities JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session_id ON search_logs(session_id);
CREATE INDEX idx_timestamp ON search_logs(timestamp);