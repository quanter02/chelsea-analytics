-- LLM 판단 신뢰도 DB
-- 원칙: 예측은 절대 수정하지 않고 쌓기만 한다. 정답은 나중에 따로 붙인다.
--       확신 오답·기권은 테이블이 아니라 뷰로 뽑는다 (분모 보존, 늦게 확정되는 정답 대응).

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY,
    model_id       TEXT    NOT NULL,
    model_version  TEXT    NOT NULL,          -- 버전이 바뀌면 다른 측정 도구로 취급
    prompt_version TEXT    NOT NULL,
    task_type      TEXT    NOT NULL,          -- 예: change_cause
    input_hash     TEXT    NOT NULL,          -- 같은 입력에 여러 모델 → 오류 상관 계산
    answer         TEXT    NOT NULL,          -- 'R' 실제 변화 | 'M' 측정 변화 | 'N' 변화 없음 | 'ABSTAIN'
    confidence     REAL,                      -- 모델이 말한 확신도 (기권이면 NULL)
    evidence       TEXT,                      -- 인용 근거 (JSON)
    evidence_ok    INTEGER,                   -- 근거를 로그와 기계적으로 대조한 결과 (1/0, 기권이면 NULL)
    cost_usd       REAL    NOT NULL,
    latency_ms     INTEGER,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_input ON predictions (input_hash);
CREATE INDEX IF NOT EXISTS ix_pred_model ON predictions (model_id, model_version, task_type);

CREATE TABLE IF NOT EXISTS outcomes (
    input_hash   TEXT PRIMARY KEY,
    task_type    TEXT NOT NULL,
    true_label   TEXT NOT NULL,
    label_source TEXT NOT NULL,               -- deploy_log | human_review | later_data
    labeled_at   TEXT NOT NULL
);

-- 틀렸을 때의 피해 (USD 환산). answer = 'ABSTAIN' 행은 사람 검토로 넘기는 비용.
CREATE TABLE IF NOT EXISTS loss_matrix (
    task_type  TEXT NOT NULL,
    true_label TEXT NOT NULL,
    answer     TEXT NOT NULL,
    loss_usd   REAL NOT NULL,
    PRIMARY KEY (task_type, true_label, answer)
);

-- 정답이 붙은 예측에 결과 범주를 매긴 뷰
CREATE VIEW IF NOT EXISTS scored AS
SELECT p.*,
       o.true_label,
       o.labeled_at,
       CASE WHEN p.answer = 'ABSTAIN'      THEN 'abstain'
            WHEN p.answer = o.true_label   THEN 'correct'
            ELSE 'confident_error' END AS outcome,
       l.loss_usd
FROM predictions p
JOIN outcomes o    ON o.input_hash = p.input_hash
LEFT JOIN loss_matrix l ON l.task_type = p.task_type AND l.true_label = o.true_label AND l.answer = p.answer;

-- 요청하신 두 "테이블": 확신 오답과 기권
CREATE VIEW IF NOT EXISTS confident_errors AS
SELECT * FROM scored WHERE outcome = 'confident_error';

CREATE VIEW IF NOT EXISTS abstentions AS      -- 정답이 아직 없는 기권도 포함 (LEFT JOIN)
SELECT p.*, o.true_label
FROM predictions p LEFT JOIN outcomes o ON o.input_hash = p.input_hash
WHERE p.answer = 'ABSTAIN';

-- 정답 대기 중인 예측 (비율 계산에서 빠지는 몫을 투명하게)
CREATE VIEW IF NOT EXISTS pending AS
SELECT p.* FROM predictions p LEFT JOIN outcomes o ON o.input_hash = p.input_hash
WHERE o.input_hash IS NULL;
