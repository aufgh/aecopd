-- =====================================================================
-- 诊断脚本 D1：随访窗口 / 信息性删失（复核问题 1）
-- 目的：量化"365 天内无事件即判为无事件"这一合同受随访长度影响的程度。
-- 只读：仅 SELECT，不 CREATE/DROP/UPDATE，不写库，不碰测试集选择。
-- 运行：psql -h localhost -U postgres -d <db> -v ON_ERROR_STOP=1 -f d1_followup_window_diagnosis.sql
--   （切到 Claude Code 完全访问模式后可由 Claude 直接执行）
-- 前置表：project_aecopd.master_cohort（含 observed_allcause_span_after_discharge_days、
--   days_to_next_observed_aecopd、dod、dischtime）、project_aecopd.model_v1_cohort（data_split、
--   training_eligible）、project_aecopd.model_v1_labels（event_type）。
-- 说明：observed_allcause_span_after_discharge_days = 患者最后一次全因出院距本次出院天数，
--   是"数据库里这次出院之后还能观察多久"的近似上界；它不是经过验证的删失时间，但
--   足以暴露"看不到 = 判为无事件"的偏倚方向。
-- 注意：CTE(WITH) 只作用于紧随其后的单条语句，故每段各自内联一份相同的 cohort 定义。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) 整体：被判为 event_type=0（365 天内无事件）者的随访窗口分布 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '1_censored_obs_span_distribution' AS section,
  data_split,
  count(*)                                              AS censored_n,
  count(*) FILTER (WHERE obs_span < 30)                 AS span_lt_30d,
  count(*) FILTER (WHERE obs_span < 90)                 AS span_lt_90d,
  count(*) FILTER (WHERE obs_span < 365)                AS span_lt_365d,
  round(100.0*count(*) FILTER (WHERE obs_span < 365)/nullif(count(*),0),2) AS pct_span_lt_365d,
  round(percentile_cont(0.25) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_p25,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_median,
  round(percentile_cont(0.75) WITHIN GROUP (ORDER BY obs_span)::numeric,1) AS obs_span_p75
FROM cohort
WHERE event_type = 0
GROUP BY data_split
ORDER BY data_split;

-- -------- 2) 关键偏倚指标：判为"无事件"但随访不足对应视界的比例 --------
-- 这些是"其实没看满、却被当成阴性"的住院；比例越高，标签越不可信。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '2_insufficient_followup_by_horizon' AS section,
  data_split,
  count(*) FILTER (WHERE event_type=0)                                    AS censored_n,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 90)                  AS censored_but_span_lt_90d,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 365)                 AS censored_but_span_lt_365d,
  round(100.0*count(*) FILTER (WHERE event_type=0 AND obs_span < 365)
        /nullif(count(*) FILTER (WHERE event_type=0),0),2)               AS pct_insufficient_365d
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 3) 开放尾部（第 365 天仍无事件者）的随访窗口分布 --------
-- 这正是尾部 none_observed 最可能与"入院年份晚、可观察短"混杂的地方。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '3_open_tail_obs_span' AS section,
  data_split,
  count(*) FILTER (WHERE event_type=0)                        AS tail_risk_n,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 365)     AS tail_span_lt_365d,
  count(*) FILTER (WHERE event_type=0 AND obs_span < 730)     AS tail_span_lt_730d,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY obs_span)
        FILTER (WHERE event_type=0)::numeric,1)               AS tail_obs_span_median
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 4) 死亡确认口径：dod 覆盖与出院后死亡可见性 --------
-- 若 dod 只覆盖到某历法边界，晚期死亡会被系统性漏记，进一步放大问题 1。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split,
    m.days_to_next_observed_aecopd AS aecopd_gap,
    m.observed_allcause_span_after_discharge_days AS obs_span,
    l.event_type,
    CASE WHEN m.dod IS NOT NULL AND m.dod >= m.dischtime::date
         THEN (m.dod - m.dischtime::date)::double precision END AS death_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '4_death_ascertainment' AS section,
  data_split,
  count(*)                                                     AS n,
  count(*) FILTER (WHERE death_gap IS NOT NULL)                AS has_any_death_date,
  count(*) FILTER (WHERE death_gap IS NOT NULL AND death_gap<=365) AS death_within_365d,
  count(*) FILTER (WHERE death_gap IS NOT NULL AND death_gap>365)  AS death_after_365d
FROM cohort
GROUP BY data_split
ORDER BY data_split;
