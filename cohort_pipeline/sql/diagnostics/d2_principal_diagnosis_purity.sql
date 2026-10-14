-- =====================================================================
-- 诊断脚本 D2：表型纯度 / 主诊断（复核问题 2）
-- 目的：strict_any 表型只要求诊断列表任意位置出现 J441/49121。用已存在的
--   best_seq_num（该住院 AECOPD 码的最小诊断序号）量化 AECOPD 究竟多常是主诊断，
--   以及"如果改为主诊断入组"队列会缩小多少。这决定入组与"再入院"结局的噪声大小。
-- 只读：仅 SELECT。运行方式同 D1。
-- 前置表：project_aecopd.aecopd_admissions_strict(best_seq_num) 或
--   project_aecopd.master_cohort(best_seq_num)。此处用 master_cohort 以对齐建模口径。
-- 注意：MIMIC-IV diagnoses_icd 中 seq_num=1 通常为主要诊断（principal diagnosis）。
--   CTE(WITH) 只作用于紧随其后的单条语句，故每段各自内联一份相同的 cohort 定义。
-- =====================================================================
\pset pager off
SET client_encoding = 'UTF8';

-- -------- 1) AECOPD 诊断序号分布：多少是主诊断(=1)、前三位、更靠后 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '1_best_seq_num_distribution' AS section,
  data_split,
  count(*)                                          AS n,
  count(*) FILTER (WHERE best_seq_num = 1)          AS principal_dx_n,
  round(100.0*count(*) FILTER (WHERE best_seq_num = 1)/nullif(count(*),0),2) AS pct_principal,
  count(*) FILTER (WHERE best_seq_num BETWEEN 1 AND 3) AS top3_n,
  round(100.0*count(*) FILTER (WHERE best_seq_num BETWEEN 1 AND 3)/nullif(count(*),0),2) AS pct_top3,
  count(*) FILTER (WHERE best_seq_num > 5)          AS beyond_5_n,
  max(best_seq_num)                                 AS max_seq
FROM cohort
GROUP BY data_split
ORDER BY data_split;

-- -------- 2) 敏感性：若把入组收紧为"AECOPD 为主诊断(seq=1)"，队列规模变化 --------
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '2_cohort_shrink_if_principal_only' AS section,
  count(*)                                          AS strict_any_admissions,
  count(*) FILTER (WHERE best_seq_num = 1)          AS principal_only_admissions,
  count(DISTINCT subject_id)                        AS strict_any_patients,
  count(DISTINCT subject_id) FILTER (WHERE best_seq_num = 1) AS principal_only_patients
FROM cohort;

-- -------- 3) 表型纯度是否与结局相关（潜在偏倚检查）--------
-- 若主诊断组与非主诊断组的观察事件率差异明显，说明 strict_any 引入了结局层面的噪声。
WITH cohort AS (
  SELECT
    m.subject_id, m.hadm_id, c.data_split, m.best_seq_num,
    l.event_type,
    m.days_to_next_observed_aecopd AS aecopd_gap
  FROM project_aecopd.master_cohort m
  JOIN project_aecopd.model_v1_cohort c USING (subject_id, hadm_id)
  JOIN project_aecopd.model_v1_labels l USING (hadm_id)
  WHERE c.training_eligible
)
SELECT '3_event_rate_by_purity' AS section,
  CASE WHEN best_seq_num = 1 THEN 'principal_dx' ELSE 'secondary_dx' END AS phenotype_group,
  count(*)                                                     AS n,
  count(*) FILTER (WHERE event_type = 1)                       AS aecopd_event_n,
  round(100.0*count(*) FILTER (WHERE event_type = 1)/nullif(count(*),0),2) AS pct_aecopd_event,
  count(*) FILTER (WHERE event_type = 2)                       AS death_event_n,
  round(100.0*count(*) FILTER (WHERE event_type = 2)/nullif(count(*),0),2) AS pct_death_event
FROM cohort
GROUP BY 2
ORDER BY 2;
