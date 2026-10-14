-- 在独立 mimiciii 数据库执行；仅查询，不创建或修改患者级表。
-- 与 work/12b_build_model_v3.sql 的患者级 seq<=3、索引及竞争事件口径对齐。
WITH dx AS (
  SELECT subject_id, hadm_id, MIN(seq_num) AS best_seq_num
  FROM public.diagnoses_icd
  WHERE icd9_code = '49121'
  GROUP BY subject_id, hadm_id
),
strict AS (
  SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.admission_location, a.hospital_expire_flag, d.best_seq_num
  FROM dx d
  JOIN public.admissions a USING (subject_id, hadm_id)
),
seq3_patients AS (
  SELECT DISTINCT subject_id FROM strict WHERE best_seq_num <= 3
),
ordered AS (
  SELECT s.*,
         LAG(s.dischtime) OVER (PARTITION BY s.subject_id ORDER BY s.admittime, s.hadm_id) AS previous_strict_dischtime
  FROM strict s
  JOIN seq3_patients p USING (subject_id)
),
v3_adm AS (
  SELECT o.*,
         (o.admission_location = 'TRANSFER FROM HOSP/EXTRAM'
           AND o.previous_strict_dischtime IS NOT NULL
           AND EXTRACT(EPOCH FROM (o.admittime - o.previous_strict_dischtime))/86400.0 <= 2) AS is_transfer_continuation
  FROM ordered o
),
next_real AS (
  SELECT a.hadm_id, nr.admittime AS next_real_admittime
  FROM v3_adm a
  LEFT JOIN LATERAL (
    SELECT b.admittime
    FROM v3_adm b
    WHERE b.subject_id = a.subject_id
      AND (b.admittime, b.hadm_id) > (a.admittime, a.hadm_id)
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime, b.hadm_id
    LIMIT 1
  ) nr ON true
),
base AS (
  SELECT a.hadm_id, a.subject_id, a.hospital_expire_flag,
         (a.dischtime > a.admittime) AS training_eligible,
         CASE WHEN n.next_real_admittime IS NOT NULL
                AND n.next_real_admittime >= a.dischtime
              THEN EXTRACT(EPOCH FROM (n.next_real_admittime - a.dischtime))/86400.0
         END AS aecopd_days,
         CASE WHEN p.dod IS NOT NULL AND p.dod::date >= a.dischtime::date
              THEN (p.dod::date - a.dischtime::date)::double precision
         END AS death_days
  FROM v3_adm a
  JOIN next_real n USING (hadm_id)
  JOIN public.patients p USING (subject_id)
),
labelled AS (
  SELECT *, CASE
    WHEN aecopd_days <= 365 AND (death_days IS NULL OR aecopd_days <= death_days) THEN 1
    WHEN death_days <= 365 AND (aecopd_days IS NULL OR death_days < aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM base
  WHERE hospital_expire_flag = 0
),
metrics AS (
  SELECT 'strict_code_patients' AS metric, COUNT(DISTINCT subject_id) AS value FROM strict
  UNION ALL SELECT 'strict_code_admissions', COUNT(*) FROM strict
  UNION ALL SELECT 'seq_le_3_patients', COUNT(*) FROM seq3_patients
  UNION ALL SELECT 'seq_le_3_code_admissions', COUNT(*) FROM strict WHERE best_seq_num <= 3
  UNION ALL SELECT 'all_strict_admissions_of_seq3_patients', COUNT(*) FROM v3_adm
  UNION ALL SELECT 'transfer_continuations_merged', COUNT(*) FROM v3_adm WHERE is_transfer_continuation
  UNION ALL SELECT 'alive_index_admissions', COUNT(*) FROM labelled
  UNION ALL SELECT 'training_eligible_index_admissions', COUNT(*) FROM labelled WHERE training_eligible
  UNION ALL SELECT 'aecopd_readmission_365d', COUNT(*) FROM labelled WHERE event_type = 1
  UNION ALL SELECT 'death_before_readmission_365d', COUNT(*) FROM labelled WHERE event_type = 2
  UNION ALL SELECT 'no_event_365d', COUNT(*) FROM labelled WHERE event_type = 0
  UNION ALL SELECT 'aecopd_readmission_90d', COUNT(*) FROM labelled WHERE event_type = 1 AND aecopd_days <= 90
  UNION ALL SELECT 'aecopd_readmission_180d', COUNT(*) FROM labelled WHERE event_type = 1 AND aecopd_days <= 180
)
SELECT metric, value FROM metrics ORDER BY metric;
