\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.aecopd_longitudinal_history;
CREATE TABLE project_aecopd.aecopd_longitudinal_history AS
WITH cohort_subjects AS (
  SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict
), all_admission_history AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime,a.hadm_id)-1 AS prior_all_admission_count
  FROM mimiciv_hosp.admissions a
  JOIN cohort_subjects c USING (subject_id)
), strict_ordered AS (
  SELECT
    s.*,
    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS aecopd_order,
    COUNT(*) OVER (PARTITION BY subject_id) AS aecopd_total,
    LAG(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS prev_aecopd_hadm_id,
    LAG(dischtime) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS prev_aecopd_dischtime,
    LEAD(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS next_observed_aecopd_hadm_id,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) AS next_observed_aecopd_admittime
  FROM project_aecopd.aecopd_admissions_strict s
), gaps AS (
  SELECT
    s.*,
    s.aecopd_order-1 AS prior_aecopd_count,
    a.prior_all_admission_count,
    EXTRACT(EPOCH FROM (s.admittime-s.prev_aecopd_dischtime))/86400.0 AS days_since_previous_aecopd,
    EXTRACT(EPOCH FROM (s.next_observed_aecopd_admittime-s.dischtime))/86400.0 AS days_to_next_observed_aecopd
  FROM strict_ordered s
  JOIN all_admission_history a USING (subject_id,hadm_id)
)
SELECT
  *,
  (aecopd_order=1) AS is_first_observed_aecopd,
  (aecopd_order=aecopd_total) AS is_last_observed_aecopd,
  (days_to_next_observed_aecopd <= 0) AS qc_nonpositive_next_gap,
  (days_to_next_observed_aecopd > 0 AND days_to_next_observed_aecopd <= 7) AS qc_short_gap,
  (days_to_next_observed_aecopd < 0 OR qc_overlap_previous_strict) AS qc_overlap,
  qc_transfer_from_hospital AS qc_transfer
FROM gaps;

ALTER TABLE project_aecopd.aecopd_longitudinal_history ADD CONSTRAINT aecopd_longitudinal_history_pk PRIMARY KEY (hadm_id);
CREATE INDEX aecopd_longitudinal_history_subject_time_idx ON project_aecopd.aecopd_longitudinal_history(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.recurrence_distribution;
CREATE TABLE project_aecopd.recurrence_distribution AS
WITH per_patient AS (
  SELECT subject_id,MAX(aecopd_total) AS aecopd_total
  FROM project_aecopd.aecopd_longitudinal_history
  GROUP BY subject_id
)
SELECT
  CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END AS aecopd_admissions_category,
  COUNT(*) AS patients
FROM per_patient
GROUP BY 1
ORDER BY CASE WHEN CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END='5+' THEN 5 ELSE (CASE WHEN aecopd_total>=5 THEN '5+' ELSE aecopd_total::text END)::int END;

DROP TABLE IF EXISTS project_aecopd.gap_summary;
CREATE TABLE project_aecopd.gap_summary AS
SELECT
  COUNT(*) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS observed_adjacent_gaps,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY days_to_next_observed_aecopd) FILTER (WHERE next_observed_aecopd_hadm_id IS NOT NULL) AS gap_p75_days,
  COUNT(*) FILTER (WHERE qc_nonpositive_next_gap) AS nonpositive_gaps,
  COUNT(*) FILTER (WHERE qc_short_gap) AS short_positive_gaps_le_7d,
  COUNT(*) FILTER (WHERE qc_overlap) AS overlap_flag_rows,
  COUNT(*) FILTER (WHERE qc_transfer) AS transfer_from_hospital_rows,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=2) AS landmarks_with_at_least_2_prior_aecopd,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=3) AS landmarks_with_at_least_3_prior_aecopd,
  COUNT(*) FILTER (WHERE prior_aecopd_count>=4) AS landmarks_with_at_least_4_prior_aecopd,
  COUNT(*) FILTER (WHERE is_first_observed_aecopd) AS first_observed_rows,
  COUNT(*) FILTER (WHERE is_last_observed_aecopd) AS last_observed_rows
FROM project_aecopd.aecopd_longitudinal_history;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_longitudinal_history ORDER BY subject_id,admittime,hadm_id) TO 'outputs/03_longitudinal_history.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.recurrence_distribution) TO 'outputs/03_recurrence_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.gap_summary) TO 'outputs/03_gap_summary.csv' CSV HEADER



