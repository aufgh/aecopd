\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.discharge_note_admission_audit;
CREATE TABLE project_aecopd.discharge_note_admission_audit AS
WITH cohort_notes AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.dischtime,
    n.note_id,
    n.note_seq,
    n.charttime,
    n.storetime,
    LENGTH(n.text) AS text_chars
  FROM project_aecopd.aecopd_admissions_strict c
  LEFT JOIN mimiciv_note.discharge n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
), aggregated AS (
  SELECT
    subject_id,
    hadm_id,
    dischtime,
    COUNT(note_id) AS note_count,
    MIN(note_seq) FILTER (WHERE note_id IS NOT NULL) AS min_note_seq,
    MAX(note_seq) FILTER (WHERE note_id IS NOT NULL) AS max_note_seq,
    MIN(charttime) AS first_charttime,
    MAX(charttime) AS last_charttime,
    MIN(storetime) AS first_storetime,
    MAX(storetime) AS last_storetime,
    MIN(text_chars) FILTER (WHERE note_id IS NOT NULL) AS min_text_chars,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY text_chars) FILTER (WHERE note_id IS NOT NULL) AS median_text_chars,
    MAX(text_chars) FILTER (WHERE note_id IS NOT NULL) AS max_text_chars,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND COALESCE(text_chars,0)=0) AS empty_text_count,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime<=dischtime AND storetime<=dischtime) AS notes_available_by_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime<=dischtime AND storetime>dischtime) AS notes_charted_by_but_stored_after_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND charttime>dischtime) AS notes_charted_after_discharge,
    COUNT(*) FILTER (WHERE note_id IS NOT NULL AND (charttime IS NULL OR storetime IS NULL)) AS notes_missing_time
  FROM cohort_notes
  GROUP BY subject_id,hadm_id,dischtime
)
SELECT
  *,
  (note_count>0) AS has_discharge_note,
  (notes_available_by_discharge>0) AS has_note_available_by_discharge,
  CASE
    WHEN note_count=0 THEN 'NO_NOTE'
    WHEN notes_available_by_discharge>0 THEN 'AVAILABLE_BY_DISCHARGE'
    WHEN notes_missing_time=note_count THEN 'MISSING_TIME'
    WHEN notes_charted_after_discharge>0 THEN 'CHARTED_AFTER_DISCHARGE'
    WHEN notes_charted_by_but_stored_after_discharge>0 THEN 'CHARTED_BY_BUT_STORED_AFTER_DISCHARGE'
    ELSE 'OTHER_TIME_PATTERN'
  END AS note_time_flag
FROM aggregated;

ALTER TABLE project_aecopd.discharge_note_admission_audit ADD CONSTRAINT discharge_note_admission_audit_pk PRIMARY KEY (hadm_id);

DROP TABLE IF EXISTS project_aecopd.discharge_note_coverage_summary;
CREATE TABLE project_aecopd.discharge_note_coverage_summary AS
SELECT
  COUNT(*) AS aecopd_admissions,
  COUNT(*) FILTER (WHERE has_discharge_note) AS with_any_discharge_note,
  ROUND(100.0*COUNT(*) FILTER (WHERE has_discharge_note)/NULLIF(COUNT(*),0),2) AS any_note_coverage_pct,
  COUNT(*) FILTER (WHERE has_note_available_by_discharge) AS with_note_available_by_discharge,
  ROUND(100.0*COUNT(*) FILTER (WHERE has_note_available_by_discharge)/NULLIF(COUNT(*),0),2) AS available_by_discharge_coverage_pct,
  COUNT(*) FILTER (WHERE note_count>1) AS admissions_with_multiple_notes,
  SUM(note_count) AS discharge_notes,
  SUM(empty_text_count) AS empty_text_notes,
  ROUND(100.0*SUM(empty_text_count)/NULLIF(SUM(note_count),0),4) AS empty_text_pct
FROM project_aecopd.discharge_note_admission_audit;

DROP TABLE IF EXISTS project_aecopd.discharge_note_time_distribution;
CREATE TABLE project_aecopd.discharge_note_time_distribution AS
SELECT note_time_flag,COUNT(*) AS admissions,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM project_aecopd.discharge_note_admission_audit),2) AS pct
FROM project_aecopd.discharge_note_admission_audit
GROUP BY note_time_flag
ORDER BY admissions DESC;

DROP TABLE IF EXISTS project_aecopd.discharge_note_seq_distribution;
CREATE TABLE project_aecopd.discharge_note_seq_distribution AS
SELECT n.note_seq,COUNT(*) AS notes,COUNT(DISTINCT n.hadm_id) AS admissions
FROM mimiciv_note.discharge n
JOIN project_aecopd.aecopd_admissions_strict c ON c.subject_id=n.subject_id AND c.hadm_id=n.hadm_id
GROUP BY n.note_seq
ORDER BY n.note_seq;

DROP TABLE IF EXISTS project_aecopd.discharge_note_timing_summary;
CREATE TABLE project_aecopd.discharge_note_timing_summary AS
WITH cohort_notes AS (
  SELECT c.dischtime,n.charttime,n.storetime
  FROM project_aecopd.aecopd_admissions_strict c
  JOIN mimiciv_note.discharge n ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
)
SELECT
  COUNT(*) AS notes,
  COUNT(*) FILTER (WHERE charttime<=dischtime) AS charted_on_or_before_discharge,
  COUNT(*) FILTER (WHERE storetime<=dischtime) AS stored_on_or_before_discharge,
  COUNT(*) FILTER (WHERE charttime<=dischtime AND storetime<=dischtime) AS both_on_or_before_discharge,
  COUNT(*) FILTER (WHERE charttime IS NULL) AS missing_charttime,
  COUNT(*) FILTER (WHERE storetime IS NULL) AS missing_storetime,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (charttime-dischtime))/86400.0) AS chart_offset_p75_days,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_p25_days,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_median_days,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (storetime-dischtime))/86400.0) AS store_offset_p75_days
FROM cohort_notes;

DROP TABLE IF EXISTS project_aecopd.discharge_note_missingness_summary;
CREATE TABLE project_aecopd.discharge_note_missingness_summary AS
WITH note_subjects AS (
  SELECT DISTINCT n.subject_id
  FROM mimiciv_note.discharge n
  JOIN (SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict) c USING (subject_id)
)
SELECT
  COUNT(*) FILTER (WHERE a.has_discharge_note) AS matching_hadm_has_note,
  COUNT(*) FILTER (WHERE NOT a.has_discharge_note AND s.subject_id IS NOT NULL) AS no_matching_hadm_but_subject_has_other_note,
  COUNT(*) FILTER (WHERE NOT a.has_discharge_note AND s.subject_id IS NULL) AS subject_has_no_discharge_note
FROM project_aecopd.discharge_note_admission_audit a
LEFT JOIN note_subjects s USING (subject_id);

DROP TABLE IF EXISTS project_aecopd.discharge_note_text_summary;
CREATE TABLE project_aecopd.discharge_note_text_summary AS
SELECT
  COUNT(*) FILTER (WHERE has_discharge_note) AS admissions_with_note,
  MIN(min_text_chars) AS min_text_chars,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_p25,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_median,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY median_text_chars) FILTER (WHERE has_discharge_note) AS text_chars_p75,
  MAX(max_text_chars) AS max_text_chars
FROM project_aecopd.discharge_note_admission_audit;

COMMIT;

\copy (SELECT * FROM project_aecopd.discharge_note_coverage_summary) TO 'outputs/04_note_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_admission_audit ORDER BY subject_id,hadm_id) TO 'outputs/04_note_admission_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_time_distribution) TO 'outputs/04_note_time_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_seq_distribution) TO 'outputs/04_note_seq_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_timing_summary) TO 'outputs/04_note_timing_summary.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_missingness_summary) TO 'outputs/04_note_missingness_summary.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.discharge_note_text_summary) TO 'outputs/04_note_text_summary.csv' CSV HEADER



