\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.master_cohort;
CREATE TABLE project_aecopd.master_cohort AS
WITH allcause_observation AS (
  SELECT a.subject_id,MAX(a.dischtime) AS last_observed_allcause_dischtime
  FROM mimiciv_hosp.admissions a
  JOIN (SELECT DISTINCT subject_id FROM project_aecopd.aecopd_admissions_strict) c USING(subject_id)
  GROUP BY a.subject_id
), base AS (
  SELECT
    h.*,
    p.gender,p.anchor_age,p.anchor_year,p.anchor_year_group,p.dod,
    n.has_discharge_note,n.note_count,n.has_note_available_by_discharge,n.note_time_flag,
    n.first_charttime AS discharge_note_first_charttime,
    n.first_storetime AS discharge_note_first_storetime,
    n.median_text_chars AS discharge_note_text_chars,
    i.has_icu_stay,i.icu_stay_count,i.first_icu_intime,i.last_icu_outtime,i.total_icu_los_days,i.icu_stays_outside_hospital_time,
    o.last_observed_allcause_dischtime,
    EXTRACT(EPOCH FROM (o.last_observed_allcause_dischtime-h.dischtime))/86400.0 AS observed_allcause_span_after_discharge_days
  FROM project_aecopd.aecopd_longitudinal_history h
  JOIN mimiciv_hosp.patients p USING(subject_id)
  JOIN project_aecopd.discharge_note_admission_audit n USING(subject_id,hadm_id)
  JOIN project_aecopd.icu_admission_audit i USING(subject_id,hadm_id)
  JOIN allcause_observation o USING(subject_id)
)
SELECT
  *,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=90) AS next_observed_aecopd_within_90d,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=180) AS next_observed_aecopd_within_180d,
  (days_to_next_observed_aecopd>0 AND days_to_next_observed_aecopd<=365) AS next_observed_aecopd_within_365d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=90 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=90 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_90d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=180 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=180 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_180d,
  CASE
    WHEN days_to_next_observed_aecopd<=0 THEN 'NONPOSITIVE_NEXT_EVENT_GAP_REVIEW'
    WHEN days_to_next_observed_aecopd<=365 THEN 'OBSERVED_AECOPD_WITHIN_WINDOW'
    WHEN observed_allcause_span_after_discharge_days>=365 THEN 'NO_AECOPD_OBSERVED_WITHIN_WINDOW_ALLCAUSE_SPAN_GE_WINDOW'
    ELSE 'NO_AECOPD_OBSERVED_INSUFFICIENT_ALLCAUSE_SPAN'
  END AS candidate_followup_status_365d
FROM base;

ALTER TABLE project_aecopd.master_cohort ADD CONSTRAINT master_cohort_pk PRIMARY KEY(hadm_id);
CREATE INDEX master_cohort_subject_time_idx ON project_aecopd.master_cohort(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.master_cohort_qc;
CREATE TABLE project_aecopd.master_cohort_qc AS
SELECT 'rows'::text metric,COUNT(*)::text value FROM project_aecopd.master_cohort
UNION ALL SELECT 'distinct_hadm_id',COUNT(DISTINCT hadm_id)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'distinct_subject_id',COUNT(DISTINCT subject_id)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'duplicate_hadm_id_groups',COUNT(*)::text FROM (SELECT hadm_id FROM project_aecopd.master_cohort GROUP BY hadm_id HAVING COUNT(*)>1) q
UNION ALL SELECT 'missing_subject_id',COUNT(*) FILTER(WHERE subject_id IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_hadm_id',COUNT(*) FILTER(WHERE hadm_id IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_admittime',COUNT(*) FILTER(WHERE admittime IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'missing_dischtime',COUNT(*) FILTER(WHERE dischtime IS NULL)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'nonpositive_los',COUNT(*) FILTER(WHERE qc_nonpositive_los)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_any_discharge_note',COUNT(*) FILTER(WHERE has_discharge_note)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_note_available_by_discharge',COUNT(*) FILTER(WHERE has_note_available_by_discharge)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'with_icu_stay',COUNT(*) FILTER(WHERE has_icu_stay)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_90d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_90d)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_180d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_180d)::text FROM project_aecopd.master_cohort
UNION ALL SELECT 'observed_next_aecopd_within_365d',COUNT(*) FILTER(WHERE next_observed_aecopd_within_365d)::text FROM project_aecopd.master_cohort;

DROP TABLE IF EXISTS project_aecopd.master_cohort_data_dictionary;
CREATE TABLE project_aecopd.master_cohort_data_dictionary AS
SELECT
  c.ordinal_position,
  c.column_name,
  c.data_type,
  CASE
    WHEN c.column_name IN('subject_id','hadm_id') THEN 'identifier'
    WHEN c.column_name LIKE 'qc_%' THEN 'quality_control'
    WHEN c.column_name LIKE '%note%' THEN 'note_availability'
    WHEN c.column_name LIKE '%icu%' THEN 'icu_availability'
    WHEN c.column_name LIKE '%aecopd%' OR c.column_name IN('aecopd_order','aecopd_total') THEN 'aecopd_history_or_observed_outcome'
    WHEN c.column_name LIKE 'candidate_followup_status_%' OR c.column_name LIKE 'observed_allcause%' OR c.column_name='last_observed_allcause_dischtime' THEN 'candidate_followup_support'
    WHEN c.column_name IN('gender','anchor_age','anchor_year','anchor_year_group','dod','deathtime','hospital_expire_flag') THEN 'demographic_or_death_support'
    ELSE 'admission_context'
  END AS field_group,
  CASE c.column_name
    WHEN 'subject_id' THEN 'MIMIC patient identifier'
    WHEN 'hadm_id' THEN 'strict AECOPD hospital admission identifier; one row per hadm_id'
    WHEN 'best_seq_num' THEN 'lowest diagnosis sequence number among strict AECOPD codes'
    WHEN 'aecopd_codes' THEN 'strict AECOPD ICD codes observed for this admission'
    WHEN 'prior_aecopd_count' THEN 'number of earlier system-observed strict AECOPD admissions'
    WHEN 'prior_all_admission_count' THEN 'number of earlier system-observed all-cause admissions'
    WHEN 'next_observed_aecopd_hadm_id' THEN 'next strict AECOPD admission observed in this database; absence is not a true negative'
    WHEN 'days_to_next_observed_aecopd' THEN 'days from current discharge to next observed strict AECOPD admission'
    WHEN 'has_discharge_note' THEN 'at least one matching discharge note exists'
    WHEN 'has_note_available_by_discharge' THEN 'at least one matching discharge note has charttime and storetime no later than dischtime'
    WHEN 'note_time_flag' THEN 'audit classification of discharge note timing relative to discharge'
    WHEN 'has_icu_stay' THEN 'at least one ICU stay is linked to this hospital admission; not an inclusion criterion'
    WHEN 'observed_allcause_span_after_discharge_days' THEN 'span to the patient last observed all-cause discharge; not validated censoring time'
    WHEN 'next_observed_aecopd_within_90d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 90 days; not a finalized label'
    WHEN 'next_observed_aecopd_within_180d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 180 days; not a finalized label'
    WHEN 'next_observed_aecopd_within_365d' THEN 'candidate indicator that a next observed strict AECOPD occurs within 365 days; not a finalized label'
    WHEN 'candidate_followup_status_90d' THEN 'candidate 90-day event/follow-up audit status; absence of event is not confirmed negative'
    WHEN 'candidate_followup_status_180d' THEN 'candidate 180-day event/follow-up audit status; absence of event is not confirmed negative'
    WHEN 'candidate_followup_status_365d' THEN 'candidate 365-day event/follow-up audit status; absence of event is not confirmed negative'
    ELSE 'See SQL and source-field name; retained without cleaning or modeling transformation'
  END AS definition
FROM information_schema.columns c
WHERE c.table_schema='project_aecopd' AND c.table_name='master_cohort'
ORDER BY c.ordinal_position;

COMMIT;

\copy (SELECT * FROM project_aecopd.master_cohort ORDER BY subject_id,admittime,hadm_id) TO 'outputs/06_master_cohort.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.master_cohort_qc) TO 'outputs/06_master_cohort_qc.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.master_cohort_data_dictionary ORDER BY ordinal_position) TO 'outputs/06_master_cohort_data_dictionary.csv' CSV HEADER


