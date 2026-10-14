\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.verification_results;
CREATE TABLE project_aecopd.verification_results AS
WITH checks AS (
    SELECT 'strict cohort rows' check_name,5950::bigint expected,COUNT(*)::bigint actual FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'strict distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'strict distinct subject_id',3548,COUNT(DISTINCT subject_id) FROM project_aecopd.aecopd_admissions_strict
    UNION ALL SELECT 'longitudinal rows',5950,COUNT(*) FROM project_aecopd.aecopd_longitudinal_history
    UNION ALL SELECT 'note audit rows',5950,COUNT(*) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'any discharge note',4556,COUNT(*) FILTER(WHERE has_discharge_note) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'note available by discharge',717,COUNT(*) FILTER(WHERE has_note_available_by_discharge) FROM project_aecopd.discharge_note_admission_audit
    UNION ALL SELECT 'ICU admission audit rows',5950,COUNT(*) FROM project_aecopd.icu_admission_audit
    UNION ALL SELECT 'admissions with ICU',2146,COUNT(*) FILTER(WHERE has_icu_stay) FROM project_aecopd.icu_admission_audit
    UNION ALL SELECT 'master cohort rows',5950,COUNT(*) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'master distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 90d',1040,COUNT(*) FILTER(WHERE next_observed_aecopd_within_90d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 180d',1381,COUNT(*) FILTER(WHERE next_observed_aecopd_within_180d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'observed next AECOPD within 365d',1752,COUNT(*) FILTER(WHERE next_observed_aecopd_within_365d) FROM project_aecopd.master_cohort
    UNION ALL SELECT 'landmark definition rows',5950,COUNT(*) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'landmark distinct hadm_id',5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'admission 24h eligible rows',5468,COUNT(*) FILTER(WHERE eligible_admission_24h) FROM project_aecopd.landmark_definitions
    UNION ALL SELECT 'landmark source availability rows',24,COUNT(*) FROM project_aecopd.landmark_source_availability
)
SELECT check_name,expected,actual,(expected=actual) AS passed FROM checks ORDER BY check_name;

COMMIT;

\copy (SELECT * FROM project_aecopd.verification_results ORDER BY check_name) TO 'outputs/99_verification_results.csv' CSV HEADER

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM project_aecopd.verification_results WHERE NOT passed) THEN
    RAISE EXCEPTION 'One or more reproducibility checks failed. Inspect outputs/99_verification_results.csv';
  END IF;
END $$;

SELECT CASE WHEN bool_and(expected=actual) THEN 'ALL_CHECKS_PASSED' ELSE 'CHECK_FAILURE' END AS verification_status
FROM (
  SELECT 5950::bigint expected,COUNT(*)::bigint actual FROM project_aecopd.master_cohort
  UNION ALL SELECT 5950,COUNT(DISTINCT hadm_id) FROM project_aecopd.master_cohort
  UNION ALL SELECT 3548,COUNT(DISTINCT subject_id) FROM project_aecopd.master_cohort
) q;


