\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

CREATE SCHEMA IF NOT EXISTS project_aecopd;

DROP TABLE IF EXISTS project_aecopd.aecopd_codebook_audit;
CREATE TABLE project_aecopd.aecopd_codebook_audit AS
WITH dictionary_candidates AS (
  SELECT
    icd_version,
    icd_code,
    long_title,
    CASE
      WHEN (icd_version = 10 AND icd_code = 'J441')
        OR (icd_version = 9 AND icd_code = '49121') THEN 'strict'
      WHEN (icd_version = 10 AND icd_code = 'J440')
        OR (icd_version = 9 AND icd_code = '49122') THEN 'expanded_addition'
      WHEN (icd_version = 10 AND (icd_code LIKE 'J41%' OR icd_code = 'J42' OR icd_code LIKE 'J43%' OR icd_code LIKE 'J44%'))
        OR (icd_version = 9 AND (icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code = '496')) THEN 'copd_all_candidate'
      ELSE 'dictionary_text_review'
    END AS phenotype_group,
    CASE
      WHEN (icd_version = 10 AND (icd_code LIKE 'J41%' OR icd_code = 'J42' OR icd_code LIKE 'J43%' OR icd_code LIKE 'J44%'))
        OR (icd_version = 9 AND (icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code = '496')) THEN TRUE
      ELSE FALSE
    END AS included_in_copd_all_candidate,
    CASE
      WHEN lower(long_title) LIKE '%asthma%' THEN 'exclude from COPD-all candidate: asthma terminology requires separate clinical decision'
      WHEN lower(long_title) LIKE '%pyelonephritis%' OR lower(long_title) LIKE '%subcutaneous%' OR lower(long_title) LIKE '%perinatal%' OR lower(long_title) LIKE '%interstitial emphysema%' OR lower(long_title) LIKE '%compensatory emphysema%' OR lower(long_title) LIKE '%screening%' THEN 'exclude: non-COPD pulmonary phenotype or non-pulmonary context'
      ELSE 'candidate based on local ICD title/code family; requires clinical confirmation'
    END AS audit_note
  FROM mimiciv_hosp.d_icd_diagnoses
  WHERE icd_code IN ('J441','J440','49121','49122','496')
     OR lower(long_title) LIKE '%chronic obstructive%'
     OR lower(long_title) LIKE '%emphysema%'
     OR lower(long_title) LIKE '%chronic bronchitis%'
)
SELECT * FROM dictionary_candidates;

DROP TABLE IF EXISTS project_aecopd.phenotype_counts_audit;
CREATE TABLE project_aecopd.phenotype_counts_audit AS
WITH definitions AS (
  SELECT 'strict_any'::text AS phenotype, d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code='J441') OR (d.icd_version=9 AND d.icd_code='49121')
  UNION ALL
  SELECT 'strict_seq1', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE d.seq_num=1 AND ((d.icd_version=10 AND d.icd_code='J441') OR (d.icd_version=9 AND d.icd_code='49121'))
  UNION ALL
  SELECT CASE WHEN d.icd_version=10 THEN 'expanded_add_J440' ELSE 'expanded_add_49122' END, d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code='J440') OR (d.icd_version=9 AND d.icd_code='49122')
  UNION ALL
  SELECT 'expanded_union', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code IN ('J441','J440')) OR (d.icd_version=9 AND d.icd_code IN ('49121','49122'))
  UNION ALL
  SELECT 'copd_all_candidate', d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND (d.icd_code LIKE 'J41%' OR d.icd_code='J42' OR d.icd_code LIKE 'J43%' OR d.icd_code LIKE 'J44%'))
     OR (d.icd_version=9 AND (d.icd_code LIKE '491%' OR d.icd_code LIKE '492%' OR d.icd_code='496'))
), dedup AS (
  SELECT DISTINCT phenotype, subject_id, hadm_id FROM definitions
), strict_admissions AS (
  SELECT subject_id, hadm_id FROM dedup WHERE phenotype='strict_any'
), strict_patients AS (
  SELECT DISTINCT subject_id FROM strict_admissions
)
SELECT
  d.phenotype,
  COUNT(DISTINCT d.subject_id) AS patients,
  COUNT(*) AS admissions,
  COUNT(DISTINCT d.subject_id) FILTER (WHERE sp.subject_id IS NULL) AS new_patients_vs_strict,
  COUNT(*) FILTER (WHERE sa.hadm_id IS NULL) AS new_admissions_vs_strict
FROM dedup d
LEFT JOIN strict_admissions sa ON sa.subject_id=d.subject_id AND sa.hadm_id=d.hadm_id
LEFT JOIN strict_patients sp ON sp.subject_id=d.subject_id
GROUP BY d.phenotype;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_codebook_audit ORDER BY icd_version, icd_code) TO 'outputs/01_aecopd_codebook.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.phenotype_counts_audit ORDER BY CASE phenotype WHEN 'strict_any' THEN 1 WHEN 'strict_seq1' THEN 2 WHEN 'expanded_add_J440' THEN 3 WHEN 'expanded_add_49122' THEN 4 WHEN 'expanded_union' THEN 5 ELSE 6 END) TO 'outputs/01_phenotype_counts.csv' CSV HEADER



