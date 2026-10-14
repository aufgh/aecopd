\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';

\copy (SELECT extname, extversion FROM pg_extension ORDER BY extname) TO 'outputs/00_postgresql_extensions.csv' CSV HEADER

\copy (SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') AND table_type = 'BASE TABLE' ORDER BY table_schema, table_name) TO 'outputs/00_schema_tables.csv' CSV HEADER

\copy (SELECT table_schema, table_name, ordinal_position, column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') ORDER BY table_schema, table_name, ordinal_position) TO 'outputs/00_schema_columns.csv' CSV HEADER

\copy (SELECT n.nspname AS table_schema, c.relname AS table_name, c.reltuples::bigint AS estimated_rows, pg_total_relation_size(c.oid) AS total_bytes FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','p') AND n.nspname IN ('mimiciv_hosp','mimiciv_icu','mimiciv_note','mimiciv_derived') ORDER BY n.nspname, c.relname) TO 'outputs/00_table_row_estimates.csv' CSV HEADER

\copy (SELECT 'mimiciv_hosp.patients' AS table_name, COUNT(*) AS exact_rows FROM mimiciv_hosp.patients UNION ALL SELECT 'mimiciv_hosp.admissions', COUNT(*) FROM mimiciv_hosp.admissions UNION ALL SELECT 'mimiciv_hosp.diagnoses_icd', COUNT(*) FROM mimiciv_hosp.diagnoses_icd UNION ALL SELECT 'mimiciv_hosp.d_icd_diagnoses', COUNT(*) FROM mimiciv_hosp.d_icd_diagnoses UNION ALL SELECT 'mimiciv_note.discharge', COUNT(*) FROM mimiciv_note.discharge UNION ALL SELECT 'mimiciv_icu.icustays', COUNT(*) FROM mimiciv_icu.icustays) TO 'outputs/00_core_table_counts.csv' CSV HEADER

\copy (SELECT MIN(admittime) AS min_admittime, MAX(admittime) AS max_admittime, MIN(dischtime) AS min_dischtime, MAX(dischtime) AS max_dischtime, COUNT(*) AS admissions FROM mimiciv_hosp.admissions) TO 'outputs/00_admissions_time_range.csv' CSV HEADER

\copy (SELECT COUNT(*) AS diagnoses_rows, COUNT(*) FILTER (WHERE a.hadm_id IS NOT NULL) AS matched_admissions_rows, COUNT(*) FILTER (WHERE p.subject_id IS NOT NULL) AS matched_patients_rows, COUNT(*) FILTER (WHERE a.hadm_id IS NULL) AS unmatched_admissions_rows, COUNT(*) FILTER (WHERE p.subject_id IS NULL) AS unmatched_patients_rows FROM mimiciv_hosp.diagnoses_icd d LEFT JOIN mimiciv_hosp.admissions a ON a.subject_id=d.subject_id AND a.hadm_id=d.hadm_id LEFT JOIN mimiciv_hosp.patients p ON p.subject_id=d.subject_id) TO 'outputs/00_core_join_audit.csv' CSV HEADER



