-- MIMIC-III：正式患者级首次起点；其余带 COPD 编码的住院保留在 08c 纵向表。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_first_index_labels_v2 AS
SELECT DISTINCT ON (subject_id) *,
       (coded_admission_order=1) AS is_first_observed_coded_admission
FROM project_aecopd.copd_coded_all_index_labels_v2
ORDER BY subject_id,admittime,hadm_id;

ALTER TABLE project_aecopd.copd_coded_first_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE UNIQUE INDEX copd_coded_first_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_first_index_labels_v2(subject_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_first_index_labels_v2)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_first_index_labels_v2 f
       JOIN project_aecopd.copd_coded_all_index_labels_v2 l USING(subject_id)
       WHERE (l.admittime,l.hadm_id)<(f.admittime,f.hadm_id)
     ) THEN
    RAISE EXCEPTION 'MIMIC-III first-coded index consistency check failed';
  END IF;
END $$;
COMMIT;
