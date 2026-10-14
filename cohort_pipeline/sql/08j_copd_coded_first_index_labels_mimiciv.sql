-- MIMIC-IV：历史首出院实验子集，不是当前完整纵向训练队列。
-- 从 08i 中每位患者只取最早一次符合条件的可存活出院；08c/08i 的其他住院不删除。
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
    RAISE EXCEPTION 'MIMIC-IV first-coded index consistency check failed';
  END IF;
END $$;
COMMIT;
