-- MIMIC-III：出院节点可证明已写入的非出院小结文本视图。
-- 出院小结的 charttime/storetime 均缺失，不能混入严格及时文本。
\set ON_ERROR_STOP on

CREATE VIEW project_aecopd.copd_coded_timed_notes_v1 AS
SELECT 'MIMIC-III'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       n.row_id::text AS note_id,
       btrim(n.category)::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN public.noteevents n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND coalesce(n.iserror,'0')<>'1'
  AND btrim(n.category)<>'Discharge summary'
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL;
