-- MIMIC-IV：出院节点可证明已写入的出院小结／影像报告文本视图。
\set ON_ERROR_STOP on

CREATE VIEW project_aecopd.copd_coded_timed_notes_v1 AS
SELECT 'MIMIC-IV'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       ('discharge:'||n.note_id)::text AS note_id,
       'Discharge summary'::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN mimiciv_note.discharge n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL
UNION ALL
SELECT 'MIMIC-IV'::text AS source_dataset,
       c.subject_id,c.hadm_id,
       ('radiology:'||n.note_id)::text AS note_id,
       'Radiology'::text AS note_role,
       n.charttime,n.storetime,n.text
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN mimiciv_note.radiology n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
WHERE c.discharge_landmark_eligible
  AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime
  AND n.text IS NOT NULL;
