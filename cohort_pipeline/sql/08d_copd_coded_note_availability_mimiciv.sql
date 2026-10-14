-- MIMIC-IV：分别审计出院小结与影像报告的出院时点可用性；不复制正文。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_note_availability AS
WITH discharge_notes AS (
  SELECT c.hadm_id,
         count(n.note_id) AS discharge_summary_rows,
         count(n.note_id) FILTER (
           WHERE n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
           AS timed_discharge_summary_rows
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  LEFT JOIN mimiciv_note.discharge n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
), radiology_notes AS (
  SELECT c.hadm_id,
         count(n.note_id) AS radiology_rows,
         count(n.note_id) FILTER (
           WHERE n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
           AS timed_radiology_rows
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  LEFT JOIN mimiciv_note.radiology n
    ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
)
SELECT c.subject_id,c.hadm_id,
       d.discharge_summary_rows,d.timed_discharge_summary_rows,
       r.radiology_rows,r.timed_radiology_rows
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN discharge_notes d USING (hadm_id)
JOIN radiology_notes r USING (hadm_id);

ALTER TABLE project_aecopd.copd_coded_note_availability ADD PRIMARY KEY (hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_note_availability)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-IV note-availability row count mismatch';
  END IF;
END $$;

COMMIT;
