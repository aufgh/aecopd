-- MIMIC-III：仅保存按住院汇总的文本可用性标记，不复制病历正文。
-- 严格时间判定要求 charttime 与 storetime 均不晚于出院时刻。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_note_availability AS
SELECT c.subject_id, c.hadm_id,
       count(n.row_id) AS note_rows,
       count(n.row_id) FILTER (WHERE btrim(n.category)='Discharge summary')
         AS discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Discharge summary'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Discharge summary'
           AND n.chartdate<=c.dischtime::date)
         AS chartdated_discharge_summary_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)<>'Discharge summary'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_other_note_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category)='Physician'
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_physician_note_rows,
       count(n.row_id) FILTER (
         WHERE btrim(n.category) IN ('Nursing','Nursing/other')
           AND n.charttime<=c.dischtime AND n.storetime<=c.dischtime)
         AS timed_nursing_note_rows
FROM project_aecopd.copd_coded_admissions_v1_gate c
LEFT JOIN public.noteevents n
  ON n.subject_id=c.subject_id AND n.hadm_id=c.hadm_id
 AND coalesce(n.iserror,'0')<>'1'
GROUP BY c.subject_id,c.hadm_id;

ALTER TABLE project_aecopd.copd_coded_note_availability ADD PRIMARY KEY (hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_note_availability)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-III note-availability row count mismatch';
  END IF;
END $$;

COMMIT;
