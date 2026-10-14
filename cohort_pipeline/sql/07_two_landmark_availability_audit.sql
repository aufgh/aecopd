\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.landmark_definitions;
CREATE TABLE project_aecopd.landmark_definitions AS
SELECT
  m.subject_id,
  m.hadm_id,
  LEAST(m.admittime,COALESCE(m.edregtime,m.admittime)) AS admission_window_start,
  m.admittime + INTERVAL '24 hours' AS admission_24h_landmark,
  m.dischtime AS discharge_landmark,
  (m.dischtime > m.admittime + INTERVAL '24 hours') AS eligible_admission_24h,
  (m.dischtime > m.admittime) AS eligible_discharge,
  m.qc_nonpositive_los
FROM project_aecopd.master_cohort m;
ALTER TABLE project_aecopd.landmark_definitions ADD CONSTRAINT landmark_definitions_pk PRIMARY KEY(hadm_id);
CREATE INDEX landmark_definitions_subject_idx ON project_aecopd.landmark_definitions(subject_id);
ANALYZE project_aecopd.landmark_definitions;

DROP TABLE IF EXISTS project_aecopd.landmark_lab_event_timing;
CREATE TABLE project_aecopd.landmark_lab_event_timing AS
SELECT
  c.subject_id,c.hadm_id,l.itemid,l.charttime,l.storetime,
  (l.valuenum IS NOT NULL) AS has_numeric_value,
  l.valueuom,
  (c.eligible_admission_24h
    AND l.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark) AS occurred_by_admission_24h,
  (c.eligible_admission_24h
    AND l.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark
    AND l.storetime IS NOT NULL
    AND l.storetime <= c.admission_24h_landmark) AS available_by_admission_24h,
  (c.eligible_discharge
    AND l.charttime BETWEEN c.admission_window_start AND c.discharge_landmark) AS occurred_by_discharge,
  (c.eligible_discharge
    AND l.charttime BETWEEN c.admission_window_start AND c.discharge_landmark
    AND l.storetime IS NOT NULL
    AND l.storetime <= c.discharge_landmark) AS available_by_discharge
FROM mimiciv_hosp.labevents l
JOIN project_aecopd.landmark_definitions c
  ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id;
CREATE INDEX landmark_lab_event_timing_hadm_idx ON project_aecopd.landmark_lab_event_timing(hadm_id);
CREATE INDEX landmark_lab_event_timing_item_idx ON project_aecopd.landmark_lab_event_timing(itemid);
ANALYZE project_aecopd.landmark_lab_event_timing;

DROP TABLE IF EXISTS project_aecopd.landmark_lab_item_coverage_top100;
CREATE TABLE project_aecopd.landmark_lab_item_coverage_top100 AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), per_admission AS (
  SELECT
    itemid,hadm_id,
    COUNT(*) FILTER(WHERE available_by_admission_24h) AS admission_24h_measurements,
    COUNT(*) FILTER(WHERE available_by_discharge) AS discharge_measurements
  FROM project_aecopd.landmark_lab_event_timing
  GROUP BY itemid,hadm_id
), item_summary AS (
  SELECT
    e.itemid,
    COUNT(*) AS measurements,
    COUNT(DISTINCT e.hadm_id) AS admissions_any,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_admission_24h) AS admissions_occurred_by_admission_24h,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_admission_24h) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_discharge) AS admissions_occurred_by_discharge,
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_discharge) AS admissions_available_by_discharge,
    COUNT(*) FILTER(WHERE e.occurred_by_admission_24h AND NOT e.available_by_admission_24h) AS admission_results_stored_after_landmark,
    COUNT(*) FILTER(WHERE e.occurred_by_discharge AND NOT e.available_by_discharge) AS discharge_results_stored_after_landmark,
    COUNT(DISTINCT e.valueuom) AS unit_count,
    STRING_AGG(DISTINCT COALESCE(e.valueuom,'<NULL>'),' | ' ORDER BY COALESCE(e.valueuom,'<NULL>')) AS observed_units
  FROM project_aecopd.landmark_lab_event_timing e
  GROUP BY e.itemid
), measurement_summary AS (
  SELECT
    itemid,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY admission_24h_measurements)
      FILTER(WHERE admission_24h_measurements>0) AS median_measurements_admission_24h,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY discharge_measurements)
      FILTER(WHERE discharge_measurements>0) AS median_measurements_by_discharge
  FROM per_admission
  GROUP BY itemid
)
SELECT
  s.itemid,d.label,d.fluid,d.category,s.measurements,s.admissions_any,
  x.admission_24h_denominator,
  s.admissions_occurred_by_admission_24h,
  s.admissions_available_by_admission_24h,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(x.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  m.median_measurements_admission_24h,
  x.discharge_denominator,
  s.admissions_occurred_by_discharge,
  s.admissions_available_by_discharge,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(x.discharge_denominator,0),2) AS discharge_coverage_pct,
  m.median_measurements_by_discharge,
  s.admission_results_stored_after_landmark,
  s.discharge_results_stored_after_landmark,
  s.unit_count,s.observed_units
FROM item_summary s
CROSS JOIN denominators x
JOIN measurement_summary m USING(itemid)
LEFT JOIN mimiciv_hosp.d_labitems d USING(itemid)
ORDER BY s.admissions_available_by_discharge DESC,s.admissions_available_by_admission_24h DESC,s.measurements DESC
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.landmark_medication_coverage_top100;
CREATE TABLE project_aecopd.landmark_medication_coverage_top100 AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), medication_rows AS (
  SELECT
    'mimiciv_hosp.prescriptions'::text AS data_source,
    COALESCE(p.drug,'<NULL>')::text AS medication,
    p.subject_id,p.hadm_id,p.starttime,p.stoptime,
    p.starttime AS availability_time,
    c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,
    c.eligible_admission_24h,c.eligible_discharge,
    'starttime proxy; source has no storetime'::text AS availability_basis
  FROM mimiciv_hosp.prescriptions p
  JOIN project_aecopd.landmark_definitions c
    ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.pharmacy',COALESCE(p.medication,'<NULL>'),
    p.subject_id,p.hadm_id,p.starttime,p.stoptime,
    COALESCE(p.verifiedtime,p.entertime,p.starttime),
    c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,
    c.eligible_admission_24h,c.eligible_discharge,
    'starttime plus verifiedtime/entertime'::text
  FROM mimiciv_hosp.pharmacy p
  JOIN project_aecopd.landmark_definitions c
    ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
), summary AS (
  SELECT
    data_source,medication,MIN(availability_basis) AS availability_basis,
    COUNT(*) AS records,
    COUNT(DISTINCT hadm_id) AS admissions_any,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h
        AND starttime BETWEEN admission_window_start AND admission_24h_landmark
        AND availability_time<=admission_24h_landmark
    ) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge
        AND starttime BETWEEN admission_window_start AND discharge_landmark
        AND availability_time<=discharge_landmark
    ) AS admissions_available_by_discharge,
    COUNT(*) FILTER(WHERE starttime>discharge_landmark) AS records_starting_after_discharge
  FROM medication_rows
  GROUP BY data_source,medication
)
SELECT
  s.*,d.admission_24h_denominator,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  d.discharge_denominator,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct
FROM summary s CROSS JOIN denominators d
ORDER BY s.admissions_available_by_discharge DESC,s.admissions_available_by_admission_24h DESC,s.records DESC
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.landmark_note_coverage;
CREATE TABLE project_aecopd.landmark_note_coverage AS
WITH denominators AS (
  SELECT
    COUNT(*) FILTER(WHERE eligible_admission_24h) AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge) AS discharge_denominator
  FROM project_aecopd.landmark_definitions
), note_rows AS (
  SELECT 'mimiciv_note.discharge'::text data_source,n.subject_id,n.hadm_id,n.charttime,n.storetime,LENGTH(n.text) AS text_chars,
         c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,c.eligible_admission_24h,c.eligible_discharge
  FROM mimiciv_note.discharge n
  JOIN project_aecopd.landmark_definitions c ON c.hadm_id=n.hadm_id AND c.subject_id=n.subject_id
  UNION ALL
  SELECT 'mimiciv_note.radiology',n.subject_id,n.hadm_id,n.charttime,n.storetime,LENGTH(n.text),
         c.admission_window_start,c.admission_24h_landmark,c.discharge_landmark,c.eligible_admission_24h,c.eligible_discharge
  FROM mimiciv_note.radiology n
  JOIN project_aecopd.landmark_definitions c ON c.hadm_id=n.hadm_id AND c.subject_id=n.subject_id
), summary AS (
  SELECT
    data_source,
    COUNT(*) AS documents,
    COUNT(DISTINCT hadm_id) AS admissions_any,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h AND charttime BETWEEN admission_window_start AND admission_24h_landmark
    ) AS admissions_charted_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_admission_24h AND charttime BETWEEN admission_window_start AND admission_24h_landmark
        AND storetime IS NOT NULL
        AND storetime<=admission_24h_landmark
    ) AS admissions_available_by_admission_24h,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge AND charttime<=discharge_landmark
        AND (data_source='mimiciv_note.discharge' OR charttime>=admission_window_start)
    ) AS admissions_charted_by_discharge,
    COUNT(DISTINCT hadm_id) FILTER(
      WHERE eligible_discharge AND charttime<=discharge_landmark
        AND (data_source='mimiciv_note.discharge' OR charttime>=admission_window_start)
        AND storetime IS NOT NULL
        AND storetime<=discharge_landmark
    ) AS admissions_available_by_discharge,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY text_chars) AS median_text_chars,
    MAX(text_chars) AS max_text_chars
  FROM note_rows GROUP BY data_source
)
SELECT
  s.*,d.admission_24h_denominator,
  ROUND(100.0*s.admissions_available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
  d.discharge_denominator,
  ROUND(100.0*s.admissions_available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct
FROM summary s CROSS JOIN denominators d
ORDER BY data_source;

DROP TABLE IF EXISTS project_aecopd.landmark_source_availability;
CREATE TABLE project_aecopd.landmark_source_availability AS
WITH c AS MATERIALIZED (
  SELECT * FROM project_aecopd.landmark_definitions
), d AS (
  SELECT
    COUNT(*)::bigint AS cohort_admissions,
    COUNT(*) FILTER(WHERE eligible_admission_24h)::bigint AS admission_24h_denominator,
    COUNT(*) FILTER(WHERE eligible_discharge)::bigint AS discharge_denominator
  FROM c
), medication_keys AS MATERIALIZED (
  SELECT DISTINCT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id
  FROM mimiciv_hosp.pharmacy p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  WHERE p.pharmacy_id IS NOT NULL OR p.poe_id IS NOT NULL
  UNION
  SELECT DISTINCT p.subject_id,p.hadm_id,p.pharmacy_id,p.poe_id
  FROM mimiciv_hosp.prescriptions p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  WHERE p.pharmacy_id IS NOT NULL OR p.poe_id IS NOT NULL
), source_rows AS (
  SELECT
    'project_aecopd.longitudinal_history'::text AS data_source,'history'::text AS data_domain,
    'computed strictly before each index admission'::text AS timing_basis,
    d.cohort_admissions AS admissions_any,
    d.admission_24h_denominator AS occurred_by_admission_24h,
    d.admission_24h_denominator AS available_by_admission_24h,
    d.discharge_denominator AS occurred_by_discharge,
    d.discharge_denominator AS available_by_discharge,
    'real-time eligible'::text AS real_time_status,
    'prior AECOPD and prior all-cause admission history'::text AS audit_note
  FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.patients','demographics','patient-level anchor fields',d.cohort_admissions,d.admission_24h_denominator,d.admission_24h_denominator,d.discharge_denominator,d.discharge_denominator,'eligible with deidentification caveats','age must be derived from anchor_age and anchor_year' FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.admissions.initial','admission context','known at/near admission',d.cohort_admissions,d.admission_24h_denominator,d.admission_24h_denominator,d.discharge_denominator,d.discharge_denominator,'real-time eligible','admission type/location and ED timing; excludes discharge-only fields' FROM d
  UNION ALL
  SELECT 'mimiciv_hosp.admissions.discharge','discharge context','known at discharge',d.cohort_admissions,0,0,d.discharge_denominator,d.discharge_denominator,'discharge landmark only','LOS, discharge location and hospital outcome are unavailable to admission model' FROM d
  UNION ALL
  SELECT
    'mimiciv_hosp.labevents','laboratory','charttime plus storetime',
    COUNT(DISTINCT e.hadm_id),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_admission_24h),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_admission_24h),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.occurred_by_discharge),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE e.available_by_discharge),
    'strict availability measurable','ED-to-landmark window; result must be stored by landmark'
  FROM project_aecopd.landmark_lab_event_timing e
  UNION ALL
  SELECT
    'mimiciv_hosp.prescriptions','medication order','starttime proxy; no storetime',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'proxy only','order availability cannot be proven because prescriptions has no storetime'
  FROM mimiciv_hosp.prescriptions p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.pharmacy','pharmacy medication','starttime plus verifiedtime/entertime',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND COALESCE(p.verifiedtime,p.entertime,p.starttime)<=c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND COALESCE(p.verifiedtime,p.entertime,p.starttime)<=c.discharge_landmark),
    'strict proxy measurable','verifiedtime preferred, then entertime, then starttime'
  FROM mimiciv_hosp.pharmacy p JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.emar','actual medication administration','charttime plus storetime; linked conservatively through pharmacy_id',
    COUNT(DISTINCT e.hadm_id),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_admission_24h AND e.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_admission_24h AND e.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND e.storetime IS NOT NULL AND e.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_discharge AND e.charttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT e.hadm_id) FILTER(WHERE c.eligible_discharge AND e.charttime BETWEEN c.admission_window_start AND c.discharge_landmark AND e.storetime IS NOT NULL AND e.storetime<=c.discharge_landmark),
    'strict availability measurable','may miss eMAR rows without a cohort-linked pharmacy_id'
  FROM mimiciv_hosp.emar e
  JOIN (SELECT DISTINCT subject_id,hadm_id,pharmacy_id FROM medication_keys WHERE pharmacy_id IS NOT NULL) k
    ON k.pharmacy_id=e.pharmacy_id AND k.hadm_id=e.hadm_id AND k.subject_id=e.subject_id
  JOIN c ON c.hadm_id=e.hadm_id AND c.subject_id=e.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.poe','provider orders','ordertime; linked conservatively through poe_id',
    COUNT(DISTINCT p.hadm_id),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.ordertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_admission_24h AND p.ordertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.ordertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT p.hadm_id) FILTER(WHERE c.eligible_discharge AND p.ordertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'time proxy measurable','may miss orders without a cohort-linked poe_id'
  FROM mimiciv_hosp.poe p
  JOIN (SELECT DISTINCT subject_id,hadm_id,poe_id FROM medication_keys WHERE poe_id IS NOT NULL) k
    ON k.poe_id=p.poe_id AND k.hadm_id=p.hadm_id AND k.subject_id=p.subject_id
  JOIN c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.microbiologyevents','microbiology','charttime/chartdate plus storetime/storedate',
    COUNT(DISTINCT m.hadm_id),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_admission_24h AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_admission_24h AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.admission_24h_landmark AND COALESCE(m.storetime,m.storedate) IS NOT NULL AND COALESCE(m.storetime,m.storedate)<=c.admission_24h_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_discharge AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT m.hadm_id) FILTER(WHERE c.eligible_discharge AND COALESCE(m.charttime,m.chartdate) BETWEEN c.admission_window_start AND c.discharge_landmark AND COALESCE(m.storetime,m.storedate) IS NOT NULL AND COALESCE(m.storetime,m.storedate)<=c.discharge_landmark),
    'strict availability measurable','culture result availability can lag specimen time by days'
  FROM mimiciv_hosp.microbiologyevents m JOIN c ON c.hadm_id=m.hadm_id AND c.subject_id=m.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.transfers','location trajectory','intime/outtime',
    COUNT(DISTINCT t.hadm_id),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_admission_24h AND t.intime<=c.admission_24h_landmark AND COALESCE(t.outtime,c.admission_24h_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_admission_24h AND t.intime<=c.admission_24h_landmark AND COALESCE(t.outtime,c.admission_24h_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_discharge AND t.intime<=c.discharge_landmark AND COALESCE(t.outtime,c.discharge_landmark)>=c.admission_window_start),
    COUNT(DISTINCT t.hadm_id) FILTER(WHERE c.eligible_discharge AND t.intime<=c.discharge_landmark AND COALESCE(t.outtime,c.discharge_landmark)>=c.admission_window_start),
    'event-time proxy','source has no storetime'
  FROM mimiciv_hosp.transfers t JOIN c ON c.hadm_id=t.hadm_id AND c.subject_id=t.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.services','clinical service','transfertime',
    COUNT(DISTINCT s.hadm_id),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_admission_24h AND s.transfertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_admission_24h AND s.transfertime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_discharge AND s.transfertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT s.hadm_id) FILTER(WHERE c.eligible_discharge AND s.transfertime BETWEEN c.admission_window_start AND c.discharge_landmark),
    'event-time proxy','source has no storetime'
  FROM mimiciv_hosp.services s JOIN c ON c.hadm_id=s.hadm_id AND c.subject_id=s.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.icustays','ICU stay','intime/outtime',
    COUNT(DISTINCT i.hadm_id),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_admission_24h AND i.intime<=c.admission_24h_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_admission_24h AND i.intime<=c.admission_24h_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_discharge AND i.intime<=c.discharge_landmark),
    COUNT(DISTINCT i.hadm_id) FILTER(WHERE c.eligible_discharge AND i.intime<=c.discharge_landmark),
    'event-time proxy','ICU status after admission landmark is an outcome for severity task, not an input'
  FROM mimiciv_icu.icustays i JOIN c ON c.hadm_id=i.hadm_id AND c.subject_id=i.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.inputevents','ICU inputs and infusions','starttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; medication/fluid ingredients require item-level review'
  FROM mimiciv_icu.inputevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.outputevents','ICU outputs','charttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.charttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.charttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.charttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; urine and other outputs require item-level review'
  FROM mimiciv_icu.outputevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.procedureevents','ICU procedures','starttime plus storetime',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.starttime BETWEEN c.admission_window_start AND c.admission_24h_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.admission_24h_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.starttime BETWEEN c.admission_window_start AND c.discharge_landmark AND x.storetime IS NOT NULL AND x.storetime<=c.discharge_landmark),
    'strict availability measurable','ICU-only source; item-level clinical interpretation required'
  FROM mimiciv_icu.procedureevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    n.data_source,'clinical text','charttime plus storetime',n.admissions_any,
    n.admissions_charted_by_admission_24h,n.admissions_available_by_admission_24h,
    n.admissions_charted_by_discharge,n.admissions_available_by_discharge,
    'strict availability measurable','text remains outside the structured matrix; only availability and length audited'
  FROM project_aecopd.landmark_note_coverage n
  UNION ALL
  SELECT
    'mimiciv_hosp.omr','outpatient measurements','subject_id plus chartdate; no hadm_id',
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_admission_24h AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.admission_24h_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_admission_24h AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.admission_24h_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_discharge AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    COUNT(DISTINCT c.hadm_id) FILTER(WHERE c.eligible_discharge AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date),
    'date-level proxy','365-day lookback; temporal linkage required because OMR lacks hadm_id'
  FROM c JOIN mimiciv_hosp.omr o ON o.subject_id=c.subject_id AND o.chartdate BETWEEN c.admission_window_start::date-365 AND c.discharge_landmark::date
  UNION ALL
  SELECT
    'mimiciv_hosp.diagnoses_icd','billing diagnoses','no event timestamp',
    COUNT(DISTINCT x.hadm_id),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'not real-time verifiable','current-stay diagnoses are post-discharge billing data; prior diagnoses remain eligible history'
  FROM mimiciv_hosp.diagnoses_icd x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.drgcodes','billing severity group','no event timestamp',
    COUNT(DISTINCT x.hadm_id),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'not real-time verifiable','DRG severity/mortality are retrospective billing fields'
  FROM mimiciv_hosp.drgcodes x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.hcpcsevents','billing procedures','chartdate but coding availability unknown',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.chartdate<=c.admission_24h_landmark::date),NULL::bigint,
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.chartdate<=c.discharge_landmark::date),NULL::bigint,
    'not real-time verifiable','event date does not establish when the billing record became available'
  FROM mimiciv_hosp.hcpcsevents x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_hosp.procedures_icd','billing procedures','chartdate but coding availability unknown',
    COUNT(DISTINCT x.hadm_id),
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_admission_24h AND x.chartdate<=c.admission_24h_landmark::date),NULL::bigint,
    COUNT(DISTINCT x.hadm_id) FILTER(WHERE c.eligible_discharge AND x.chartdate<=c.discharge_landmark::date),NULL::bigint,
    'not real-time verifiable','event date does not establish when the billing code became available'
  FROM mimiciv_hosp.procedures_icd x JOIN c ON c.hadm_id=x.hadm_id AND c.subject_id=x.subject_id
  UNION ALL
  SELECT
    'mimiciv_icu.chartevents','ICU vitals and observations','charttime/storetime; data scan deferred',
    COUNT(*) FILTER(WHERE i.has_icu_stay),NULL::bigint,NULL::bigint,NULL::bigint,NULL::bigint,
    'deferred for controlled extraction','432,997,491-row source lacks a cohort-key index; must select itemids and extraction strategy before scanning'
  FROM project_aecopd.icu_admission_audit i
), final AS (
  SELECT
    s.data_source,s.data_domain,s.timing_basis,d.cohort_admissions,d.admission_24h_denominator,
    s.admissions_any,s.occurred_by_admission_24h,s.available_by_admission_24h,
    ROUND(100.0*s.available_by_admission_24h/NULLIF(d.admission_24h_denominator,0),2) AS admission_24h_coverage_pct,
    d.discharge_denominator,s.occurred_by_discharge,s.available_by_discharge,
    ROUND(100.0*s.available_by_discharge/NULLIF(d.discharge_denominator,0),2) AS discharge_coverage_pct,
    s.real_time_status,s.audit_note
  FROM source_rows s CROSS JOIN d
)
SELECT * FROM final ORDER BY data_domain,data_source;

DROP TABLE IF EXISTS project_aecopd.landmark_audit_qc;
CREATE TABLE project_aecopd.landmark_audit_qc AS
SELECT 'landmark rows'::text AS check_name,5950::bigint AS expected,COUNT(*)::bigint AS actual,(COUNT(*)=5950) AS passed FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'landmark unique hadm_id',5950,COUNT(DISTINCT hadm_id),(COUNT(DISTINCT hadm_id)=5950) FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'nonpositive discharge windows',1,COUNT(*) FILTER(WHERE NOT eligible_discharge),(COUNT(*) FILTER(WHERE NOT eligible_discharge)=1) FROM project_aecopd.landmark_definitions
UNION ALL SELECT 'source availability rows',24,COUNT(*),(COUNT(*)=24) FROM project_aecopd.landmark_source_availability;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM project_aecopd.landmark_audit_qc WHERE NOT passed) THEN
    RAISE EXCEPTION 'Stage 7 landmark audit QC failed. Transaction will be rolled back.';
  END IF;
END $$;

COMMIT;

\copy (SELECT * FROM project_aecopd.landmark_definitions ORDER BY subject_id,admission_24h_landmark,hadm_id) TO 'outputs/07_landmark_definitions.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_source_availability ORDER BY data_domain,data_source) TO 'outputs/07_source_availability.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_lab_item_coverage_top100) TO 'outputs/07_lab_landmark_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_medication_coverage_top100) TO 'outputs/07_medication_landmark_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_note_coverage ORDER BY data_source) TO 'outputs/07_note_landmark_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.landmark_audit_qc ORDER BY check_name) TO 'outputs/07_landmark_audit_qc.csv' CSV HEADER

SELECT 'STAGE_07_LANDMARK_AUDIT_PASSED' AS status;


