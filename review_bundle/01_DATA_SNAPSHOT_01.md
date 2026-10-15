# 01_DATA_SNAPSHOT · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md · lines 1–23

<!-- REVIEW_FRAGMENT {"path":"docs/PUBLIC_DATA_SNAPSHOT_2026-10-15.md","part":1,"start_line":1,"end_line":23,"source_sha256":"ea6fa6ee3fb78bef0d74084780b74be6358ea14a0452b077f2ccbee8dae0f4a0","fragment_sha256":"ea6fa6ee3fb78bef0d74084780b74be6358ea14a0452b077f2ccbee8dae0f4a0","added_terminal_newline":false} -->
````markdown
# 数据与实验汇总快照：公开范围及阅读方法

2026-10-15。公开快照包括文件级来源、表/队列规模、数据字典、数组字段与shape/dtype、文件级SHA256和历史汇总指标。没有导出数组取值或患者行；元数据不能用来代替受控原始数据，也不代表获得病历再分发许可。

| 预测队列划分 | 出院次数 | 患者数 |
|---|---:|---:|
| train | 13,592 | 3,766 |
| tune | 1,531 | 402 |
| confirmation | 1,418 | 395 |
| historical test | 2,860 | 788 |
| 合计 | 19,401 | 5,351 |

同患者不跨划分。原验证集为2,949次出院/797患者，按历史固定患者哈希拆为调参和确认。确认/历史测试曾被项目早期使用，不冒称新盲测。当前披露历史已有汇总结果，没有为本次发布重新评估患者结局。

原始导出底账保留19,774次COPD编码住院/5,390患者，预测资格为19,401次出院/5,351患者，两者不是同一个计数范围。17张原始来源表的规模/文件级hash，以及66份派生NPZ的字段、形状、dtype见`docs/public_snapshot/DATA_SNAPSHOT.json`，均已纳入Markdown分卷。NPZ仅解压文件头，没有读取数组行或导出逐人标签/划分。结构目录含历史中间/失败版本；出现在目录不表示已通过验证或正在使用，实际输入以运行合同和对应manifest为准。

`docs/public_snapshot/AGGREGATE_RESULTS.json`索引146个实验目录，从658份指定命名的汇总报告中提取70,421个数值指标条目。每条保留原报告相对路径及hash；不根据目录存在断言实验完成。字段提取排除个体键、样本示例、逐人预测、数组与自由文本；除显式两端置信区间外，不递归导出列表记录，以免把无法判定聚合层级的项目混入。2份报告没有命中所选指标字段，仍保留来源记录。历史逐epoch完整日志、检查点和患者级原始报告保留本地，因此这里的“汇总快照”不是原始运行目录的全量副本。

结果按原实验/模型路径存放，不把不同划分、结局、风险集或版本的指标混成排行榜。当前V7主比较为同一tune的365天严格AECOPD首再入院及其前的竞争死亡；IBS为7–365日积分Brier，不能把WorldA本库记录终点冒称完整院外随访。

此前公开的全部文本源码和本次增量进入[Markdown阅读包](../review_bundle/00_READ_FIRST.md)。源码按原字节写入，长文件按行切分；`source_catalog.json`列出范围，`manifest.json`绑定分卷hash。执行`python tools/build_markdown_review.py --verify`核对分卷及逐字节还原，不需要病历或训练权重。

仍未公开的9份历史原型含需保留本地的病例引用，清单见阅读包首页。可重装依赖、缓存、重复发布现场、原始/派生患者数据、向量与权重不进入公开包。原始受限文件的hash只标记来源身份，不提供文件下载。
````
<!-- END REVIEW_FRAGMENT -->

## docs/public_snapshot/DATA_SNAPSHOT.json · lines 1–5332

<!-- REVIEW_FRAGMENT {"path":"docs/public_snapshot/DATA_SNAPSHOT.json","part":1,"start_line":1,"end_line":5332,"source_sha256":"218ec2b7db8f4cb43d7b42cf74ae932e935e79dd60e3126cd91e370b4fcda9c7","fragment_sha256":"218ec2b7db8f4cb43d7b42cf74ae932e935e79dd60e3126cd91e370b4fcda9c7","added_terminal_newline":false} -->
````json
{
  "scope": "aggregate metadata and array headers only; no patient rows, ids, labels or values",
  "raw_manifest_status": "validated",
  "raw_manifest_sha256": "f458ef8da5a7724dbfb22329e68690774e996daf3cadd6aed765602881e85f0c",
  "raw_export_utc_started": "2026-09-27T13:35:44.447014+00:00",
  "raw_export_utc_finished": "2026-09-27T13:39:25.977338+00:00",
  "raw_sources": {
    "cohort/admissions": {
      "source_table": "project_aecopd.copd_coded_admissions_v1_gate",
      "sha256": "52b0199bb08970f83f0de48dbac6f5c4a68298fede3ccd45ef8b3e77a5c029dd",
      "compressed_bytes": 677972,
      "rows": 19774
    },
    "raw/patients": {
      "source_table": "mimiciv_hosp.patients",
      "sha256": "95fc69b203f2ded10e462f4bdff7079f268a6cd066b6fdf5b630cd63083f8c8b",
      "compressed_bytes": 60124,
      "rows": 5390
    },
    "raw/admissions": {
      "source_table": "mimiciv_hosp.admissions",
      "sha256": "f142107021294575dea5cb35fea3e4771268018dcc18bcd9c6239fd1259d374d",
      "compressed_bytes": 865167,
      "rows": 19774
    },
    "raw/labevents": {
      "source_table": "mimiciv_hosp.labevents",
      "sha256": "14308cecfd9fc0d08538b415ac0c6a95a4d81a3624e441526040c932fc89aa4e",
      "compressed_bytes": 61954298,
      "rows": 3152512
    },
    "raw/diagnoses_icd": {
      "source_table": "mimiciv_hosp.diagnoses_icd",
      "sha256": "008eb7d4084ddae5f3afd40224c08e77dbc7e37b7630a9e74d1bea0b3e4b83e0",
      "compressed_bytes": 2052701,
      "rows": 307709
    },
    "raw/emar": {
      "source_table": "mimiciv_hosp.emar",
      "sha256": "242dbe2206da25e19a9e90415c89a4ceab9aaf4aeddc275eb60d4f6eee25368e",
      "compressed_bytes": 43800913,
      "rows": 1926692
    },
    "raw/prescriptions": {
      "source_table": "mimiciv_hosp.prescriptions",
      "sha256": "aec14dd54baf83606aba47a56e7f4b05b7bd9485e14b513203fb0b7b0afc27c4",
      "compressed_bytes": 31404602,
      "rows": 861954
    },
    "raw/pharmacy": {
      "source_table": "mimiciv_hosp.pharmacy",
      "sha256": "9d0cc625c5ff19d2b2699f58facaccaeeb52abb71a08502fc314be7fd4fe1164",
      "compressed_bytes": 28993507,
      "rows": 780177
    },
    "raw/emar_detail": {
      "source_table": "mimiciv_hosp.emar_detail",
      "sha256": "0a435c0ef4d45dc4587348782634bebd0041e9e026cc798f8bd90870da8eae29",
      "compressed_bytes": 87047026,
      "rows": 3949754
    },
    "raw/discharge": {
      "source_table": "mimiciv_note.discharge",
      "sha256": "087a266a5b027cd5062c7d7b87dc15c1178e147df940124ae7b82d43f7fcbd11",
      "compressed_bytes": 60702289,
      "rows": 14480
    },
    "raw/radiology": {
      "source_table": "mimiciv_note.radiology",
      "sha256": "530cceb478df0e8f4fca4fc703038382619e04444e4717de12c3b0208a62ee4c",
      "compressed_bytes": 17602233,
      "rows": 47117
    },
    "raw/labevents_unassigned": {
      "source_table": "mimiciv_hosp.labevents",
      "sha256": "a162f33888b09fef4e7ee1aa86734284e2b5d64bd7fb8c048fb7fdd9e3bd5982",
      "compressed_bytes": 1053444,
      "rows": 47882
    },
    "raw/d_labitems": {
      "source_table": "mimiciv_hosp.d_labitems",
      "sha256": "f856bd2a33a163902b95d326bfa3d7c11b435f7ab73d01b3b50ba0de8ec57200",
      "compressed_bytes": 15250,
      "rows": 1650
    },
    "raw/d_icd_diagnoses": {
      "source_table": "mimiciv_hosp.d_icd_diagnoses",
      "sha256": "d025ef991579197ed930cdd5de3031428bacd55e0e4af0df37e9533c11a43309",
      "compressed_bytes": 1035477,
      "rows": 112107
    },
    "outcomes/readmission_death_labels": {
      "source_table": "project_aecopd.copd_qualified_readmission_labels_v1",
      "sha256": "779267ae9ee4dedbb5412c17c35cd136e799f168cf9a1f236ffc6f4c33e0577a",
      "compressed_bytes": 783800,
      "rows": 19401
    },
    "metadata/columns": {
      "source_table": "information_schema.columns",
      "sha256": "9f0e1745ab7abf52cabc8c687ed1b9839d4dfb456d0396e401a6af7930dbd77a",
      "compressed_bytes": 2381,
      "rows": 203
    },
    "cohort/patient_splits": {
      "sha256": "bfb95a89d06df2e87adbe57ad4835d82c09c50fda9e255a86f8f13607b117b8e",
      "rows": 5390
    }
  },
  "modeling_split": {
    "total_discharges": 19401,
    "total_patients": 5351,
    "groups": {
      "train": {
        "discharges": 13592,
        "patients": 3766
      },
      "tune": {
        "discharges": 1531,
        "patients": 402
      },
      "confirmation": {
        "discharges": 1418,
        "patients": 395
      },
      "test": {
        "discharges": 2860,
        "patients": 788
      }
    },
    "validation_split": "historical patient SHA256 parity split into tune and confirmation",
    "patients_disjoint": true,
    "confirmation_test_outcomes_evaluated": false
  },
  "processed_interfaces": [
    {
      "path": "patient_data/processed/abnormality_clinical_reviewed_2026-09-29/abnormality.npz",
      "bytes": 1115210,
      "sha256": "65e10cf27c56673143cba1f9a4783430fd7bfd4ebb99a259f5f14fe80d2e9fd6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/abnormality_only_2026-09-29/abnormality.npz",
      "bytes": 1096473,
      "sha256": "971d2093e379cd8e9c4079a271f9390564a77c23713b21b0390a8c6bbf3aeadc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/abnormality_uncapped_2026-09-29/abnormality.npz",
      "bytes": 1115210,
      "sha256": "65e10cf27c56673143cba1f9a4783430fd7bfd4ebb99a259f5f14fe80d2e9fd6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            132
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "feature_names",
          "shape": [
            132
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 28112
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_2026-09-29/lab_recovery.npz",
      "bytes": 592648,
      "sha256": "bf4234262bbc8b2f6ad4e4209dc12d651d79e09605f7cf4e5b4fc39f0d76b002",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_clinical_reviewed_2026-09-29/lab_recovery.npz",
      "bytes": 596149,
      "sha256": "4f280abf4219e8cd6b0e851fd53650099048171142d64b64b77f5edef0e82daf",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/lab_recovery_uncapped_2026-09-29/lab_recovery.npz",
      "bytes": 596149,
      "sha256": "4f280abf4219e8cd6b0e851fd53650099048171142d64b64b77f5edef0e82daf",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "features",
          "shape": [
            19401,
            54
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "feature_names",
          "shape": [
            54
          ],
          "dtype": "<U53",
          "fortran_order": false,
          "uncompressed_bytes": 11576
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_2026-09-27/event_arrays.npz",
      "bytes": 2569255,
      "sha256": "cb07c790edd8e899acdafe74437ac540d973a36edd9d76efe17f079e1c064ba0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            828367
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "values",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "numeric_mask",
          "shape": [
            828367
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 828495
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_2026-09-27_incomplete_audit/event_arrays.npz",
      "bytes": 2569255,
      "sha256": "cb07c790edd8e899acdafe74437ac540d973a36edd9d76efe17f079e1c064ba0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            828367
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "values",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        },
        {
          "name": "numeric_mask",
          "shape": [
            828367
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 828495
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            828367
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3313596
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29/event_arrays.npz",
      "bytes": 2560244,
      "sha256": "994c0027efa86a3099a693093583e896a159289f1ad26becc9d34c1930c7fc5b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825644
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        },
        {
          "name": "values",
          "shape": [
            825644
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        },
        {
          "name": "numeric_mask",
          "shape": [
            825644
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825772
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825644
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3302704
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29/event_arrays.npz",
      "bytes": 2559967,
      "sha256": "a20d4e8cdc5a7d86b77ef9d5f8e388515f878e1be6c885136e79eb383eaac451",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825766
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        },
        {
          "name": "values",
          "shape": [
            825766
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        },
        {
          "name": "numeric_mask",
          "shape": [
            825766
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825894
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825766
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303192
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_gold_review_v2_2026-09-29/event_arrays.npz",
      "bytes": 2560811,
      "sha256": "0fca2ecc7d166aa72b00fb5d463d7ba3ba046aeb7a99679d31ab33632c299ac9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825776
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "values",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "numeric_mask",
          "shape": [
            825776
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825904
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_gold_review_v2_checked_2026-09-29/event_arrays.npz",
      "bytes": 2560811,
      "sha256": "0fca2ecc7d166aa72b00fb5d463d7ba3ba046aeb7a99679d31ab33632c299ac9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825776
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "values",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        },
        {
          "name": "numeric_mask",
          "shape": [
            825776
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 825904
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825776
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303232
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_invalid_regex_2026-09-29/event_arrays.npz",
      "bytes": 2523971,
      "sha256": "201e01ac8c8904974d84039c128e9a2b13c0c89d2b209bafd4570094f8eb7f8e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            815884
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        },
        {
          "name": "values",
          "shape": [
            815884
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        },
        {
          "name": "numeric_mask",
          "shape": [
            815884
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 816012
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            815884
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3263664
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_medication_screened_2026-09-29/event_arrays.npz",
      "bytes": 2560181,
      "sha256": "c1aedff2678578f49ccce7eb34f9aa32ea8dc3ade7ecb83dadbb426537fcd116",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825894
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        },
        {
          "name": "values",
          "shape": [
            825894
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        },
        {
          "name": "numeric_mask",
          "shape": [
            825894
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 826022
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825894
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3303704
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/mimiciv_copd_events_needs_local_form_fix_2026-09-29/event_arrays.npz",
      "bytes": 2560486,
      "sha256": "740c3c32d7a6c6291d8e441be0c29ec7fa166dfd21752a49dbf6a3211d73afb6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "offsets",
          "shape": [
            19402
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155344
        },
        {
          "name": "concept_ids",
          "shape": [
            825991
          ],
          "dtype": "<i4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        },
        {
          "name": "values",
          "shape": [
            825991
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        },
        {
          "name": "numeric_mask",
          "shape": [
            825991
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 826119
        },
        {
          "name": "elapsed_log_days",
          "shape": [
            825991
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3304092
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_audited_coarse_fine_2026-10-12/interface.npz",
      "bytes": 6993064,
      "sha256": "7fccc944bb19a87951efd43a14a89a508cef999895012098c85ca4a6e2f9d3fc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "labs",
          "shape": [
            19401,
            34,
            24
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 63324992
        },
        {
          "name": "lab_present",
          "shape": [
            19401,
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 659762
        },
        {
          "name": "lab_names",
          "shape": [
            34
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4072
        },
        {
          "name": "lab_family",
          "shape": [
            34
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 400
        },
        {
          "name": "lab_keep",
          "shape": [
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 162
        },
        {
          "name": "medications",
          "shape": [
            19401,
            38,
            11
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 32438600
        },
        {
          "name": "med_present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "med_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        },
        {
          "name": "med_keep",
          "shape": [
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 166
        },
        {
          "name": "drug_parent",
          "shape": [
            38,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1648
        },
        {
          "name": "coarse",
          "shape": [
            19401,
            10,
            2
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1552208
        },
        {
          "name": "micro",
          "shape": [
            19401,
            8
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "micro_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "support",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "support_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2/interface.npz",
      "bytes": 6983199,
      "sha256": "19def1f6c4e22763db514ed9155cafbf09682a7ca0c1d6721c154f76a9361f06",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "labs",
          "shape": [
            19401,
            34,
            24
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 63324992
        },
        {
          "name": "lab_present",
          "shape": [
            19401,
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 659762
        },
        {
          "name": "lab_names",
          "shape": [
            34
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4072
        },
        {
          "name": "lab_family",
          "shape": [
            34
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 400
        },
        {
          "name": "lab_keep",
          "shape": [
            34
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 162
        },
        {
          "name": "medications",
          "shape": [
            19401,
            38,
            11
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 32438600
        },
        {
          "name": "med_present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "med_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        },
        {
          "name": "med_keep",
          "shape": [
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 166
        },
        {
          "name": "drug_parent",
          "shape": [
            38,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1648
        },
        {
          "name": "coarse",
          "shape": [
            19401,
            10,
            2
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1552208
        },
        {
          "name": "micro",
          "shape": [
            19401,
            8
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "micro_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "support",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "support_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_r3/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_bonsai_tasks_2026-10-10_validated/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cbc_combinations_2026-10-10/combinations.npz",
      "bytes": 1565183,
      "sha256": "fdafca6b31e1b7d9a81279bd3b8321de37c7415b0cd3a3602927a3788caceeeb",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            5,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3492308
        },
        {
          "name": "raw",
          "shape": [
            19401,
            5,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6984488
        },
        {
          "name": "names",
          "shape": [
            5
          ],
          "dtype": "<U19",
          "fortran_order": false,
          "uncompressed_bytes": 508
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_chexbert_cxr_2026-10-09/cxr_context.npz",
      "bytes": 143564,
      "sha256": "0eebf16d9a78271032286ab3e4e17a6c6a54b90a1215947b6ed5354034afdcec",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_states",
          "shape": [
            19401,
            7
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "cxr_onehot",
          "shape": [
            19401,
            28
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2173040
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "state_names",
          "shape": [
            4
          ],
          "dtype": "<U9",
          "fortran_order": false,
          "uncompressed_bytes": 272
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_clinical_state_2026-10-09/clinical_state.npz",
      "bytes": 6070435,
      "sha256": "e5ae5208ad930fafb55c403210dd448755f3c137be79aaece9d911d7dce4a8d9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "lab_axis_raw",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 30731312
        },
        {
          "name": "lab_axis_inputs",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 15365720
        },
        {
          "name": "lab_value_masks",
          "shape": [
            19401,
            22,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2561060
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "feature_names",
          "shape": [
            9
          ],
          "dtype": "<U24",
          "fortran_order": false,
          "uncompressed_bytes": 992
        },
        {
          "name": "peak_audit_days",
          "shape": [
            19401,
            22,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_clinical_state_2026-10-09_r2/clinical_state.npz",
      "bytes": 6070435,
      "sha256": "e5ae5208ad930fafb55c403210dd448755f3c137be79aaece9d911d7dce4a8d9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "lab_axis_raw",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 30731312
        },
        {
          "name": "lab_axis_inputs",
          "shape": [
            19401,
            22,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 15365720
        },
        {
          "name": "lab_value_masks",
          "shape": [
            19401,
            22,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2561060
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "feature_names",
          "shape": [
            9
          ],
          "dtype": "<U24",
          "fortran_order": false,
          "uncompressed_bytes": 992
        },
        {
          "name": "peak_audit_days",
          "shape": [
            19401,
            22,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 10243856
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_compact_multisource_2026-10-09/compact_context.npz",
      "bytes": 238971,
      "sha256": "9c5ed7e38f6f5b81776d54d045c3ef9c1dc3ec97b84f7a54dadd992801c92da4",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "axis_names",
          "shape": [
            22
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 2680
        },
        {
          "name": "med_features",
          "shape": [
            19401,
            7,
            4
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2173040
        },
        {
          "name": "med_group_names",
          "shape": [
            7
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 940
        },
        {
          "name": "med_channel_names",
          "shape": [
            4
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 528
        },
        {
          "name": "med_pharmacy_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_linked_rx_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_unlinked_rx_audit",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 388148
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 97133
        },
        {
          "name": "micro_names",
          "shape": [
            5
          ],
          "dtype": "<U32",
          "fortran_order": false,
          "uncompressed_bytes": 768
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            3
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 58331
        },
        {
          "name": "support_names",
          "shape": [
            3
          ],
          "dtype": "<U28",
          "fortran_order": false,
          "uncompressed_bytes": 464
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_reports_validated_2026-10-09/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_selected_2026-10-09/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_selected_2026-10-09_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_trajectory_2026-10-10/cxr_trajectory.npz",
      "bytes": 172999,
      "sha256": "f059feadfd49eea66952ab09b60f61c830c2e11a039c937522f4c2b898f9234f",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_values",
          "shape": [
            19401,
            98
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7605320
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "feature_names",
          "shape": [
            14
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 1416
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_cxr_trajectory_2026-10-10_r2/cxr_trajectory.npz",
      "bytes": 175857,
      "sha256": "00e6528fd68700f5d92c21fe89d195b24875694c014f012f47df737bfb2fb3b8",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "cxr_values",
          "shape": [
            19401,
            98
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7605320
        },
        {
          "name": "cxr_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "condition_names",
          "shape": [
            7
          ],
          "dtype": "<U16",
          "fortran_order": false,
          "uncompressed_bytes": 576
        },
        {
          "name": "feature_names",
          "shape": [
            14
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 1416
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_detail_extension_2026-10-11/extension.npz",
      "bytes": 2050319,
      "sha256": "a61b71adc05bafac44339d771c7cb5f3392d68c98e62a1ae2b7dfad3e40e1585",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "new_labs",
          "shape": [
            19401,
            5,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "eos",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "med_details",
          "shape": [
            19401,
            12,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 14900096
        },
        {
          "name": "lab_values_raw",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "lab_masks",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 485153
        },
        {
          "name": "med_values_raw",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13037600
        },
        {
          "name": "med_masks",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1629812
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_detail_extension_2026-10-11_r2/extension.npz",
      "bytes": 2329704,
      "sha256": "e20e7506054567930e660e9909617463abdf37fa907cf8a0717bd918923eb047",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "new_labs",
          "shape": [
            19401,
            5,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "eos",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "med_details",
          "shape": [
            19401,
            12,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 14900096
        },
        {
          "name": "lab_values_raw",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "lab_masks",
          "shape": [
            19401,
            5,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 485153
        },
        {
          "name": "med_values_raw",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13037600
        },
        {
          "name": "med_masks",
          "shape": [
            19401,
            12,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1629812
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_discharge_regimen_interface_2026-10-14/interface.npz",
      "bytes": 160153,
      "sha256": "bde2092a19749fa93f4c1cfaa3c851c425dfc3a6c998c03e4dd3dad0fdc83553",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "section_available",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "candidate_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "positions",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "products",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "roles",
          "shape": [
            13516
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 108256
        },
        {
          "name": "class_memberships",
          "shape": [
            13516,
            3
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 40676
        },
        {
          "name": "as_needed",
          "shape": [
            13516
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 13644
        },
        {
          "name": "features",
          "shape": [
            19401,
            34
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 2638664
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09_r2/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_eos_med_context_2026-10-09_r3/eos_med_context.npz",
      "bytes": 566848,
      "sha256": "a92f1bc2f853b80883008a1181ea6f297198229effe6259daa291559a38f584e",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "eos_values_raw",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_values_scaled",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "eos_value_masks",
          "shape": [
            19401,
            2,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "eos_relation_flags",
          "shape": [
            19401,
            2,
            8
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 2483456
        },
        {
          "name": "eos_features",
          "shape": [
            19401,
            2,
            20
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3104288
        },
        {
          "name": "eos_names",
          "shape": [
            2
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 192
        },
        {
          "name": "med_exposure",
          "shape": [
            19401,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 543356
        },
        {
          "name": "med_source_mask",
          "shape": [
            19401,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 135935
        },
        {
          "name": "med_availability_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12/medications/medication_identity.npz",
      "bytes": 1501742,
      "sha256": "4b6c243c1c7f9227776f7244970c6a1737cf2cfec258a089e5397dcb7efc1c47",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r2/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r2/medications/medication_identity.npz",
      "bytes": 1494097,
      "sha256": "076b8a16389e375cf60da7ca7e5996f1243a5f944b43faeb747b8db439c14da7",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r3/labs/expanded_labs.npz",
      "bytes": 2045663,
      "sha256": "4b8840e5d708c50d07f4471cddf8b0d1db2a2b358d913c56289aa2551739c8a6",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            10,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7760528
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 3880328
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            10,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 970178
        },
        {
          "name": "names",
          "shape": [
            10
          ],
          "dtype": "<U20",
          "fortran_order": false,
          "uncompressed_bytes": 928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_expanded_candidates_2026-10-12_r3/medications/medication_identity.npz",
      "bytes": 1494694,
      "sha256": "06d5cdcd860a9347003aa51523fcf9cc8d5c23ca99505dfefdb1a870968b5f8c",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            38,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 67826024
        },
        {
          "name": "present",
          "shape": [
            19401,
            38
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 737366
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 20642792
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            38,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 5160794
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            38
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 5898032
        },
        {
          "name": "drug_names",
          "shape": [
            38
          ],
          "dtype": "<U25",
          "fortran_order": false,
          "uncompressed_bytes": 3928
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_imaging_all_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_imaging_all_2026-10-10_r2/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_local_text_interface_2026-10-13/interface.npz",
      "bytes": 10317427,
      "sha256": "845137f083949e13ec0c3c2ed829666d737ba5aa82cc0348ad9c5a451398f69b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "sentence_positions",
          "shape": [
            9339
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 74840
        },
        {
          "name": "metadata",
          "shape": [
            9339,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 373688
        },
        {
          "name": "containing_vectors",
          "shape": [
            9339,
            768
          ],
          "dtype": "<f2",
          "fortran_order": false,
          "uncompressed_bytes": 14344832
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_local_text_interface_2026-10-13_r2/interface.npz",
      "bytes": 10317427,
      "sha256": "845137f083949e13ec0c3c2ed829666d737ba5aa82cc0348ad9c5a451398f69b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "sentence_positions",
          "shape": [
            9339
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 74840
        },
        {
          "name": "metadata",
          "shape": [
            9339,
            10
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 373688
        },
        {
          "name": "containing_vectors",
          "shape": [
            9339,
            768
          ],
          "dtype": "<f2",
          "fortran_order": false,
          "uncompressed_bytes": 14344832
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_medication_identity_2026-10-12/medication_identity.npz",
      "bytes": 1187829,
      "sha256": "3dda8de610a9b11cc19bbc22b715c79eef31f936a867a001f7b52d6986b8bcc0",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            21,
            23
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 37482860
        },
        {
          "name": "present",
          "shape": [
            19401,
            21
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 407549
        },
        {
          "name": "values_raw",
          "shape": [
            19401,
            21,
            7
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 11407916
        },
        {
          "name": "value_masks",
          "shape": [
            19401,
            21,
            7
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 2852075
        },
        {
          "name": "dose_unit_id",
          "shape": [
            19401,
            21
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 3259496
        },
        {
          "name": "drug_names",
          "shape": [
            21
          ],
          "dtype": "<U23",
          "fortran_order": false,
          "uncompressed_bytes": 2060
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_model_inputs_bert_2026-10-09/context.npz",
      "bytes": 446479,
      "sha256": "d97f53555f959f5ee2d21b8c09dd8fde98c45e7466b3e2c2b1efd9f5424243bc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "history_inputs",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "history_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "current_values",
          "shape": [
            19401,
            14
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1086584
        },
        {
          "name": "current_masks",
          "shape": [
            19401,
            14
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 271742
        },
        {
          "name": "history_names",
          "shape": [
            6
          ],
          "dtype": "<U31",
          "fortran_order": false,
          "uncompressed_bytes": 872
        },
        {
          "name": "current_names",
          "shape": [
            14
          ],
          "dtype": "<U36",
          "fortran_order": false,
          "uncompressed_bytes": 2144
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_model_inputs_qwen_2026-10-09/context.npz",
      "bytes": 446479,
      "sha256": "d97f53555f959f5ee2d21b8c09dd8fde98c45e7466b3e2c2b1efd9f5424243bc",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "history_inputs",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "history_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "current_values",
          "shape": [
            19401,
            14
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1086584
        },
        {
          "name": "current_masks",
          "shape": [
            19401,
            14
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 271742
        },
        {
          "name": "history_names",
          "shape": [
            6
          ],
          "dtype": "<U31",
          "fortran_order": false,
          "uncompressed_bytes": 872
        },
        {
          "name": "current_names",
          "shape": [
            14
          ],
          "dtype": "<U36",
          "fortran_order": false,
          "uncompressed_bytes": 2144
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_oxygen_device_2026-10-13/oxygen_device.npz",
      "bytes": 527217,
      "sha256": "c58094b5dcc813893435f67914dfc6802a898b6c6c66e66cf55bdbf85eb7ec60",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "availability",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "timing",
          "shape": [
            19401,
            115
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 8924588
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "device_multihot",
          "shape": [
            19401,
            5,
            18
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 13968848
        },
        {
          "name": "device_flags",
          "shape": [
            19401,
            4
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 620960
        },
        {
          "name": "time_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "time_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "procedure_evidence",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1397000
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_pft_set_interface_2026-10-13/interface.npz",
      "bytes": 109594,
      "sha256": "d7553cc3e5539357b17f31a8891fd0cb686ee025485c22e053235e4f6b85e379",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "positions",
          "shape": [
            994
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "concepts",
          "shape": [
            994
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "raw_values",
          "shape": [
            994
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 8080
        },
        {
          "name": "values",
          "shape": [
            994
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4104
        },
        {
          "name": "metadata",
          "shape": [
            994,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 12056
        },
        {
          "name": "aggregate",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radbert_imaging_2026-10-10_r2/imaging.npz",
      "bytes": 27091123,
      "sha256": "b0b5d497bbb94346b3e54f4727de6983e5078305e3dd96560467122d3f4cde23",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_imaging_2026-10-10_r2/imaging.npz",
      "bytes": 437777,
      "sha256": "d07ec662d690f318f3a440242ff211762ab38e675f049be0728201b63a755d2b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10/imaging.npz",
      "bytes": 438027,
      "sha256": "41d6f6bfe3df5d98f72a54529a445dd332e0bef1fe165047df6db4350032b28d",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_radgraph_smoke_2026-10-10_r2/imaging.npz",
      "bytes": 331405,
      "sha256": "15648a4a0b3bf2429c3a582b7b8a62dd38b5900dbb7af03a6cb2d42eb3a954f3",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "values",
          "shape": [
            19401,
            2,
            1536
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 238399616
        },
        {
          "name": "present",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_fact_set_interface_2026-10-13/interface.npz",
      "bytes": 109491,
      "sha256": "c494a1ab9edd3023386c939cf6d1f5d44fd37d88f43681f78ef5303aa2432b5a",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "positions",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "concepts",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "roles",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "polarities",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "units",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "conditions",
          "shape": [
            997
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "numeric_present",
          "shape": [
            997
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1125
        },
        {
          "name": "raw_values",
          "shape": [
            997
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 8104
        },
        {
          "name": "values",
          "shape": [
            997
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 4116
        },
        {
          "name": "numeric_valid",
          "shape": [
            997
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 1125
        },
        {
          "name": "aggregate",
          "shape": [
            19401,
            3
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 232940
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_micro_context_2026-10-09/resp_micro_context.npz",
      "bytes": 586726,
      "sha256": "4da575e4a26447fd04469d3b83ccc2e0295d22f4f670c71fe1ec6ef69a4588f2",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "micro_source_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "icu_stay_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "resp_numeric_raw",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_scaled",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_masks",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 174737
        },
        {
          "name": "recorded_duration_hours",
          "shape": [
            19401,
            2
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 310544
        },
        {
          "name": "recorded_duration_masks",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_resp_micro_context_2026-10-09_r2/resp_micro_context.npz",
      "bytes": 586726,
      "sha256": "4da575e4a26447fd04469d3b83ccc2e0295d22f4f670c71fe1ec6ef69a4588f2",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "micro_values",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 465752
        },
        {
          "name": "micro_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        },
        {
          "name": "micro_source_state",
          "shape": [
            19401
          ],
          "dtype": "|i1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "icu_stay_present",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "resp_numeric_raw",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_scaled",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 6208448
        },
        {
          "name": "resp_numeric_masks",
          "shape": [
            19401,
            8,
            5
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "support_values",
          "shape": [
            19401,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "support_masks",
          "shape": [
            19401,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 174737
        },
        {
          "name": "recorded_duration_hours",
          "shape": [
            19401,
            2
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 310544
        },
        {
          "name": "recorded_duration_masks",
          "shape": [
            19401,
            2
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 38930
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_respiratory_transition_2026-10-13/respiratory_transition.npz",
      "bytes": 1126146,
      "sha256": "bdf62fa01cb2de4b95f91cacd29ac41d91f7ffe9e4122b69379a3cad62b581d7",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "features",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "availability",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "timing",
          "shape": [
            19401,
            96
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 7450112
        },
        {
          "name": "observed",
          "shape": [
            19401
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 19529
        },
        {
          "name": "numeric_raw",
          "shape": [
            19401,
            4,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 5587616
        },
        {
          "name": "numeric_masks",
          "shape": [
            19401,
            4,
            9
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 698564
        },
        {
          "name": "numeric_flags",
          "shape": [
            19401,
            4,
            3
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 1862624
        },
        {
          "name": "support_pairs_raw",
          "shape": [
            19401,
            6
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 931376
        },
        {
          "name": "support_pairs_masks",
          "shape": [
            19401,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 116534
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_saved_sections_aligned_2026-10-11/alignment.npz",
      "bytes": 454304,
      "sha256": "4802112ba42487e8afd8ace9f3dca1b64bbea26714dde8630232128d22c583f9",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "refs",
          "shape": [
            75086,
            3
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 1802192
        },
        {
          "name": "source_rows",
          "shape": [
            75086
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 600816
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topic_audit_96_2026-10-10/pilot_facts.npz",
      "bytes": 2677,
      "sha256": "7448d31016c1666bc85c54d2ced4e3ee12c3e2c70905994f886250ea6f237c16",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            96
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 3200
        },
        {
          "name": "values",
          "shape": [
            96,
            69
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 26624
        },
        {
          "name": "names",
          "shape": [
            69
          ],
          "dtype": "<U58",
          "fortran_order": false,
          "uncompressed_bytes": 16136
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topic_audit_96_2026-10-10_r2/pilot_facts.npz",
      "bytes": 2677,
      "sha256": "7448d31016c1666bc85c54d2ced4e3ee12c3e2c70905994f886250ea6f237c16",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            96
          ],
          "dtype": "<U8",
          "fortran_order": false,
          "uncompressed_bytes": 3200
        },
        {
          "name": "values",
          "shape": [
            96,
            69
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 26624
        },
        {
          "name": "names",
          "shape": [
            69
          ],
          "dtype": "<U58",
          "fortran_order": false,
          "uncompressed_bytes": 16136
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_short_topics_96_2026-10-10/members.npz",
      "bytes": 99134,
      "sha256": "a40d9266d4958e54da573c5dc09d120869b30a42a70939d90c1e3a91f1018a6b",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_text_geometry_2026-10-13/transform.npz",
      "bytes": 8765111,
      "sha256": "1f642052032c27430a40503d2d3956fc9006cf118c421a4b45294e7844e3192c",
      "arrays": [
        {
          "name": "means",
          "shape": [
            4,
            768
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 12416
        },
        {
          "name": "whitening",
          "shape": [
            4,
            768,
            768
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 9437312
        },
        {
          "name": "field_ids",
          "shape": [
            76974
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 615920
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    },
    {
      "path": "patient_data/processed/v6_unified_lab_state_2026-10-14/interface.npz",
      "bytes": 6749960,
      "sha256": "46b896255802a1283a4fb89fc28845342fc0274cea44a240a46c64290aa27031",
      "arrays": [
        {
          "name": "hadm_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "subject_ids",
          "shape": [
            19401
          ],
          "dtype": "<i8",
          "fortran_order": false,
          "uncompressed_bytes": 155336
        },
        {
          "name": "patient_splits",
          "shape": [
            19401
          ],
          "dtype": "<U10",
          "fortran_order": false,
          "uncompressed_bytes": 776168
        },
        {
          "name": "raw",
          "shape": [
            19401,
            36,
            9
          ],
          "dtype": "<f8",
          "fortran_order": false,
          "uncompressed_bytes": 50287520
        },
        {
          "name": "values",
          "shape": [
            19401,
            36,
            9
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 25143824
        },
        {
          "name": "valid",
          "shape": [
            19401,
            36,
            6
          ],
          "dtype": "|b1",
          "fortran_order": false,
          "uncompressed_bytes": 4190744
        },
        {
          "name": "eos_context",
          "shape": [
            19401,
            16
          ],
          "dtype": "<f4",
          "fortran_order": false,
          "uncompressed_bytes": 1241792
        },
        {
          "name": "names",
          "shape": [
            36
          ],
          "dtype": "<U29",
          "fortran_order": false,
          "uncompressed_bytes": 4304
        }
      ],
      "withheld_nonstandard_array_names": 0,
      "values_exported": false
    }
  ],
  "limits": "file hashes bind local artifacts, not an authorization to redistribute them; NPY headers read without array values"
}
````
<!-- END REVIEW_FRAGMENT -->
