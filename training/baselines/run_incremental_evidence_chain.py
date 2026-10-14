"""运行 B0→B5 增量证据链，并生成验证集 JSON/Markdown 聚合报告。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from baselines.incremental_evidence_chain import (
    IncrementalEvidenceConfig,
    run_incremental_evidence_chain,
)
from baselines.shared_contract import SplitContract


def _format_interval(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _markdown_report(result: dict[str, object]) -> str:
    stages = result["stages"]
    assert isinstance(stages, list)
    lines = [
        "# AECOPD 增量证据链验证结果",
        "",
        "本报告只使用训练集拟合、验证集比较；冻结测试集未参与架构选择。",
        "主要估计目标为住院等权，增量区间按患者聚类配对 bootstrap 计算；患者等权 AUC 作为敏感性分析。",
        "",
        "| 层级 | 输入 | 0–30 天 AUC | 31–90 天 AUC | 91–365 天 AUC | 住院等权平均 AUC | 患者等权平均 AUC | 相对当前保留层均值差（95%CI） | 支持保留 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stage in stages:
        assert isinstance(stage, dict)
        auc = stage["aecopd_auc"]
        assert isinstance(auc, dict)
        delta = stage.get("paired_delta_vs_previous")
        if isinstance(delta, dict):
            mean = delta["mean"]
            assert isinstance(mean, dict)
            delta_text = (
                f"vs {stage.get('reference_stage', 'reference')}: {_format_interval(mean['point'])} "
                f"[{_format_interval(mean['lower'])}, {_format_interval(mean['upper'])}]"
            )
            supported = "是" if stage.get("supported_increment_vs_previous") else "否"
        else:
            delta_text = "基线"
            supported = "—"
        lines.append(
            "| {name} | {label} | {a0:.4f} | {a1:.4f} | {a2:.4f} | {mean:.4f} | {patient_mean:.4f} | {delta} | {supported} |".format(
                name=stage["name"],
                label=stage["label"],
                a0=float(auc["days_0_30"]),
                a1=float(auc["days_31_90"]),
                a2=float(auc["days_91_365"]),
                mean=float(stage["aecopd_mean_auc"]),
                patient_mean=float(stage["aecopd_mean_auc_patient_equal"]),
                delta=delta_text,
                supported=supported,
            )
        )
    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "只有相对当前已保留模型的三时间窗平均 AUC 配对差异 95%CI 下界大于 0，才记为“支持保留”并传入下一层；未获支持的信息层会从后续候选中移除。",
            "单个验证集结果用于内部架构筛选，不替代外部验证，也不恢复已经反复查看的测试集为选择集。",
            "",
            "## 审计",
            "",
            f"- 估计目标：`{result['estimand']}`",
            f"- 敏感性估计目标：`{result['sensitivity_estimand']}`",
            f"- 测试集是否使用：`{str(result['test_set_used']).lower()}`",
            f"- 训练住院：{result['train_admissions']}",
            f"- 验证住院：{result['validation_admissions']}",
            f"- 验证患者：{result['validation_patients']}",
            f"- 患者聚类 bootstrap：{result['bootstrap_resamples']} 次",
            f"- 固定残差 L2：{result['residual_l2']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the admission-equal incremental evidence chain")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/workspace/outputs/01a08b96-2988-7001-af1d-b40d901f3400/private"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--residual-l2", type=float, default=0.01)
    parser.add_argument("--text-pca-dim", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--disable-text",
        action="store_true",
        help="只运行 B0-B4；默认复用 model_v2 冻结 Qwen 章节向量运行 B5。",
    )
    args = parser.parse_args()

    model_v3 = args.base_dir / "model_v3"
    text_dir = None if args.disable_text else args.base_dir / "model_v2"
    output_dir = args.output_dir or args.base_dir / "incremental_evidence_chain"
    config = IncrementalEvidenceConfig(
        data_dir=model_v3,
        artifacts_path=model_v3 / "curated_timeline_loader_preprocessing.json",
        split=SplitContract(patient_sequence_index_path=model_v3 / "patient_sequence_index.csv.gz"),
        text_dir=text_dir,
        residual_l2=args.residual_l2,
        text_pca_dim=args.text_pca_dim,
        bootstrap_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    result = run_incremental_evidence_chain(config)
    json_path = output_dir / "validation_results.json"
    report_path = output_dir / "validation_report.md"
    _atomic_write(json_path, json.dumps(result, ensure_ascii=False, indent=2))
    _atomic_write(report_path, _markdown_report(result))

    print("[incremental-evidence-chain] validation only; test_set_used=false")
    for stage in result["stages"]:
        auc = stage["aecopd_auc"]
        print(
            f"  {stage['name']}: "
            f"{auc['days_0_30']:.4f} / {auc['days_31_90']:.4f} / "
            f"{auc['days_91_365']:.4f}; mean={stage['aecopd_mean_auc']:.4f}"
        )
    print(f"[incremental-evidence-chain] JSON -> {json_path}")
    print(f"[incremental-evidence-chain] report -> {report_path}")


if __name__ == "__main__":
    main()
