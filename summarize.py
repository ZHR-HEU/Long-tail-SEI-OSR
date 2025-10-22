#!/usr/bin/env python3
"""
Comprehensive analysis script for sweep results
Generates detailed comparison tables and identifies best configurations
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# Results Parser
# =============================================================================

def parse_results_file(results_path: str) -> Dict:
    """Parse results.json file"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def parse_summary_file(summary_path: str) -> Dict:
    """Parse summary.txt file"""
    metrics = {}

    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    patterns = {
        'OA': r'Overall Accuracy \(OA\)\s*:\s*([\d.]+)%',
        'mAcc': r'Mean Per-Class Accuracy \(mAcc\):\s*([\d.]+)%',
        'Macro-F1': r'Macro-F1\s*:\s*([\d.]+)%',
        'Balanced-Acc': r'Balanced Accuracy\s*:\s*([\d.]+)%',
        'Many-Acc': r'Majority\s+Acc:\s*([\d.]+)%',
        'Medium-Acc': r'Medium\s+Acc:\s*([\d.]+)%',
        'Few-Acc': r'Minority\s+Acc:\s*([\d.]+)%',
        'Many-F1': r'Majority\s+Acc:.*?F1:\s*([\d.]+)%',
        'Medium-F1': r'Medium\s+Acc:.*?F1:\s*([\d.]+)%',
        'Few-F1': r'Minority\s+Acc:.*?F1:\s*([\d.]+)%',
        'HM': r'Harmonic Mean\s*:\s*([\d.]+)%',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        metrics[key] = float(match.group(1)) if match else None

    return metrics


def extract_config_info(config: Dict) -> Dict:
    """Extract key configuration information"""
    stage2 = config.get('stage2', {})

    return {
        'model': config.get('model', {}).get('name', 'Unknown'),
        'stage1_loss': config.get('loss', {}).get('name', 'Unknown'),
        'stage1_sampling': config.get('sampling', {}).get('name', 'none'),
        'stage2_enabled': stage2.get('enabled', False),
        'stage2_mode': stage2.get('mode', '-'),
        'stage2_loss': stage2.get('loss', '-'),
        'stage2_sampling': stage2.get('sampler', '-'),
        'stage1_epochs': config.get('training', {}).get('epochs', 0),
        'stage2_epochs': stage2.get('epochs', 0),
    }


# =============================================================================
# Results Collector
# =============================================================================

def collect_all_results(exp_dir: str = 'experiments') -> pd.DataFrame:
    """
    Collect results from all experiments

    Returns:
        DataFrame with all experiment results
    """
    print("=" * 80)
    print("COLLECTING EXPERIMENT RESULTS")
    print("=" * 80)

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return pd.DataFrame()

    all_results = []

    # Iterate through all experiment directories
    for exp_folder in sorted(exp_path.iterdir()):
        if not exp_folder.is_dir():
            continue

        # Skip _latest symlinks
        if exp_folder.name.endswith('_latest'):
            continue

        summary_file = exp_folder / 'results' / 'summary.txt'
        results_file = exp_folder / 'results' / 'results.json'
        config_file = exp_folder / 'config.json'

        # Check if all required files exist
        if not all([summary_file.exists(), results_file.exists(), config_file.exists()]):
            print(f"⊘ Skipping incomplete: {exp_folder.name}")
            continue

        try:
            # Parse files
            metrics = parse_summary_file(str(summary_file))
            results = parse_results_file(str(results_file))

            with open(config_file, 'r') as f:
                config = json.load(f)

            config_info = extract_config_info(config)

            # Combine all information
            result = {
                'Experiment': exp_folder.name,
                **config_info,
                **metrics,
                'total_epochs': results['training']['total_epochs'],
                'best_val_acc_stage1': results['training']['best_val_acc_stage1'],
                'best_val_acc_stage2': results['training'].get('best_val_acc_stage2'),
            }

            all_results.append(result)
            print(f"✓ {exp_folder.name}")

        except Exception as e:
            print(f"✗ Error parsing {exp_folder.name}: {e}")

    if not all_results:
        print("\nNo valid experiment results found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"\n✓ Collected {len(df)} experiments\n")

    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def classify_experiment_type(row) -> str:
    """Classify experiment into categories"""
    if not row['stage2_enabled']:
        return 'Baseline'

    s2_loss = str(row['stage2_loss'])
    s2_samp = str(row['stage2_sampling'])

    if s2_samp != 'none' and s2_loss == 'CrossEntropy':
        return 'Sampling Only'
    elif s2_samp == 'none' and s2_loss != 'CrossEntropy':
        return 'Loss Only'
    elif s2_samp != 'none' and s2_loss != 'CrossEntropy':
        return 'Combined'
    else:
        return 'Other'


def generate_comparison_tables(df: pd.DataFrame, output_dir: str):
    """Generate detailed comparison tables"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING COMPARISON TABLES")
    print("=" * 80)

    # Add experiment type column
    df['Type'] = df.apply(classify_experiment_type, axis=1)

    # Key metrics to compare
    metrics = ['OA', 'mAcc', 'Macro-F1', 'Balanced-Acc',
               'Many-Acc', 'Few-Acc', 'HM']

    # =========================================================================
    # 1. Overall Ranking
    # =========================================================================
    ranking_df = df[['Experiment', 'Type', 'stage2_loss', 'stage2_sampling'] + metrics].copy()
    ranking_df = ranking_df.sort_values('OA', ascending=False)

    ranking_file = output_path / 'ranking_overall.csv'
    ranking_df.to_csv(ranking_file, index=False, float_format='%.2f')
    print(f"\n✓ Overall ranking: {ranking_file}")

    # Print top 10
    print("\n" + "=" * 80)
    print("TOP 10 EXPERIMENTS (by Overall Accuracy)")
    print("=" * 80)
    print(ranking_df.head(10)[['Experiment', 'Type', 'OA', 'mAcc', 'HM', 'Few-Acc']].to_string(index=False))

    # =========================================================================
    # 2. Comparison by Type
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON BY EXPERIMENT TYPE")
    print("=" * 80)

    type_comparison = df.groupby('Type')[metrics].agg(['mean', 'std', 'max', 'min'])
    type_file = output_path / 'comparison_by_type.csv'
    type_comparison.to_csv(type_file, float_format='%.2f')
    print(f"\n✓ Type comparison: {type_file}")

    # Print summary
    for metric in ['OA', 'mAcc', 'HM', 'Few-Acc']:
        print(f"\n{metric}:")
        summary = df.groupby('Type')[metric].agg(['mean', 'std', 'max'])
        print(summary.to_string(float_format='%.2f'))

    # =========================================================================
    # 3. Sampling Methods Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAMPLING METHODS COMPARISON")
    print("=" * 80)

    sampling_df = df[df['Type'] == 'Sampling Only'].copy()
    if len(sampling_df) > 0:
        sampling_comparison = sampling_df.groupby('stage2_sampling')[metrics].mean()
        sampling_comparison = sampling_comparison.sort_values('OA', ascending=False)

        sampling_file = output_path / 'comparison_sampling.csv'
        sampling_comparison.to_csv(sampling_file, float_format='%.2f')
        print(f"\n✓ Sampling comparison: {sampling_file}")
        print("\n" + sampling_comparison.to_string(float_format='%.2f'))
    else:
        print("\n⊘ No sampling-only experiments found")

    # =========================================================================
    # 4. Loss Functions Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOSS FUNCTIONS COMPARISON")
    print("=" * 80)

    loss_df = df[df['Type'] == 'Loss Only'].copy()
    if len(loss_df) > 0:
        loss_comparison = loss_df.groupby('stage2_loss')[metrics].mean()
        loss_comparison = loss_comparison.sort_values('OA', ascending=False)

        loss_file = output_path / 'comparison_loss.csv'
        loss_comparison.to_csv(loss_file, float_format='%.2f')
        print(f"\n✓ Loss comparison: {loss_file}")
        print("\n" + loss_comparison.to_string(float_format='%.2f'))
    else:
        print("\n⊘ No loss-only experiments found")

    # =========================================================================
    # 5. Full Combination Matrix
    # =========================================================================
    print("\n" + "=" * 80)
    print("FULL COMBINATION MATRIX")
    print("=" * 80)

    combined_df = df[df['Type'] == 'Combined'].copy()
    if len(combined_df) > 0:
        # Create pivot table for each metric
        for metric in ['OA', 'mAcc', 'HM', 'Few-Acc']:
            pivot = combined_df.pivot_table(
                values=metric,
                index='stage2_sampling',
                columns='stage2_loss',
                aggfunc='mean'
            )

            pivot_file = output_path / f'matrix_{metric.lower().replace("-", "_")}.csv'
            pivot.to_csv(pivot_file, float_format='%.2f')
            print(f"\n✓ {metric} matrix: {pivot_file}")

            print(f"\n{metric} Matrix:")
            print(pivot.to_string(float_format='%.2f', na_rep='-'))
    else:
        print("\n⊘ No combined experiments found")

    # =========================================================================
    # 6. Best Configuration for Each Metric
    # =========================================================================
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS PER METRIC")
    print("=" * 80)

    best_configs = []
    for metric in metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]
            best_configs.append({
                'Metric': metric,
                'Value': best_row[metric],
                'Experiment': best_row['Experiment'],
                'Type': best_row['Type'],
                'Loss': best_row['stage2_loss'],
                'Sampling': best_row['stage2_sampling'],
            })
            print(f"\n{metric}: {best_row[metric]:.2f}%")
            print(f"  Experiment: {best_row['Experiment']}")
            print(f"  Config: {best_row['stage2_loss']} + {best_row['stage2_sampling']}")

    best_configs_df = pd.DataFrame(best_configs)
    best_file = output_path / 'best_configurations.csv'
    best_configs_df.to_csv(best_file, index=False, float_format='%.2f')
    print(f"\n✓ Best configs: {best_file}")

    # =========================================================================
    # 7. Statistical Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    stats_summary = df[metrics].describe()
    stats_file = output_path / 'statistics_summary.csv'
    stats_summary.to_csv(stats_file, float_format='%.2f')
    print(f"\n✓ Statistics: {stats_file}")
    print("\n" + stats_summary.to_string(float_format='%.2f'))

    # =========================================================================
    # 8. Improvement Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 80)

    baseline_df = df[df['Type'] == 'Baseline']
    if len(baseline_df) > 0:
        baseline_metrics = baseline_df[metrics].mean()

        improvements = []
        for exp_type in ['Sampling Only', 'Loss Only', 'Combined']:
            type_df = df[df['Type'] == exp_type]
            if len(type_df) > 0:
                type_metrics = type_df[metrics].mean()
                improvement = type_metrics - baseline_metrics
                improvements.append({
                    'Type': exp_type,
                    **{f'Δ{m}': improvement[m] for m in metrics}
                })

        if improvements:
            improvement_df = pd.DataFrame(improvements)
            improvement_file = output_path / 'improvement_over_baseline.csv'
            improvement_df.to_csv(improvement_file, index=False, float_format='%.2f')
            print(f"\n✓ Improvements: {improvement_file}")
            print("\n" + improvement_df.to_string(index=False, float_format='%.2f'))

    print("\n" + "=" * 80 + "\n")


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive markdown report"""
    output_path = Path(output_dir)
    report_file = output_path / 'REPORT.md'

    print(f"Generating markdown report: {report_file}")

    df['Type'] = df.apply(classify_experiment_type, axis=1)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Imbalanced Learning Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(df)}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        best_oa = df.loc[df['OA'].idxmax()]
        best_macc = df.loc[df['mAcc'].idxmax()]
        best_hm = df.loc[df['HM'].idxmax()]
        best_few = df.loc[df['Few-Acc'].idxmax()]

        f.write("### Best Performers\n\n")
        f.write(f"- **Best Overall Accuracy (OA):** {best_oa['OA']:.2f}% - {best_oa['Experiment']}\n")
        f.write(f"- **Best Mean Accuracy (mAcc):** {best_macc['mAcc']:.2f}% - {best_macc['Experiment']}\n")
        f.write(f"- **Best Harmonic Mean (HM):** {best_hm['HM']:.2f}% - {best_hm['Experiment']}\n")
        f.write(f"- **Best Minority Accuracy:** {best_few['Few-Acc']:.2f}% - {best_few['Experiment']}\n\n")

        # Top 10 Table
        f.write("### Top 10 Experiments\n\n")
        top10 = df.nlargest(10, 'OA')[['Experiment', 'Type', 'stage2_loss', 'stage2_sampling',
                                       'OA', 'mAcc', 'HM', 'Few-Acc']]
        f.write(top10.to_markdown(index=False, floatfmt='.2f'))
        f.write("\n\n---\n\n")

        # By Type Analysis
        f.write("## Performance by Experiment Type\n\n")
        type_stats = df.groupby('Type')[['OA', 'mAcc', 'HM', 'Few-Acc']].agg(['mean', 'std', 'max'])
        f.write(type_stats.to_markdown(floatfmt='.2f'))
        f.write("\n\n---\n\n")

        # Sampling Comparison
        f.write("## Sampling Methods Comparison\n\n")
        sampling_df = df[df['Type'] == 'Sampling Only']
        if len(sampling_df) > 0:
            sampling_comp = sampling_df.groupby('stage2_sampling')[['OA', 'mAcc', 'HM', 'Few-Acc']].mean()
            sampling_comp = sampling_comp.sort_values('OA', ascending=False)
            f.write(sampling_comp.to_markdown(floatfmt='.2f'))
        else:
            f.write("*No sampling-only experiments found.*\n")
        f.write("\n\n---\n\n")

        # Loss Comparison
        f.write("## Loss Functions Comparison\n\n")
        loss_df = df[df['Type'] == 'Loss Only']
        if len(loss_df) > 0:
            loss_comp = loss_df.groupby('stage2_loss')[['OA', 'mAcc', 'HM', 'Few-Acc']].mean()
            loss_comp = loss_comp.sort_values('OA', ascending=False)
            f.write(loss_comp.to_markdown(floatfmt='.2f'))
        else:
            f.write("*No loss-only experiments found.*\n")
        f.write("\n\n---\n\n")

        # Full Results Table
        f.write("## Complete Results Table\n\n")
        full_results = df[['Experiment', 'Type', 'stage2_loss', 'stage2_sampling',
                           'OA', 'mAcc', 'Macro-F1', 'HM', 'Many-Acc', 'Few-Acc']].copy()
        full_results = full_results.sort_values('OA', ascending=False)
        f.write(full_results.to_markdown(index=False, floatfmt='.2f'))
        f.write("\n\n---\n\n")

        f.write("## Notes\n\n")
        f.write("- OA: Overall Accuracy\n")
        f.write("- mAcc: Mean Per-Class Accuracy\n")
        f.write("- HM: Harmonic Mean (Many vs Few)\n")
        f.write("- Many-Acc: Majority class accuracy\n")
        f.write("- Few-Acc: Minority class accuracy\n")

    print(f"✓ Report generated: {report_file}\n")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main analysis execution"""

    # Collect results
    df = collect_all_results('experiments')

    if df.empty:
        print("No results to analyze!")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analysis_results_{timestamp}"

    # Generate all comparisons
    generate_comparison_tables(df, output_dir)

    # Generate markdown report
    generate_markdown_report(df, output_dir)

    # Save full dataframe
    full_csv = Path(output_dir) / 'all_results_full.csv'
    df.to_csv(full_csv, index=False, float_format='%.2f')
    print(f"✓ Full results saved: {full_csv}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print(f"  - View REPORT.md for comprehensive summary")
    print(f"  - Check comparison_*.csv for detailed comparisons")
    print(f"  - See matrix_*.csv for combination heatmaps")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()