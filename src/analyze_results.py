import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "cumulative_research_results.csv"

# 1. Load data
try:
    df = pd.read_csv(RESULTS_CSV)
except FileNotFoundError:
    print(f"[ERROR] '{RESULTS_CSV}' not found. Run judge_pipeline.py first!")
    sys.exit()

# Automatically find all columns starting with "score_"
ai_score_columns = [col for col in df.columns if col.startswith("score_")]

if not ai_score_columns:
    print("[ERROR] No AI score columns found in the CSV!")
    sys.exit()

print("======================================================")
print(" STATISTICAL EVALUATION (LLM-as-a-Judge)")
print("======================================================\n")

# Make sure the human scores are numeric
df['human_score'] = pd.to_numeric(df['human_score'], errors='coerce')

# 2. Answer length (word count)
if 'model_answer' in df.columns:
    df['answer_length'] = df['model_answer'].apply(lambda x: len(str(x).split()))
    # Only rows with a valid human score for the baseline length correlation
    valid_human = df['human_score'].notna() & df['answer_length'].notna()
    if valid_human.any():
        length_vs_human_corr, _ = spearmanr(df['answer_length'][valid_human], df['human_score'][valid_human])
        print(f"Baseline: correlation answer length vs. human score: {length_vs_human_corr:.3f}")
    print("-" * 54 + "\n")

# 3. Loop over all AI judge models
for col in ai_score_columns:
    model_display_name = col.replace("score_", "")
    print(f"=== MODEL: {model_display_name} ===")
    
    # Convert failed entries to NaN so the script does not crash
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only rows where both human and AI have a valid score
    valid_idx = df[col].notna() & df['human_score'].notna()
    
    if not valid_idx.any():
        print("  -> No valid numeric data for this model.\n")
        continue
        
    y_human = df['human_score'][valid_idx].astype(int)
    y_ai = df[col][valid_idx].astype(int)
    
    # A) Spearman-Rangkorrelation
    if len(y_human) > 1:
        spearman_corr, p_value = spearmanr(y_human, y_ai)
        print(f"  -> Evaluated cases:             {len(y_human)}")
        print(f"  -> Spearman-Korrelation:        {spearman_corr:.3f} (p-value: {p_value:.4e})")
    else:
        print("  -> Not enough data for a correlation.")
    
    # B) Cohen's kappa (exact vs. weighted)
    kappa_exact = cohen_kappa_score(y_human, y_ai)
    kappa_weighted = cohen_kappa_score(y_human, y_ai, weights='quadratic')
    print(f"  -> Cohen's kappa (exact):       {kappa_exact:.3f}")
    print(f"  -> Weighted kappa (quadratic):  {kappa_weighted:.3f}")
    
    # C) Length bias analysis
    if 'answer_length' in df.columns:
        lengths = df['answer_length'][valid_idx]
        length_vs_ai_corr, _ = spearmanr(lengths, y_ai)
        print(f"  -> Length bias (length vs. AI): {length_vs_ai_corr:.3f}")
        
        if length_vs_ai_corr > 0.4:
            print("  [WARNING] This model shows a clear length bias!")
            
    print("\n" + "-" * 54 + "\n")



# =====================================================================
# 4. AUTOMATIC PLOTTING: BOXPLOT (Positivity Bias)
# =====================================================================
print("======================================================")
print(" GENERATING COMPARATIVE BOXPLOT...")
print("======================================================")

# Columns to compare in the boxplot (human + all AI judges)
columns_to_plot = ['human_score'] + ai_score_columns

# Temporary dataframe with only the relevant scores, without NaNs
df_boxplot = df[columns_to_plot].dropna().copy()

if not df_boxplot.empty:
    # Reshape from wide to long format (needed for seaborn boxplots)
    df_melted = df_boxplot.melt(var_name='Evaluator', value_name='Score')
    
    # Clean up names for the chart (e.g. "score_Llama_3_1_8B_Criteria_Rubric" -> "Llama_3_1_8B")
    df_melted['Evaluator'] = df_melted['Evaluator'].apply(lambda x: x.replace("score_", "").replace(f"_{ai_score_columns[0].split('_')[-2]}_{ai_score_columns[0].split('_')[-1]}", "") if "score_" in x else "Human Expert")

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Draw boxplot
    sns.boxplot(x='Evaluator', y='Score', data=df_melted, palette='Set2', width=0.5)
    
    # Labels and limits
    plt.title('Global Score Distribution: Visualizing LLM Positivity Bias', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Evaluator (Judge)', fontsize=12, labelpad=10)
    plt.ylabel('Assigned Score (1 - 5)', fontsize=12, labelpad=10)
    plt.ylim(0.5, 5.5)  # scores range from 1 to 5
    
    # Save chart
    output_filename = RESULTS_DIR / "global_positivity_bias_boxplot.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[SUCCESS] Boxplot saved as '{output_filename}'.\n")
else:
    print("[WARNING] Not enough shared data across all models to create a boxplot.\n")

    