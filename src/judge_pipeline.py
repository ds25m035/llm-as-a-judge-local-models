import os
import sys
import json
from pathlib import Path
import requests
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. SETUP & CONFIGURATION
# =====================================================================
# Local LM Studio server (OpenAI-compatible API). Can be overridden via environment variable.
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")

# Paths relative to the repository root
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_EXCEL = BASE_DIR / "data" / "prometheus_subset.xlsx"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "cumulative_research_results.csv"

# MODEL SELECTION
ACTIVE_SELECTION = 3

MODEL_REGISTRY = {
    1: {"name": "Llama_2_7B", "api_id": "llama-2-7b-chat"},
    2: {"name": "Llama_3_1_8B", "api_id": "meta-llama-3.1-8b-instruct"},
    3: {"name": "Gemma_4_12B", "api_id": "google/gemma-4-12b"}
}

CURRENT_MODEL_NAME = MODEL_REGISTRY[ACTIVE_SELECTION]["name"]
CURRENT_API_ID = MODEL_REGISTRY[ACTIVE_SELECTION]["api_id"]

# Updated Prompt Version to keep baseline data safe in the CSV
PROMPT_VERSION = "Criteria_Rubric" 

# =====================================================================
# 2. DYNAMIC TEST SELECTION
# =====================================================================
# Select how you want to test:
# "all"    = The complete dataset
# "single" = Only a specific ID
# "range"  = A specific range (e.g., ID 1 to 10)
TEST_MODE = "range" 

# If "single" is selected:
TARGET_ID = 5 

# If "range" is selected:
RANGE_START = 1
RANGE_END = 50

# =====================================================================
# 3. LOAD & FILTER DATA
# =====================================================================
print(f"[INFO] Loading base data from '{INPUT_EXCEL}'...")
df_base = pd.read_excel(INPUT_EXCEL)

# Filter based on test mode
if TEST_MODE == "single":
    df_run = df_base[df_base['case_id'] == TARGET_ID].copy()
    print(f"[FILTER] Mode: SINGLE -> Testing only Case ID {TARGET_ID}")
elif TEST_MODE == "range":
    df_run = df_base[(df_base['case_id'] >= RANGE_START) & (df_base['case_id'] <= RANGE_END)].copy()
    print(f"[FILTER] Mode: RANGE -> Testing cases from {RANGE_START} to {RANGE_END}")
else: # "all"
    df_run = df_base.copy()
    print("[FILTER] Mode: ALL -> Testing the complete dataset")

if df_run.empty:
    print("[WARNING] Your filter settings returned 0 results. Please check the IDs!")
    sys.exit()

# Load existing results for incremental testing
if os.path.exists(RESULTS_CSV):
    df_results = pd.read_csv(RESULTS_CSV)
    print(f"[INFO] Loading existing results from '{RESULTS_CSV}'.")
else:
    df_results = df_base.copy()
    print("[INFO] Starting a new results file.")

# Add word count (for verbosity bias plot)
if "word_count" not in df_results.columns:
    df_results["word_count"] = df_results["model_answer"].apply(lambda x: len(str(x).split()))

# =====================================================================
# 4 & 5. DYNAMIC PROMPTING & EVALUATION
# =====================================================================
print("\n" + "="*80)
print(f"[RUNNING] Testing: {CURRENT_MODEL_NAME} | Prompt: {PROMPT_VERSION}")
print("="*80)

target_score_col = f"score_{CURRENT_MODEL_NAME}_{PROMPT_VERSION}"
target_reasoning_col = f"reasoning_{CURRENT_MODEL_NAME}_{PROMPT_VERSION}"

# Ensure target columns exist
if target_score_col not in df_results.columns:
    df_results[target_score_col] = pd.NA
    df_results[target_reasoning_col] = pd.NA

for idx, row in df_run.iterrows():
    current_case_id = row['case_id']
    result_idx = df_results.index[df_results['case_id'] == current_case_id].tolist()
    
# Safely fetch the specific criteria for this case
    case_criteria = row.get('orig_criteria', 'Overall correctness, clarity, and helpfulness')
    
    # 1. Print the case header
    print("\n========================================================================")
    print(f">> CASE ID {current_case_id:03d} | Human: {row['human_score']}")
    
    # 2. Show the criterion used for this case
    print(f"   [DEBUG] Criteria used: {case_criteria}")
    
    # DYNAMIC SYSTEM PROMPT (Built fresh for every row)
    system_prompt = (
        f"You are an expert academic evaluator. Your job is to grade the quality of an AI's answer "
        f"strictly based on the following criteria:\n"
        f"CRITERIA: {case_criteria}\n\n"
        f"Use the following universal grading scale for your evaluation:\n"
        f"1 - The criteria is not met at all (completely insufficient).\n"
        f"2 - The criteria is only barely met (severe flaws).\n"
        f"3 - The criteria is partially met (solid, but with noticeable errors).\n"
        f"4 - The criteria is mostly met (good performance, minor weaknesses).\n"
        f"5 - The criteria is perfectly and completely met (excellent).\n\n"
        f"You MUST respond ONLY with a valid JSON object containing two fields: 'score' (an integer from 1 to 5) "
        f"and 'reasoning' (a short explanation in English)."
    )
    
    user_content = f"Prompt given to AI: {row['prompt']}\n\nAI's Answer to evaluate: {row['model_answer']}"
    
    payload = {
        "model": CURRENT_API_ID, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1
    }
    
    # 3. Prepare the output line for the AI evaluation
    print("   AI Evaluation: ", end="", flush=True)



    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=110)
        result = response.json()
        
        if "error" in result:
            raise ValueError(f"Server Error: {result['error']['message']}")
            
        content = result['choices'][0]['message']['content'].strip()
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        data = json.loads(content)
        score = int(data['score'])  # a missing score raises an error instead of defaulting to 3
        reasoning = data.get('reasoning', 'No reasoning parsed.')
        score = max(1, min(5, score)) 
        
        # Print the score right after "AI: "
        print(f"{score} [OK]")
        print(f"   Reasoning: {reasoning}")
        
        if result_idx:
            df_results.at[result_idx[0], target_score_col] = score
            df_results.at[result_idx[0], target_reasoning_col] = reasoning
        
    except Exception as e:
        print(f"FAILED (Error: {e})")
        if result_idx:
            # Failed calls are stored as missing values so they do not distort the statistics
            df_results.at[result_idx[0], target_score_col] = pd.NA
            df_results.at[result_idx[0], target_reasoning_col] = "Error during API call."

# Save results
df_results.to_csv(RESULTS_CSV, index=False)
print(f"\n[SUCCESS] Run finished! Results saved to '{RESULTS_CSV}'.")


# =====================================================================
# 6. PLOTTING
# =====================================================================
# Create a copy to avoid modifying the real data directly
df_plot = df_results.copy()

# IMPORTANT: Force Pandas to read scores as actual numbers (Float/Int)
df_plot["human_score"] = pd.to_numeric(df_plot["human_score"], errors='coerce')
df_plot[target_score_col] = pd.to_numeric(df_plot[target_score_col], errors='coerce')
df_plot["word_count"] = pd.to_numeric(df_plot["word_count"], errors='coerce')

# Filter out empty rows (NAs) before plotting
df_plot = df_plot.dropna(subset=[target_score_col, "human_score"])

if not df_plot.empty and len(df_plot) > 2:
    print("\n[STATS] Generating plots...")
    spearman_corr, p_value = stats.spearmanr(df_plot["human_score"], df_plot[target_score_col])
    verbosity_corr, _ = stats.spearmanr(df_plot["word_count"], df_plot[target_score_col])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Alignment (Human vs AI)
    sns.regplot(x="human_score", y=target_score_col, data=df_plot, ax=axes[0], 
                x_jitter=0.1, scatter_kws={"s": 100, "alpha": 0.7}, line_kws={"color": "red"})
    axes[0].set_title(f"{CURRENT_MODEL_NAME} ({PROMPT_VERSION})\nAlignment (Spearman r = {spearman_corr:.2f})")
    axes[0].set_xlabel("Expert Ground Truth Score")
    axes[0].set_ylabel("AI Judge Score")
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)

    # Plot 2: Verbosity Bias
    unique_scores = df_plot[target_score_col].nunique()
    palette_style = "Blues_d" if unique_scores > 1 else ["#3182bd"]
    sns.barplot(x=target_score_col, y="word_count", data=df_plot, ax=axes[1], palette=palette_style, hue=target_score_col, legend=False)
    axes[1].set_title(f"Verbosity Bias (Corr = {verbosity_corr:.2f})")
    axes[1].set_xlabel("AI Judge Score")
    axes[1].set_ylabel("Word Count of Answer")

    plt.tight_layout()
    output_image = RESULTS_DIR / f"results_{CURRENT_MODEL_NAME}_{PROMPT_VERSION}.png"
    plt.savefig(output_image, dpi=300)
    print(f"[SUCCESS] Plot saved as '{output_image}'!")
else:
    print("\n[STATS] Not enough evaluated data available to generate meaningful plots.")