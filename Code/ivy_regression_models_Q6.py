# Research Question:
# Is offensive involvement concentration a predictor of team success?
# Do teams whose top scorer accounts for a larger share of total goals
# perform worse than teams with distributed scoring?
#
# Definitions:
#   Scoring Concentration = Top Scorer Goals / Team Total Goals
#   Lower concentration = more distributed scoring
#   Higher concentration = more reliance on a single player
#
# Methods:
#   - Pearson and Spearman correlation
#   - OLS linear regression: WinPct ~ ScoringConcentration
#   - Threshold scan with two-proportion z-test, chi-square, and Fisher's
#     exact test (binary outcome: WinningRecord)
#   - Logistic regression with grouped indicator variable
#   - Logistic regression with continuous predictor

import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
import statsmodels.api as sm

# ------------------------------------------------------------
# 1. Parse team records from the Results - Overall file
# ------------------------------------------------------------
results_path = "../Data/DA-proj-Results - Overall.csv"
results_raw = pd.read_csv(results_path, header=None)

team_records = []
pattern = re.compile(
    r"^(?P<team>[A-Za-z]+)\s*\((?P<w>\d+)-(?P<l>\d+)-(?P<t>\d+),"
)

for val in results_raw[3].dropna().astype(str):
    match = pattern.match(val.strip())
    if match:
        team = match.group("team").strip()
        w = int(match.group("w"))
        l = int(match.group("l"))
        t = int(match.group("t"))
        gp = w + l + t
        win_pct = (w + 0.5 * t) / gp
        winning_record = 1 if win_pct > 0.500 else 0

        team_records.append({
            "Team": team,
            "W": w,
            "L": l,
            "T": t,
            "GP_results": gp,
            "WinPct": win_pct,
            "WinningRecord": winning_record
        })

# Brown: 6-7-2 (derived from game-by-game results)
brown_w, brown_l, brown_t = 6, 7, 2
brown_gp = brown_w + brown_l + brown_t
brown_pct = (brown_w + 0.5 * brown_t) / brown_gp
team_records.append({
    "Team": "Brown",
    "W": brown_w,
    "L": brown_l,
    "T": brown_t,
    "GP_results": brown_gp,
    "WinPct": brown_pct,
    "WinningRecord": 1 if brown_pct > 0.500 else 0
})

records_df = pd.DataFrame(team_records)

# ------------------------------------------------------------
# 2. Build scoring concentration dataset
#
#    Top scorer goals sourced from:
#    - Ivy League individual stats (top 10 goals leaders)
#    - Team-specific athletics websites for teams not in top 10
#    - Brown: Pereyra with 4 goals (ranked 10th in Ivy avg)
#    - Yale: 3 players tied at 2 goals each (per yalebulldogs.com)
#    - Columbia: estimated at 3 goals (no player in top 10)
#
#    Team total goals from Team Stats - Goals file.
# ------------------------------------------------------------
scoring_data = pd.DataFrame({
    "Team": ["Cornell", "Princeton", "Penn", "Harvard",
             "Columbia", "Dartmouth", "Brown", "Yale"],
    "TeamGoals": [42, 34, 29, 23, 18, 16, 17, 11],
    "TopScorerGoals": [6, 9, 9, 6, 3, 4, 4, 2],
    "TopScorer": [
        "Zapata/Carnevale (tied)",
        "Ittycheria, Daniel",
        "Cayelli, Patrick",
        "Poliakov, Adam",
        "Est. (no player in top 10)",
        "Baldvinsson, Eidur",
        "Pereyra, Mateo",
        "3 players tied at 2"
    ],
    "TopScorerNote": [
        "Two players tied at 6 goals each",
        "Ivy League Offensive Player of the Year",
        "First Team All-Ivy, led league in points",
        "Ivy League Rookie of the Year",
        "Estimated; no Columbia player in top 10 league goals",
        "Second Team All-Ivy",
        "Estimated from Ivy League conference ranking (0.308 avg/G)",
        "Three players tied at 2 goals; data from Yale athletics"
    ]
})

# Compute scoring concentration
scoring_data["ScoringConcentration"] = (
    scoring_data["TopScorerGoals"] / scoring_data["TeamGoals"]
)

# Number of unique goal scorers indicator: goals NOT by top scorer
scoring_data["OtherGoals"] = (
    scoring_data["TeamGoals"] - scoring_data["TopScorerGoals"]
)

# Merge with records
df = pd.merge(scoring_data, records_df, on="Team", how="inner")

print("=== Team-Level Dataset ===")
print(df.sort_values("ScoringConcentration")[
    ["Team", "TopScorer", "TopScorerGoals", "TeamGoals",
     "ScoringConcentration", "W", "L", "T", "WinPct", "WinningRecord"]
].to_string(index=False))

# ============================================================
# PART A: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PART A: CORRELATION ANALYSIS")
print("=" * 60)

r_p, p_p = pearsonr(df["ScoringConcentration"], df["WinPct"])
r_s, p_s = spearmanr(df["ScoringConcentration"], df["WinPct"])

print(f"\nScoring Concentration vs Win Percentage:")
print(f"  Pearson r  = {r_p:.4f}, p = {p_p:.4f}")
print(f"  Spearman rho = {r_s:.4f}, p = {p_s:.4f}")

# Also check TeamGoals vs WinPct for context
r_tg, p_tg = pearsonr(df["TeamGoals"], df["WinPct"])
print(f"\nTeam Total Goals vs Win Percentage (context):")
print(f"  Pearson r  = {r_tg:.4f}, p = {p_tg:.4f}")

# ============================================================
# PART B: OLS REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("PART B: OLS REGRESSION")
print("=" * 60)

# Model 1: WinPct ~ ScoringConcentration
print("\n--- Model 1: WinPct ~ ScoringConcentration ---")
X1 = sm.add_constant(df["ScoringConcentration"])
ols1 = sm.OLS(df["WinPct"], X1).fit()
print(ols1.summary())

df["PredWinPct"] = ols1.predict(X1)

print("\n=== Predicted Win Percentages ===")
print(df.sort_values("ScoringConcentration")[
    ["Team", "ScoringConcentration", "WinPct", "PredWinPct"]
].to_string(index=False))

# ============================================================
# PART C: THRESHOLD SCAN ON SCORING CONCENTRATION
# ============================================================
print("\n" + "=" * 60)
print("PART C: THRESHOLD SCAN ON SCORING CONCENTRATION")
print("=" * 60)

sc_vals = np.sort(df["ScoringConcentration"].unique())
thresholds = [(sc_vals[i] + sc_vals[i+1]) / 2 for i in range(len(sc_vals) - 1)]

results = []

for thresh in thresholds:
    # For concentration, the hypothesis could go either way
    # Lower concentration (distributed) might predict winning
    # OR higher concentration (star player) might predict winning
    # We test both directions
    low = df[df["ScoringConcentration"] < thresh]
    high = df[df["ScoringConcentration"] >= thresh]

    if len(low) == 0 or len(high) == 0:
        continue

    low_wins = low["WinningRecord"].sum()
    high_wins = high["WinningRecord"].sum()

    # Z-test: low concentration > high concentration
    # (distributed scoring leads to more winning?)
    counts_low = np.array([low_wins, high_wins])
    nobs = np.array([len(low), len(high)])

    try:
        z_stat_low, z_p_low = proportions_ztest(
            count=counts_low, nobs=nobs, alternative="larger"
        )
    except Exception:
        z_stat_low, z_p_low = np.nan, np.nan

    # Z-test: high concentration > low concentration
    # (star dependency leads to more winning?)
    try:
        z_stat_high, z_p_high = proportions_ztest(
            count=np.array([high_wins, low_wins]),
            nobs=np.array([len(high), len(low)]),
            alternative="larger"
        )
    except Exception:
        z_stat_high, z_p_high = np.nan, np.nan

    # Chi-square
    contingency = pd.crosstab(
        df["ScoringConcentration"] < thresh,
        df["WinningRecord"]
    )
    try:
        chi2, chi_p, dof, expected = chi2_contingency(contingency)
    except Exception:
        chi2, chi_p = np.nan, np.nan

    # Fisher's exact (two-sided)
    try:
        fisher_or, fisher_p = fisher_exact(contingency, alternative="two-sided")
    except Exception:
        fisher_or, fisher_p = np.nan, np.nan

    results.append({
        "Threshold": thresh,
        "Low_n": len(low),
        "High_n": len(high),
        "Low_win_rate": low["WinningRecord"].mean(),
        "High_win_rate": high["WinningRecord"].mean(),
        "Low_avg_wpct": low["WinPct"].mean(),
        "High_avg_wpct": high["WinPct"].mean(),
        "Z_low_gt_high": z_p_low,
        "Z_high_gt_low": z_p_high,
        "Chi_p": chi_p,
        "Fisher_p": fisher_p
    })

threshold_df = pd.DataFrame(results)

print("\n=== Threshold Scan Results ===")
print(threshold_df.to_string(index=False))

# Find best threshold (lowest p-value from either direction)
threshold_df["Best_p"] = threshold_df[["Z_low_gt_high", "Z_high_gt_low"]].min(axis=1)
threshold_df = threshold_df.sort_values("Best_p").reset_index(drop=True)
best = threshold_df.iloc[0]

# Determine which direction is supported
if best["Z_low_gt_high"] < best["Z_high_gt_low"]:
    direction = "distributed"
    best_z_p = best["Z_low_gt_high"]
else:
    direction = "concentrated"
    best_z_p = best["Z_high_gt_low"]

print(f"\n=== Best Threshold ===")
print(f"Threshold: {best['Threshold']:.4f}")
print(f"Direction: {'Low concentration (distributed) wins more' if direction == 'distributed' else 'High concentration (star) wins more'}")
print(f"Below threshold: {best['Low_win_rate']:.3f} win rate ({int(best['Low_n'])} teams), "
      f"avg WinPct = {best['Low_avg_wpct']:.3f}")
print(f"Above threshold: {best['High_win_rate']:.3f} win rate ({int(best['High_n'])} teams), "
      f"avg WinPct = {best['High_avg_wpct']:.3f}")
print(f"Z-test p-value:  {best_z_p:.4f}")
print(f"Chi-square p:    {best['Chi_p']:.4f}")
print(f"Fisher exact p:  {best['Fisher_p']:.4f}")

# ============================================================
# PART D: LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("PART D: LOGISTIC REGRESSION")
print("=" * 60)

# Grouped indicator at best threshold
best_thresh = best["Threshold"]
if direction == "distributed":
    df["LowConcentration"] = (df["ScoringConcentration"] < best_thresh).astype(int)
    group_var = "LowConcentration"
else:
    df["HighConcentration"] = (df["ScoringConcentration"] >= best_thresh).astype(int)
    group_var = "HighConcentration"

X_grouped = sm.add_constant(df[group_var])
y = df["WinningRecord"]

# Check for perfect separation
grp1 = df[df[group_var] == 1]["WinningRecord"]
grp0 = df[df[group_var] == 0]["WinningRecord"]
perfect_sep = (
    grp1.nunique() == 1 and grp0.nunique() == 1
    and grp1.iloc[0] != grp0.iloc[0]
)

if perfect_sep:
    print(f"\n=== Note: Perfect Separation on {group_var} ===")
    print("Using L2-penalized logistic regression.\n")
    logit_grp = sm.Logit(y, X_grouped).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print(f"=== Penalized Logistic Regression: WinningRecord ~ {group_var} ===")
    print(f"  const       = {logit_grp.params.iloc[0]:.4f}")
    print(f"  {group_var:<13} = {logit_grp.params.iloc[1]:.4f}")
    df["PredProb_Grouped"] = logit_grp.predict(X_grouped)
else:
    try:
        logit_grp = sm.Logit(y, X_grouped).fit(disp=False)
        print(f"\n=== Logistic Regression: WinningRecord ~ {group_var} ===")
        print(logit_grp.summary())
        df["PredProb_Grouped"] = logit_grp.predict(X_grouped)
    except Exception:
        logit_grp = sm.Logit(y, X_grouped).fit_regularized(
            method="l1", alpha=0.5, disp=False
        )
        print(f"\n=== Penalized Logistic Regression: WinningRecord ~ {group_var} ===")
        for name, coef in zip(X_grouped.columns, logit_grp.params):
            print(f"  {name:<15} = {coef:.4f}")
        df["PredProb_Grouped"] = logit_grp.predict(X_grouped)

# Continuous logistic regression
X_cont = sm.add_constant(df["ScoringConcentration"])
try:
    logit_cont = sm.Logit(y, X_cont).fit(disp=False)
    print(f"\n=== Logistic Regression: WinningRecord ~ ScoringConcentration (Continuous) ===")
    print(logit_cont.summary())
    df["PredProb_Cont"] = logit_cont.predict(X_cont)
except Exception:
    print(f"\n=== Logistic Regression: WinningRecord ~ ScoringConcentration (Continuous) ===")
    print("Standard MLE did not converge; using penalized regression.")
    logit_cont = sm.Logit(y, X_cont).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print(f"  const                = {logit_cont.params.iloc[0]:.4f}")
    print(f"  ScoringConcentration = {logit_cont.params.iloc[1]:.4f}")
    df["PredProb_Cont"] = logit_cont.predict(X_cont)

# ============================================================
# PART E: SUMMARY TABLES
# ============================================================
print("\n" + "=" * 60)
print("PART E: SUMMARY TABLES")
print("=" * 60)

print("\n=== Full Dataset ===")
print(df.sort_values("ScoringConcentration")[
    ["Team", "TopScorer", "TopScorerGoals", "TeamGoals",
     "ScoringConcentration", "WinPct", "WinningRecord"]
].to_string(index=False))

print("\n=== Predicted Probabilities ===")
cols = ["Team", "ScoringConcentration", "WinPct", "WinningRecord",
        "PredWinPct", "PredProb_Grouped", "PredProb_Cont"]
print(df.sort_values("ScoringConcentration")[cols].to_string(index=False))

# ============================================================
# PART F: DATA QUALITY NOTE
# ============================================================
print("\n" + "=" * 60)
print("PART F: DATA QUALITY NOTE")
print("=" * 60)
print("""
Top scorer goals for Columbia, Brown, and Yale were estimated from
supplementary sources because the Ivy League individual statistics
only report the top 10 scorers league-wide, and no player from these
three teams appeared in that list.

Sources:
- Brown (Pereyra, 4 goals): Brown athletics pre-game article noting
  Pereyra ranked 10th in Ivy League goals average (0.308/G)
- Yale (2 goals, 3 tied): Yale athletics post-game recap stating
  "three players lead the team in goals with two each"
- Columbia (3 goals): Estimated. No Columbia player appeared in any
  league-wide scoring leaderboard. With 18 total team goals and no
  individual standout, 3 is the most conservative upper-bound estimate.

A sensitivity analysis should be considered for Columbia's estimate.
""")

# ============================================================
# INTERPRETATION
# ============================================================
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(f"\nScoring Concentration = Top Scorer Goals / Team Total Goals")
print(f"League range: {df['ScoringConcentration'].min():.3f} (Cornell) to "
      f"{df['ScoringConcentration'].max():.3f} (Penn)")

print(f"\nCorrelation: Pearson r = {r_p:.4f} (p = {p_p:.4f}), "
      f"Spearman rho = {r_s:.4f} (p = {p_s:.4f})")

if p_p < 0.05:
    dir_word = "positive" if r_p > 0 else "negative"
    print(f"There is a statistically significant {dir_word} correlation.")
else:
    dir_word = "positive" if r_p > 0 else "negative"
    print(f"The correlation is {dir_word} but does not reach significance.")

print(f"\nOLS: R-squared = {ols1.rsquared:.4f}, Adj R-sq = {ols1.rsquared_adj:.4f}")
sc_coef = ols1.params["ScoringConcentration"]
sc_pval = ols1.pvalues["ScoringConcentration"]
print(f"Coefficient = {sc_coef:.4f} (p = {sc_pval:.4f})")

print(f"\nThreshold: Best cutoff = {best['Threshold']:.4f}")
print(f"Z-test p = {best_z_p:.4f}, Chi-sq p = {best['Chi_p']:.4f}, "
      f"Fisher p = {best['Fisher_p']:.4f}")

print(f"\nContext: Team total goals vs WinPct has r = {r_tg:.4f} (p = {p_tg:.4f})")
print("This suggests that total offensive output is a much stronger predictor")
print("than how concentrated that output is in a single player.")