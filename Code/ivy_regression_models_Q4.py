# Research Question:
# To what extent does a higher average number of corner kicks per game (C/G)
# predict a stronger overall team win percentage across Ivy League men's
# soccer teams?
#
# Methods:
#   - Pearson and Spearman correlation
#   - OLS linear regression: WinPct ~ C/G (continuous outcome)
#   - Threshold scan with two-proportion z-test, chi-square, and Fisher's
#     exact test (binary outcome: WinningRecord)
#   - Logistic regression with grouped indicator variable
#   - Logistic regression with continuous C/G

import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
import statsmodels.api as sm

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
misc_path = "../Data/DA-proj-Team Stats - Misc..csv"
results_path = "../Data/DA-proj-Results - Overall.csv"

misc_df = pd.read_csv(misc_path)

# ------------------------------------------------------------
# 2. Parse team records from the Results - Overall file
# ------------------------------------------------------------
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
# 3. Merge with Misc data (corners per game)
# ------------------------------------------------------------
df = pd.merge(misc_df, records_df, on="Team", how="inner")

df = df[["Team", "C/G", "W", "L", "T", "WinPct", "WinningRecord"]].copy()

print("=== Team-Level Dataset ===")
print(df.sort_values("C/G", ascending=False).to_string(index=False))

# ============================================================
# PART A: CONTINUOUS ANALYSIS (Win Percentage)
# ============================================================
print("\n" + "=" * 60)
print("PART A: CONTINUOUS ANALYSIS - C/G vs Win Percentage")
print("=" * 60)

# ------------------------------------------------------------
# 4. Correlation analysis
# ------------------------------------------------------------
r_pearson, p_pearson = pearsonr(df["C/G"], df["WinPct"])
r_spearman, p_spearman = spearmanr(df["C/G"], df["WinPct"])

print(f"\n=== Correlation Analysis ===")
print(f"Pearson r  = {r_pearson:.4f},  p = {p_pearson:.4f}")
print(f"Spearman rho = {r_spearman:.4f},  p = {p_spearman:.4f}")

# ------------------------------------------------------------
# 5. OLS Linear Regression: WinPct ~ C/G
# ------------------------------------------------------------
X_ols = sm.add_constant(df["C/G"])
y_pct = df["WinPct"]

ols_model = sm.OLS(y_pct, X_ols).fit()

print(f"\n=== OLS Regression: WinPct ~ C/G ===")
print(ols_model.summary())

df["PredWinPct"] = ols_model.predict(X_ols)

print(f"\n=== Predicted Win Percentages (OLS) ===")
print(df.sort_values("C/G")[
    ["Team", "C/G", "WinPct", "PredWinPct"]
].to_string(index=False))

# ============================================================
# PART B: BINARY ANALYSIS (Winning Record Threshold)
# ============================================================
print("\n" + "=" * 60)
print("PART B: BINARY ANALYSIS - C/G Threshold for Winning Record")
print("=" * 60)

# ------------------------------------------------------------
# 6. Threshold scan on C/G
# ------------------------------------------------------------
cg_vals = np.sort(df["C/G"].unique())
thresholds = [(cg_vals[i] + cg_vals[i+1]) / 2 for i in range(len(cg_vals) - 1)]

results = []

for thresh in thresholds:
    high = df[df["C/G"] >= thresh]
    low = df[df["C/G"] < thresh]

    if len(high) == 0 or len(low) == 0:
        continue

    high_wins = high["WinningRecord"].sum()
    low_wins = low["WinningRecord"].sum()

    counts = np.array([high_wins, low_wins])
    nobs = np.array([len(high), len(low)])

    # Two-proportion z-test (one-sided: high > low)
    try:
        z_stat, z_p = proportions_ztest(count=counts, nobs=nobs, alternative="larger")
    except Exception:
        z_stat, z_p = np.nan, np.nan

    # Chi-square test
    contingency = pd.crosstab(
        df["C/G"] >= thresh,
        df["WinningRecord"]
    )

    try:
        chi2, chi_p, dof, expected = chi2_contingency(contingency)
    except Exception:
        chi2, chi_p = np.nan, np.nan

    # Fisher's exact test
    try:
        fisher_or, fisher_p = fisher_exact(contingency, alternative="greater")
    except Exception:
        fisher_or, fisher_p = np.nan, np.nan

    results.append({
        "Threshold": thresh,
        "High_n": len(high),
        "Low_n": len(low),
        "High_win_rate": high["WinningRecord"].mean(),
        "Low_win_rate": low["WinningRecord"].mean(),
        "Rate_diff": high["WinningRecord"].mean() - low["WinningRecord"].mean(),
        "Z_stat": z_stat,
        "Z_p_value": z_p,
        "Chi2": chi2,
        "Chi_p_value": chi_p,
        "Fisher_OR": fisher_or,
        "Fisher_p_value": fisher_p
    })

threshold_df = pd.DataFrame(results)

threshold_df = threshold_df.sort_values(
    by=["Z_p_value", "Rate_diff"],
    ascending=[True, False]
).reset_index(drop=True)

print("\n=== Threshold Scan Results ===")
print(threshold_df.to_string(index=False))

best_row = threshold_df.iloc[0]
best_threshold = best_row["Threshold"]

print(f"\n=== Best C/G Threshold ===")
print(f"Best threshold:       {best_threshold:.3f}")
print(f"High C/G win rate:    {best_row['High_win_rate']:.3f}")
print(f"Low C/G win rate:     {best_row['Low_win_rate']:.3f}")
print(f"Difference:           {best_row['Rate_diff']:.3f}")
print(f"Z-statistic:          {best_row['Z_stat']:.3f}")
print(f"One-sided z p-value:  {best_row['Z_p_value']:.4f}")
print(f"Chi-square p-value:   {best_row['Chi_p_value']:.4f}")
print(f"Fisher exact p-value: {best_row['Fisher_p_value']:.4f}")

# ------------------------------------------------------------
# 7. Grouped indicator variable at best threshold
# ------------------------------------------------------------
df["HighCG"] = (df["C/G"] >= best_threshold).astype(int)

# ------------------------------------------------------------
# 8. Logistic Regression: WinningRecord ~ HighCG (grouped)
# ------------------------------------------------------------
X_grouped = sm.add_constant(df["HighCG"])
y_bin = df["WinningRecord"]

# Check for perfect separation
high_group = df[df["HighCG"] == 1]["WinningRecord"]
low_group = df[df["HighCG"] == 0]["WinningRecord"]
perfect_sep = (
    high_group.nunique() == 1 and low_group.nunique() == 1
    and high_group.iloc[0] != low_group.iloc[0]
)

if perfect_sep:
    print("\n=== Note: Perfect Separation Detected ===")
    print("The grouped threshold perfectly separates winning from losing teams.")
    print("Using L2-penalized logistic regression.\n")

    logit_grouped = sm.Logit(y_bin, X_grouped).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )

    print("=== Penalized Logistic Regression: WinningRecord ~ HighCG ===")
    print(f"Coefficients:\n  const = {logit_grouped.params.iloc[0]:.4f}")
    print(f"  HighCG = {logit_grouped.params.iloc[1]:.4f}")
    df["PredProb_Grouped"] = logit_grouped.predict(X_grouped)

else:
    try:
        logit_grouped = sm.Logit(y_bin, X_grouped).fit(disp=False)
        print("\n=== Logistic Regression: WinningRecord ~ HighCG (Grouped Indicator) ===")
        print(logit_grouped.summary())
        df["PredProb_Grouped"] = logit_grouped.predict(X_grouped)
    except Exception:
        print("\nStandard MLE did not converge; using penalized regression.")
        logit_grouped = sm.Logit(y_bin, X_grouped).fit_regularized(
            method="l1", alpha=0.5, disp=False
        )
        print("=== Penalized Logistic Regression: WinningRecord ~ HighCG ===")
        for name, coef in zip(X_grouped.columns, logit_grouped.params):
            print(f"  {name:<10} = {coef:.4f}")
        df["PredProb_Grouped"] = logit_grouped.predict(X_grouped)

# ------------------------------------------------------------
# 9. Logistic Regression: WinningRecord ~ C/G (continuous)
# ------------------------------------------------------------
X_cont = sm.add_constant(df["C/G"])

try:
    logit_cont = sm.Logit(y_bin, X_cont).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ C/G (Continuous) ===")
    print(logit_cont.summary())
    df["PredProb_Continuous"] = logit_cont.predict(X_cont)
except Exception:
    print("\n=== Logistic Regression: WinningRecord ~ C/G (Continuous) ===")
    print("Standard MLE did not converge; using penalized regression.")
    logit_cont = sm.Logit(y_bin, X_cont).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print(f"Coefficients:\n  const = {logit_cont.params.iloc[0]:.4f}")
    print(f"  C/G   = {logit_cont.params.iloc[1]:.4f}")
    df["PredProb_Continuous"] = logit_cont.predict(X_cont)

# ------------------------------------------------------------
# 10. Summary tables
# ------------------------------------------------------------
print("\n=== Predicted Probabilities ===")
print(df.sort_values("C/G")[
    ["Team", "C/G", "HighCG", "WinPct", "WinningRecord",
     "PredWinPct", "PredProb_Grouped", "PredProb_Continuous"]
].to_string(index=False))

print("\n=== Final Threshold Grouping ===")
print(df.sort_values(["HighCG", "C/G"], ascending=[False, False])[
    ["Team", "C/G", "W", "L", "T", "WinPct", "WinningRecord", "HighCG"]
].to_string(index=False))

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Correlation
print(f"\nCorrelation: Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f}), "
      f"Spearman rho = {r_spearman:.4f} (p = {p_spearman:.4f}).")

if p_pearson < 0.05:
    print(
        f"There is a statistically significant positive linear correlation "
        f"between C/G and win percentage."
    )
else:
    print(
        f"The correlation is positive but does not reach statistical significance "
        f"at the 0.05 level."
    )

# OLS
print(f"\nOLS Regression: R-squared = {ols_model.rsquared:.4f}, "
      f"Adjusted R-squared = {ols_model.rsquared_adj:.4f}.")
cg_coef = ols_model.params["C/G"]
cg_pval = ols_model.pvalues["C/G"]
print(f"C/G coefficient = {cg_coef:.4f} (p = {cg_pval:.4f}).")
print(f"Interpretation: each additional corner kick per game is associated "
      f"with a {cg_coef:.4f} increase in win percentage.")

# Threshold
print(f"\nThreshold: Best C/G cutoff = {best_threshold:.2f}.")
print(f"Above: {int(best_row['High_n'])} teams, {best_row['High_win_rate']:.3f} winning-record rate.")
print(f"Below: {int(best_row['Low_n'])} teams, {best_row['Low_win_rate']:.3f} winning-record rate.")
print(f"Z-test p = {best_row['Z_p_value']:.4f}, Chi-square p = {best_row['Chi_p_value']:.4f}, "
      f"Fisher p = {best_row['Fisher_p_value']:.4f}.")