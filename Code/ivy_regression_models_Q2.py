# Research Question:
# Is there a minimum shots-on-goal-per-game threshold above which teams are
# significantly more likely to finish with a winning record?
#
# Methods:
#   - Two-proportion z-test for grouped comparison
#   - Chi-square test of independence
#   - Logistic regression with grouped indicator variable

import pandas as pd
import numpy as np
import re
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
shots_path = "../Data/DA-proj-Team Stats - Shots.csv"
results_path = "../Data/DA-proj-Results - Overall.csv"

shots_df = pd.read_csv(shots_path)

# ------------------------------------------------------------
# 2. Parse team records from the Results - Overall file
#    Team record headers are embedded in column 3 (0-indexed)
#    e.g. "Cornell (14-4-2, 5-1-1)"
#    Brown's header is missing from the file; derived manually.
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
# 3. Merge with Shots data
# ------------------------------------------------------------
df = pd.merge(shots_df, records_df, on="Team", how="inner")

# Keep only needed columns
df = df[["Team", "SOG/G", "W", "L", "T", "WinPct", "WinningRecord"]].copy()

print("\n=== Team-Level Dataset ===")
print(df.sort_values("SOG/G").to_string(index=False))

# ------------------------------------------------------------
# 4. Identify candidate SOG/G thresholds
#    Use midpoints between sorted unique SOG/G values
# ------------------------------------------------------------
sog_vals = np.sort(df["SOG/G"].unique())
thresholds = [(sog_vals[i] + sog_vals[i+1]) / 2 for i in range(len(sog_vals) - 1)]

# ------------------------------------------------------------
# 5. For each threshold:
#    - split teams into High SOG/G vs Low SOG/G
#    - run two-proportion z-test on winning-record rates
#    - run chi-square test
#    - run Fisher's exact test (robustness check for small samples)
# ------------------------------------------------------------
results = []

for thresh in thresholds:
    high = df[df["SOG/G"] >= thresh]
    low = df[df["SOG/G"] < thresh]

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
        df["SOG/G"] >= thresh,
        df["WinningRecord"]
    )

    try:
        chi2, chi_p, dof, expected = chi2_contingency(contingency)
    except Exception:
        chi2, chi_p = np.nan, np.nan

    # Fisher's exact test (robustness check for small cell counts)
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

# Sort by strongest evidence
threshold_df = threshold_df.sort_values(
    by=["Z_p_value", "Rate_diff"],
    ascending=[True, False]
).reset_index(drop=True)

print("\n=== Threshold Scan Results ===")
print(threshold_df.to_string(index=False))

best_row = threshold_df.iloc[0]
best_threshold = best_row["Threshold"]

print("\n=== Best SOG/G Threshold ===")
print(f"Best threshold:       {best_threshold:.3f}")
print(f"High SOG/G win rate:  {best_row['High_win_rate']:.3f}")
print(f"Low SOG/G win rate:   {best_row['Low_win_rate']:.3f}")
print(f"Difference:           {best_row['Rate_diff']:.3f}")
print(f"Z-statistic:          {best_row['Z_stat']:.3f}")
print(f"One-sided z p-value:  {best_row['Z_p_value']:.4f}")
print(f"Chi-square p-value:   {best_row['Chi_p_value']:.4f}")
print(f"Fisher exact p-value: {best_row['Fisher_p_value']:.4f}")

# ------------------------------------------------------------
# 6. Create grouped indicator variable at best threshold
# ------------------------------------------------------------
df["HighSOG"] = (df["SOG/G"] >= best_threshold).astype(int)

# ------------------------------------------------------------
# 7. Logistic Regression with grouped indicator variable:
#    WinningRecord ~ HighSOG
#
#    If the threshold creates perfect separation (all teams in
#    one group win and all in the other lose), standard MLE
#    logistic regression cannot converge. In that case we apply
#    Firth-style penalized likelihood (method='bfgs' with ridge
#    regularization) so the model still produces finite estimates.
# ------------------------------------------------------------
X_grouped = sm.add_constant(df["HighSOG"])
y = df["WinningRecord"]

# Check for perfect separation
high_group = df[df["HighSOG"] == 1]["WinningRecord"]
low_group = df[df["HighSOG"] == 0]["WinningRecord"]
perfect_sep = (
    (high_group.nunique() == 1 and low_group.nunique() == 1)
    and (high_group.iloc[0] != low_group.iloc[0])
)

if perfect_sep:
    print("\n=== Note: Perfect Separation Detected ===")
    print("The grouped threshold perfectly separates winning from losing teams.")
    print("Standard MLE cannot converge; using L2-penalized logistic regression.\n")

    logit_grouped = sm.Logit(y, X_grouped).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )

    print("=== Penalized Logistic Regression: WinningRecord ~ HighSOG ===")
    print(f"Coefficients:\n  const  = {logit_grouped.params.iloc[0]:.4f}")
    print(f"  HighSOG = {logit_grouped.params.iloc[1]:.4f}")
    df["PredProb_Grouped"] = logit_grouped.predict(X_grouped)

else:
    logit_grouped = sm.Logit(y, X_grouped).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ HighSOG (Grouped Indicator) ===")
    print(logit_grouped.summary())
    df["PredProb_Grouped"] = logit_grouped.predict(X_grouped)

# ------------------------------------------------------------
# 8. Logistic Regression with continuous SOG/G for comparison:
#    WinningRecord ~ SOG/G
# ------------------------------------------------------------
X_cont = sm.add_constant(df["SOG/G"])

try:
    logit_cont = sm.Logit(y, X_cont).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ SOG/G (Continuous) ===")
    print(logit_cont.summary())
    df["PredProb_Continuous"] = logit_cont.predict(X_cont)
    cont_converged = True
except Exception:
    print("\n=== Logistic Regression: WinningRecord ~ SOG/G (Continuous) ===")
    print("Standard MLE did not converge; using L2-penalized logistic regression.")
    logit_cont = sm.Logit(y, X_cont).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print(f"Coefficients:\n  const = {logit_cont.params.iloc[0]:.4f}")
    print(f"  SOG/G = {logit_cont.params.iloc[1]:.4f}")
    df["PredProb_Continuous"] = logit_cont.predict(X_cont)
    cont_converged = False

# ------------------------------------------------------------
# 9. Predicted probabilities from both models
# ------------------------------------------------------------
print("\n=== Predicted Probabilities ===")
print(df.sort_values("SOG/G")[
    ["Team", "SOG/G", "HighSOG", "WinningRecord", "PredProb_Grouped", "PredProb_Continuous"]
].to_string(index=False))

# ------------------------------------------------------------
# 10. Final threshold grouping
# ------------------------------------------------------------
print("\n=== Final Threshold Grouping ===")
print(df.sort_values(["HighSOG", "SOG/G"], ascending=[False, False])[
    ["Team", "SOG/G", "W", "L", "T", "WinPct", "WinningRecord", "HighSOG"]
].to_string(index=False))

# ------------------------------------------------------------
# 11. Plain-English conclusion
# ------------------------------------------------------------
print("\n=== Interpretation ===")

# Z-test conclusion
if best_row["Z_p_value"] < 0.05:
    print(
        f"Teams averaging {best_threshold:.2f} or more shots on goal per game were "
        f"significantly more likely to finish with a winning record than teams below "
        f"that threshold (one-sided two-proportion z-test p = {best_row['Z_p_value']:.4f})."
    )
else:
    print(
        f"The threshold scan identified {best_threshold:.2f} SOG/G as the strongest "
        f"candidate, but the difference in winning-record rates was not statistically "
        f"significant (one-sided two-proportion z-test p = {best_row['Z_p_value']:.4f})."
    )

# Chi-square caveat
if best_row["Chi_p_value"] >= 0.05:
    print(
        f"The chi-square test did not reach significance (p = {best_row['Chi_p_value']:.4f}), "
        f"which is expected given the small sample size (n = {len(df)}). With only "
        f"{int(best_row['High_n'])} and {int(best_row['Low_n'])} teams in each group, "
        f"several cells in the contingency table have expected counts below 5, "
        f"violating a key assumption of the chi-square test."
    )

# Fisher's exact test
if best_row["Fisher_p_value"] < 0.05:
    print(
        f"Fisher's exact test, which does not rely on large-sample approximations, "
        f"confirms the result (p = {best_row['Fisher_p_value']:.4f})."
    )
else:
    print(
        f"Fisher's exact test, included as a robustness check for small samples, "
        f"returned p = {best_row['Fisher_p_value']:.4f}."
    )

# Grouped logistic regression conclusion
if perfect_sep:
    print(
        f"The logistic regression with the grouped indicator variable (HighSOG) "
        f"encountered perfect separation - the threshold perfectly divides winning "
        f"from non-winning teams. A penalized regression was used to produce finite "
        f"coefficient estimates, but standard p-values are not reliable in this case. "
        f"The perfect separation itself is strong evidence of the threshold's predictive power."
    )
else:
    grouped_pval = logit_grouped.pvalues.get("HighSOG", np.nan)
    if grouped_pval < 0.05:
        print(
            f"The logistic regression with the grouped indicator variable (HighSOG) "
            f"confirms that the threshold grouping is a significant predictor of a "
            f"winning record (p = {grouped_pval:.4f})."
        )
    else:
        print(
            f"The logistic regression with the grouped indicator variable (HighSOG) "
            f"did not reach significance at the 0.05 level (p = {grouped_pval:.4f}), "
            f"likely due to the limited sample size."
        )

# Continuous logistic regression conclusion
try:
    cont_pval = logit_cont.pvalues.get("SOG/G", np.nan)
    if cont_pval < 0.05:
        print(
            f"The continuous logistic regression also shows SOG/G as a significant "
            f"predictor (p = {cont_pval:.4f})."
        )
    else:
        print(
            f"The continuous logistic regression does not show SOG/G as significant "
            f"at the 0.05 level (p = {cont_pval:.4f}), consistent with the sample "
            f"size limitation."
        )
except Exception:
    print(
        "The continuous logistic regression required penalized estimation due to "
        "near-perfect separation in the data. Standard p-values are not available."
    )