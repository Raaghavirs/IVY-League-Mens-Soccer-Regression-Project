# Research Question:
# Do teams with high shot accuracy but below-average shot volume outperform
# teams with high shot volume but below-average shot accuracy?
#
# Definitions:
#   Shot accuracy = Goals / Shots (scoring conversion rate)
#   Shot volume   = Shots per game (Avg/G from Shots file)
#   "High" / "Low" = above / below the league average for that metric
#
# Methods:
#   - Two-proportion z-test for grouped comparison
#   - Chi-square test of independence
#   - Fisher's exact test (small-sample robustness check)
#   - Logistic regression with grouped indicator variables

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
goals_path = "../Data/DA-proj-Team Stats - Goals.csv"
results_path = "../Data/DA-proj-Results - Overall.csv"

shots_df = pd.read_csv(shots_path)
goals_df = pd.read_csv(goals_path)

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
# 3. Build the merged dataset with derived metrics
# ------------------------------------------------------------
# Rename goals columns to avoid conflicts with shots columns
goals_df = goals_df.rename(columns={"NO": "Goals", "GP": "GP_goals"})

df = pd.merge(shots_df, goals_df[["Team", "Goals"]], on="Team", how="inner")
df = pd.merge(df, records_df, on="Team", how="inner")

# Compute shot accuracy: Goals / Shots
df["ShotAccuracy"] = df["Goals"] / df["Shots"]

# Shot volume per game is already in the Shots file as "Avg/G"
df = df.rename(columns={"Avg/G": "ShotsPerGame"})

# ------------------------------------------------------------
# 4. Define High / Low groups using league averages
# ------------------------------------------------------------
avg_accuracy = df["ShotAccuracy"].mean()
avg_volume = df["ShotsPerGame"].mean()

print(f"League average shot accuracy (Goals/Shots): {avg_accuracy:.4f}")
print(f"League average shot volume (Shots/G):       {avg_volume:.2f}")

df["HighAccuracy"] = (df["ShotAccuracy"] >= avg_accuracy).astype(int)
df["HighVolume"] = (df["ShotsPerGame"] >= avg_volume).astype(int)

# Assign groups
def assign_group(row):
    if row["HighAccuracy"] == 1 and row["HighVolume"] == 0:
        return "High Acc / Low Vol"
    elif row["HighAccuracy"] == 0 and row["HighVolume"] == 1:
        return "Low Acc / High Vol"
    elif row["HighAccuracy"] == 1 and row["HighVolume"] == 1:
        return "Both High"
    else:
        return "Both Low"

df["Group"] = df.apply(assign_group, axis=1)

print("\n=== Team-Level Dataset ===")
print(df[["Team", "ShotAccuracy", "ShotsPerGame", "HighAccuracy", "HighVolume",
          "Group", "W", "L", "T", "WinPct", "WinningRecord"]].sort_values(
    "ShotAccuracy", ascending=False
).to_string(index=False))

# ------------------------------------------------------------
# 5. Isolate the two comparison groups
# ------------------------------------------------------------
high_acc_low_vol = df[df["Group"] == "High Acc / Low Vol"]
low_acc_high_vol = df[df["Group"] == "Low Acc / High Vol"]

print("\n=== Group: High Accuracy / Low Volume ===")
print(high_acc_low_vol[["Team", "ShotAccuracy", "ShotsPerGame", "WinPct",
                         "WinningRecord"]].to_string(index=False))

print("\n=== Group: Low Accuracy / High Volume ===")
print(low_acc_high_vol[["Team", "ShotAccuracy", "ShotsPerGame", "WinPct",
                          "WinningRecord"]].to_string(index=False))

n_halv = len(high_acc_low_vol)
n_lahv = len(low_acc_high_vol)
wins_halv = high_acc_low_vol["WinningRecord"].sum()
wins_lahv = low_acc_high_vol["WinningRecord"].sum()

print(f"\nHigh Acc / Low Vol:  {n_halv} teams, {wins_halv} with winning records "
      f"({high_acc_low_vol['WinningRecord'].mean():.3f} rate)")
print(f"Low Acc / High Vol:  {n_lahv} teams, {wins_lahv} with winning records "
      f"({low_acc_high_vol['WinningRecord'].mean():.3f} rate)")

# ------------------------------------------------------------
# 6. Two-proportion z-test
#    H0: winning-record rate is equal in both groups
#    HA: Low Acc / High Vol has higher winning-record rate
#        (testing whether volume outperforms accuracy)
#
#    We test both directions and report both, then highlight
#    whichever is significant (or neither).
# ------------------------------------------------------------
counts = np.array([wins_lahv, wins_halv])
nobs = np.array([n_lahv, n_halv])

# Test: Low Acc / High Vol > High Acc / Low Vol
try:
    z_stat_vol, z_p_vol = proportions_ztest(
        count=counts, nobs=nobs, alternative="larger"
    )
except Exception:
    z_stat_vol, z_p_vol = np.nan, np.nan

# Test: High Acc / Low Vol > Low Acc / High Vol (reverse)
try:
    z_stat_acc, z_p_acc = proportions_ztest(
        count=np.array([wins_halv, wins_lahv]),
        nobs=np.array([n_halv, n_lahv]),
        alternative="larger"
    )
except Exception:
    z_stat_acc, z_p_acc = np.nan, np.nan

# Two-sided test
try:
    z_stat_two, z_p_two = proportions_ztest(
        count=counts, nobs=nobs, alternative="two-sided"
    )
except Exception:
    z_stat_two, z_p_two = np.nan, np.nan

print("\n=== Two-Proportion Z-Tests ===")
print(f"H_A: Volume group > Accuracy group:  z = {z_stat_vol:.4f}, p = {z_p_vol:.4f}")
print(f"H_A: Accuracy group > Volume group:  z = {z_stat_acc:.4f}, p = {z_p_acc:.4f}")
print(f"Two-sided test:                      z = {z_stat_two:.4f}, p = {z_p_two:.4f}")

# ------------------------------------------------------------
# 7. Chi-square test on the 2x2 contingency table
# ------------------------------------------------------------
comparison_df = pd.concat([high_acc_low_vol, low_acc_high_vol])
comparison_df["GroupLabel"] = comparison_df["Group"].apply(
    lambda x: 1 if x == "Low Acc / High Vol" else 0
)

contingency = pd.crosstab(
    comparison_df["GroupLabel"],
    comparison_df["WinningRecord"]
)

print("\n=== Contingency Table ===")
print(contingency.to_string())

try:
    chi2, chi_p, dof, expected = chi2_contingency(contingency, correction=False)
    print(f"\nChi-square (no continuity correction): {chi2:.4f}, p = {chi_p:.4f}")
    print(f"Expected cell counts:\n{pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_string()}")
except Exception as e:
    chi2, chi_p = np.nan, np.nan
    print(f"\nChi-square test could not be computed: {e}")

# With continuity correction (Yates')
try:
    chi2_yates, chi_p_yates, _, _ = chi2_contingency(contingency, correction=True)
    print(f"\nChi-square (Yates' correction):       {chi2_yates:.4f}, p = {chi_p_yates:.4f}")
except Exception:
    chi2_yates, chi_p_yates = np.nan, np.nan

# ------------------------------------------------------------
# 8. Fisher's exact test (robustness check)
# ------------------------------------------------------------
try:
    fisher_or, fisher_p = fisher_exact(contingency, alternative="two-sided")
    print(f"\nFisher's exact test (two-sided): OR = {fisher_or:.4f}, p = {fisher_p:.4f}")
except Exception:
    fisher_or, fisher_p = np.nan, np.nan

# One-sided: volume group > accuracy group
try:
    _, fisher_p_vol = fisher_exact(contingency, alternative="greater")
    print(f"Fisher's exact test (volume > accuracy): p = {fisher_p_vol:.4f}")
except Exception:
    fisher_p_vol = np.nan

# ------------------------------------------------------------
# 9. Logistic Regression with grouped indicator variables
#    on the FULL 8-team dataset
#
#    WinningRecord ~ HighAccuracy + HighVolume
#    This tests whether accuracy or volume (or both) predict
#    winning records when controlling for the other.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION ON FULL DATASET (n = 8)")
print("=" * 60)

X_full = sm.add_constant(df[["HighAccuracy", "HighVolume"]])
y = df["WinningRecord"]

try:
    logit_full = sm.Logit(y, X_full).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ HighAccuracy + HighVolume ===")
    print(logit_full.summary())
    df["PredProb_Full"] = logit_full.predict(X_full)
    full_converged = True
except Exception as e:
    print(f"\nStandard MLE did not converge: {e}")
    print("Using L2-penalized logistic regression.")
    logit_full = sm.Logit(y, X_full).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print("\n=== Penalized Logistic Regression: WinningRecord ~ HighAccuracy + HighVolume ===")
    print(f"Coefficients:")
    for name, coef in zip(X_full.columns, logit_full.params):
        print(f"  {name:<15} = {coef:.4f}")
    df["PredProb_Full"] = logit_full.predict(X_full)
    full_converged = False

# ------------------------------------------------------------
# 10. Logistic Regression with single group indicator
#     on the comparison subset only (High Acc/Low Vol vs
#     Low Acc/High Vol)
#
#     WinningRecord ~ IsVolumeGroup
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION ON COMPARISON SUBSET (n = {})".format(len(comparison_df)))
print("=" * 60)

comparison_df = comparison_df.copy()
X_comp = sm.add_constant(comparison_df["GroupLabel"])
y_comp = comparison_df["WinningRecord"]

# Check for perfect separation in comparison subset
vol_wins = comparison_df[comparison_df["GroupLabel"] == 1]["WinningRecord"]
acc_wins = comparison_df[comparison_df["GroupLabel"] == 0]["WinningRecord"]
perfect_sep = (
    vol_wins.nunique() == 1 and acc_wins.nunique() == 1
    and vol_wins.iloc[0] != acc_wins.iloc[0]
)

if perfect_sep:
    print("\n=== Note: Perfect Separation Detected ===")
    print("The group indicator perfectly separates winning from losing teams.")
    print("Using L2-penalized logistic regression.\n")
    logit_comp = sm.Logit(y_comp, X_comp).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print("=== Penalized Logistic Regression: WinningRecord ~ IsVolumeGroup ===")
    for name, coef in zip(X_comp.columns, logit_comp.params):
        print(f"  {name:<15} = {coef:.4f}")
    comparison_df["PredProb_Comp"] = logit_comp.predict(X_comp)
else:
    try:
        logit_comp = sm.Logit(y_comp, X_comp).fit(disp=False)
        print("\n=== Logistic Regression: WinningRecord ~ IsVolumeGroup ===")
        print(logit_comp.summary())
        comparison_df["PredProb_Comp"] = logit_comp.predict(X_comp)
    except Exception:
        print("\nStandard MLE did not converge; using penalized regression.")
        logit_comp = sm.Logit(y_comp, X_comp).fit_regularized(
            method="l1", alpha=0.5, disp=False
        )
        print("=== Penalized Logistic Regression: WinningRecord ~ IsVolumeGroup ===")
        for name, coef in zip(X_comp.columns, logit_comp.params):
            print(f"  {name:<15} = {coef:.4f}")
        comparison_df["PredProb_Comp"] = logit_comp.predict(X_comp)

# ------------------------------------------------------------
# 11. Summary tables
# ------------------------------------------------------------
print("\n=== Predicted Probabilities (Full Model) ===")
print(df.sort_values("Group")[
    ["Team", "Group", "ShotAccuracy", "ShotsPerGame", "WinningRecord", "PredProb_Full"]
].to_string(index=False))

print("\n=== Predicted Probabilities (Comparison Subset) ===")
print(comparison_df.sort_values("GroupLabel")[
    ["Team", "Group", "ShotAccuracy", "ShotsPerGame", "WinningRecord", "PredProb_Comp"]
].to_string(index=False))

# ------------------------------------------------------------
# 12. Winning percentage comparison (continuous, not just binary)
# ------------------------------------------------------------
print("\n=== Average Win Percentage by Group ===")
group_stats = df.groupby("Group").agg(
    Teams=("Team", "count"),
    Avg_WinPct=("WinPct", "mean"),
    Avg_Accuracy=("ShotAccuracy", "mean"),
    Avg_Volume=("ShotsPerGame", "mean"),
    WinningRecords=("WinningRecord", "sum")
).reset_index()
print(group_stats.to_string(index=False))

# ------------------------------------------------------------
# 13. Plain-English interpretation
# ------------------------------------------------------------
print("\n=== Interpretation ===")

# Describe the groups
print(
    f"Shot accuracy was defined as Goals/Shots (league avg: {avg_accuracy:.4f}). "
    f"Shot volume was defined as Shots/G (league avg: {avg_volume:.2f})."
)

halv_rate = high_acc_low_vol["WinningRecord"].mean()
lahv_rate = low_acc_high_vol["WinningRecord"].mean()
halv_pct = high_acc_low_vol["WinPct"].mean()
lahv_pct = low_acc_high_vol["WinPct"].mean()

print(
    f"\nHigh Accuracy / Low Volume teams ({n_halv} teams: "
    f"{', '.join(high_acc_low_vol['Team'].tolist())}): "
    f"winning-record rate = {halv_rate:.3f}, avg win% = {halv_pct:.3f}"
)
print(
    f"Low Accuracy / High Volume teams ({n_lahv} teams: "
    f"{', '.join(low_acc_high_vol['Team'].tolist())}): "
    f"winning-record rate = {lahv_rate:.3f}, avg win% = {lahv_pct:.3f}"
)

# Z-test conclusion
if z_p_vol < 0.05:
    print(
        f"\nThe two-proportion z-test indicates that High Volume / Low Accuracy teams "
        f"were significantly more likely to finish with a winning record "
        f"(z = {z_stat_vol:.4f}, one-sided p = {z_p_vol:.4f})."
    )
elif z_p_acc < 0.05:
    print(
        f"\nThe two-proportion z-test indicates that High Accuracy / Low Volume teams "
        f"were significantly more likely to finish with a winning record "
        f"(z = {z_stat_acc:.4f}, one-sided p = {z_p_acc:.4f})."
    )
else:
    print(
        f"\nThe two-proportion z-test does not show a statistically significant "
        f"difference in winning-record rates between the two groups "
        f"(two-sided p = {z_p_two:.4f})."
    )

# Chi-square caveat
if not np.isnan(chi_p):
    print(
        f"\nThe chi-square test returned p = {chi_p:.4f}. With only {n_halv + n_lahv} "
        f"teams in the comparison, expected cell counts fall below 5, which violates "
        f"a key assumption of the chi-square test and reduces its reliability."
    )

# Fisher's exact test
if not np.isnan(fisher_p):
    print(
        f"Fisher's exact test (two-sided p = {fisher_p:.4f}), which does not rely "
        f"on large-sample approximations, provides a more trustworthy result for "
        f"this sample size."
    )

# Logistic regression
if perfect_sep:
    print(
        f"\nThe logistic regression on the comparison subset encountered perfect "
        f"separation. A penalized regression was used to produce finite estimates."
    )

print(
    f"\nThe full-dataset logistic regression (WinningRecord ~ HighAccuracy + HighVolume) "
    f"tests whether accuracy or volume independently predicts winning records when "
    f"controlling for the other."
)

if full_converged:
    acc_pval = logit_full.pvalues.get("HighAccuracy", np.nan)
    vol_pval = logit_full.pvalues.get("HighVolume", np.nan)
    print(f"  HighAccuracy p-value: {acc_pval:.4f}")
    print(f"  HighVolume p-value:   {vol_pval:.4f}")
    if vol_pval < 0.05 and acc_pval >= 0.05:
        print("  -> Shot volume is a significant predictor; shot accuracy is not.")
    elif acc_pval < 0.05 and vol_pval >= 0.05:
        print("  -> Shot accuracy is a significant predictor; shot volume is not.")
    elif acc_pval < 0.05 and vol_pval < 0.05:
        print("  -> Both shot accuracy and shot volume are significant predictors.")
    else:
        print(
            "  -> Neither reaches significance individually, likely due to the "
            "small sample size (n = 8)."
        )
else:
    print(
        "  Standard MLE did not converge; penalized estimates suggest the direction "
        "of the relationship but p-values are not reliable."
    )