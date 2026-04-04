# Research Question:
# Is there a save-percentage threshold above which teams are significantly more likely
# to finish with a winning record?
 
import pandas as pd
import numpy as np
import re
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
import statsmodels.api as sm
 
# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
saves_path = "../Data/DA-proj-Team Stats - Saves.csv"
results_path = "../Data/DA-proj-Results - Overall.csv"
 
saves_df = pd.read_csv(saves_path)
 
# ------------------------------------------------------------
# 2. Parse team records from the Results - Overall file
#    Team record headers are embedded in column 3 (0-indexed)
#    e.g. "Columbia (3-8-4, 1-5-1)"
#    The first right-side section (rows before the first header)
#    belongs to Brown, whose header is missing from the file.
# ------------------------------------------------------------
results_raw = pd.read_csv(results_path, header=None)
 
team_records = []
pattern = re.compile(
    r"^(?P<team>[A-Za-z]+)\s*\((?P<w>\d+)-(?P<l>\d+)-(?P<t>\d+),"
)
 
# Scan column 3 for team record headers
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
 
# Brown's record header is missing from the file.
# Brown's results (rows 2-16, right-side columns) give: 6-7-2
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
# 3. Merge with Save% data
# ------------------------------------------------------------
df = pd.merge(saves_df, records_df, on="Team", how="inner")
 
# Keep only needed columns
df = df[["Team", "Save%", "W", "L", "T", "WinPct", "WinningRecord"]].copy()
 
print("\n=== Team-Level Dataset ===")
print(df.sort_values("Save%").to_string(index=False))
 
# ------------------------------------------------------------
# 4. Identify candidate Save% thresholds
#    We use midpoints between sorted unique Save% values
# ------------------------------------------------------------
save_vals = np.sort(df["Save%"].unique())
thresholds = [(save_vals[i] + save_vals[i+1]) / 2 for i in range(len(save_vals) - 1)]
 
# ------------------------------------------------------------
# 5. For each threshold:
#    - split teams into High Save% vs Low Save%
#    - run two-proportion z-test on winning-record rates
#    - run chi-square test
# ------------------------------------------------------------
results = []
 
for thresh in thresholds:
    high = df[df["Save%"] >= thresh]
    low = df[df["Save%"] < thresh]
 
    # Need both groups to have at least 1 team
    if len(high) == 0 or len(low) == 0:
        continue
 
    high_wins = high["WinningRecord"].sum()
    low_wins = low["WinningRecord"].sum()
 
    counts = np.array([high_wins, low_wins])
    nobs = np.array([len(high), len(low)])
 
    # Only run z-test if both groups have at least 1 win/loss possibility
    try:
        z_stat, z_p = proportions_ztest(count=counts, nobs=nobs, alternative="larger")
    except Exception:
        z_stat, z_p = np.nan, np.nan
 
    contingency = pd.crosstab(
        df["Save%"] >= thresh,
        df["WinningRecord"]
    )
 
    try:
        chi2, chi_p, dof, expected = chi2_contingency(contingency)
    except Exception:
        chi2, chi_p = np.nan, np.nan
 
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
        "Chi_p_value": chi_p
    })
 
threshold_df = pd.DataFrame(results)
 
# Sort by strongest evidence:
# prioritize lowest p-value, then biggest win-rate separation
threshold_df = threshold_df.sort_values(
    by=["Z_p_value", "Rate_diff"],
    ascending=[True, False]
).reset_index(drop=True)
 
print("\n=== Threshold Scan Results ===")
print(threshold_df.to_string(index=False))
 
best_row = threshold_df.iloc[0]
best_threshold = best_row["Threshold"]
 
print("\n=== Best Save% Threshold ===")
print(f"Best threshold: {best_threshold:.3f}")
print(f"High Save% win rate: {best_row['High_win_rate']:.3f}")
print(f"Low Save% win rate:  {best_row['Low_win_rate']:.3f}")
print(f"Difference:          {best_row['Rate_diff']:.3f}")
print(f"Z-statistic:         {best_row['Z_stat']:.3f}")
print(f"One-sided z p-value: {best_row['Z_p_value']:.4f}")
print(f"Chi-square p-value:  {best_row['Chi_p_value']:.4f}")
 
# ------------------------------------------------------------
# 6. Binary Logistic Regression:
#    WinningRecord ~ Save%
# ------------------------------------------------------------
X = sm.add_constant(df["Save%"])
y = df["WinningRecord"]
 
logit_model = sm.Logit(y, X).fit(disp=False)
 
print("\n=== Logistic Regression: WinningRecord ~ Save% ===")
print(logit_model.summary())
 
# Predicted probabilities at each team's Save%
df["PredictedProb_WinningRecord"] = logit_model.predict(X)
 
print("\n=== Predicted Probabilities from Logistic Regression ===")
print(df.sort_values("Save%")[["Team", "Save%", "WinningRecord", "PredictedProb_WinningRecord"]].to_string(index=False))
 
# ------------------------------------------------------------
# 7. Optional: interpret threshold groups clearly
# ------------------------------------------------------------
df["HighSaveGroup"] = (df["Save%"] >= best_threshold).astype(int)
 
print("\n=== Final Threshold Grouping ===")
print(df.sort_values(["HighSaveGroup", "Save%"], ascending=[False, False]).to_string(index=False))
 
# ------------------------------------------------------------
# 8. Plain-English conclusion template
# ------------------------------------------------------------
print("\n=== Interpretation Template ===")
if best_row["Z_p_value"] < 0.05:
    print(
        f"Teams with Save% >= {best_threshold:.3f} were significantly more likely "
        f"to finish with a winning record than teams below that threshold "
        f"(one-sided two-proportion z-test p = {best_row['Z_p_value']:.4f})."
    )
else:
    print(
        f"The threshold scan identified {best_threshold:.3f} as the strongest candidate, "
        f"but the difference in winning-record rates was not statistically significant "
        f"(one-sided two-proportion z-test p = {best_row['Z_p_value']:.4f})."
    )
 
if logit_model.pvalues["Save%"] < 0.05:
    print(
        f"The logistic regression also suggests Save% is a significant predictor of "
        f"whether a team finishes with a winning record (p = {logit_model.pvalues['Save%']:.4f})."
    )
else:
    print(
        f"The logistic regression does not show Save% as a statistically significant "
        f"predictor at the 0.05 level (p = {logit_model.pvalues['Save%']:.4f})."
    )