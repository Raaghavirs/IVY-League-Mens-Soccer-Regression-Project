# Research Question:
# Is goalkeeper workload an inverse predictor of team success? Do teams
# whose goalkeepers face fewer saves per game perform better, and does
# this reflect defensive system quality rather than individual goalkeeper
# ability?
#
# Definitions:
#   Keeper Workload  = Saves per game (Sa/G) - how busy the keeper is
#   Save%            = Save percentage - individual keeper quality
#   GA/G             = Goals against per game - defensive outcome
#   Sho/G            = Shutouts per game - clean sheet rate
#
# Core hypothesis: A low-workload keeper on a winning team reflects a
# defensive system that prevents shots from reaching the goalkeeper,
# rather than a goalkeeper who simply makes fewer saves because they
# face fewer shots. If true, Sa/G should predict winning better than
# (or independently of) Save%, and the combination of low workload
# with high save% should identify the best defensive systems.
#
# Methods:
#   - Pearson and Spearman correlation for all defensive metrics
#   - OLS regression: WinPct ~ Sa/G, WinPct ~ Save%, multivariate
#   - Threshold scan on Sa/G with z-test, chi-square, Fisher's exact
#   - Logistic regression with grouped indicator and continuous predictor
#   - Comparative analysis: workload vs save% vs GA/G as predictors
#   - Defensive profile grouping: Low Workload + High Save% vs other combos

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
saves_path = "../Data/DA-proj-Team Stats - Saves.csv"
ga_path = "../Data/DA-proj-Team Stats - Goals Allowed.csv"
results_path = "../Data/DA-proj-Results - Overall.csv"

saves_df = pd.read_csv(saves_path)
ga_df = pd.read_csv(ga_path)

# ------------------------------------------------------------
# 2. Parse team records
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
            "Team": team, "W": w, "L": l, "T": t,
            "GP_results": gp, "WinPct": win_pct,
            "WinningRecord": winning_record
        })

brown_w, brown_l, brown_t = 6, 7, 2
brown_gp = brown_w + brown_l + brown_t
brown_pct = (brown_w + 0.5 * brown_t) / brown_gp
team_records.append({
    "Team": "Brown", "W": brown_w, "L": brown_l, "T": brown_t,
    "GP_results": brown_gp, "WinPct": brown_pct,
    "WinningRecord": 1 if brown_pct > 0.500 else 0
})

records_df = pd.DataFrame(team_records)

# ------------------------------------------------------------
# 3. Merge datasets
# ------------------------------------------------------------
ga_df = ga_df.rename(columns={"GA Avg/G": "GA/G"})

df = pd.merge(saves_df, records_df, on="Team", how="inner")
df = pd.merge(df, ga_df[["Team", "GA", "GA/G"]], on="Team", how="inner")

# Rename for clarity
df = df.rename(columns={"Sa/G": "SavesPerGame", "Save%": "SavePct",
                         "Sho/G": "ShutoutsPerGame"})

# Compute shots faced per game (saves + goals allowed per game)
df["ShotsFacedPerGame"] = df["SavesPerGame"] + df["GA/G"]

df = df[["Team", "SavesPerGame", "SavePct", "ShutoutsPerGame",
         "GA/G", "GA", "Saves", "ShotsFacedPerGame",
         "W", "L", "T", "WinPct", "WinningRecord"]].copy()

print("=== Team-Level Dataset ===")
print(df.sort_values("SavesPerGame").to_string(index=False))

# ============================================================
# PART A: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PART A: CORRELATION ANALYSIS")
print("=" * 60)

metrics = {
    "SavesPerGame (Keeper Workload)": "SavesPerGame",
    "SavePct (Individual Quality)": "SavePct",
    "GA/G (Defensive Outcome)": "GA/G",
    "ShutoutsPerGame (Clean Sheets)": "ShutoutsPerGame",
    "ShotsFacedPerGame (Total Pressure)": "ShotsFacedPerGame"
}

print(f"\n{'Metric':<40} {'Pearson r':<12} {'p-value':<10} {'Spearman':<12} {'p-value':<10}")
print("-" * 84)

for label, col in metrics.items():
    r_p, p_p = pearsonr(df[col], df["WinPct"])
    r_s, p_s = spearmanr(df[col], df["WinPct"])
    print(f"{label:<40} {r_p:>8.4f}     {p_p:>8.4f}   {r_s:>8.4f}     {p_s:>8.4f}")

# ============================================================
# PART B: OLS REGRESSION MODELS
# ============================================================
print("\n" + "=" * 60)
print("PART B: OLS REGRESSION MODELS")
print("=" * 60)

# Model 1: WinPct ~ SavesPerGame (workload alone)
print("\n--- Model 1: WinPct ~ SavesPerGame ---")
X1 = sm.add_constant(df["SavesPerGame"])
ols1 = sm.OLS(df["WinPct"], X1).fit()
print(ols1.summary())

# Model 2: WinPct ~ SavePct (individual quality alone)
print("\n--- Model 2: WinPct ~ SavePct ---")
X2 = sm.add_constant(df["SavePct"])
ols2 = sm.OLS(df["WinPct"], X2).fit()
print(ols2.summary())

# Model 3: WinPct ~ GA/G (defensive outcome alone)
print("\n--- Model 3: WinPct ~ GA/G ---")
X3 = sm.add_constant(df["GA/G"])
ols3 = sm.OLS(df["WinPct"], X3).fit()
print(ols3.summary())

# Model 4: WinPct ~ SavesPerGame + SavePct (workload + quality)
print("\n--- Model 4: WinPct ~ SavesPerGame + SavePct ---")
X4 = sm.add_constant(df[["SavesPerGame", "SavePct"]])
ols4 = sm.OLS(df["WinPct"], X4).fit()
print(ols4.summary())

# Model 5: WinPct ~ ShutoutsPerGame (clean sheets)
print("\n--- Model 5: WinPct ~ ShutoutsPerGame ---")
X5 = sm.add_constant(df["ShutoutsPerGame"])
ols5 = sm.OLS(df["WinPct"], X5).fit()
print(ols5.summary())

# ============================================================
# PART C: PREDICTIVE POWER COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("PART C: PREDICTIVE POWER COMPARISON")
print("=" * 60)

print(f"\n{'Model':<40} {'R-squared':<12} {'Adj R-sq':<12} {'Predictor p':<12}")
print("-" * 76)
print(f"{'WinPct ~ SavesPerGame':<40} {ols1.rsquared:>8.4f}     {ols1.rsquared_adj:>8.4f}     {ols1.pvalues['SavesPerGame']:>8.4f}")
print(f"{'WinPct ~ SavePct':<40} {ols2.rsquared:>8.4f}     {ols2.rsquared_adj:>8.4f}     {ols2.pvalues['SavePct']:>8.4f}")
print(f"{'WinPct ~ GA/G':<40} {ols3.rsquared:>8.4f}     {ols3.rsquared_adj:>8.4f}     {ols3.pvalues['GA/G']:>8.4f}")
print(f"{'WinPct ~ ShutoutsPerGame':<40} {ols5.rsquared:>8.4f}     {ols5.rsquared_adj:>8.4f}     {ols5.pvalues['ShutoutsPerGame']:>8.4f}")
print(f"{'WinPct ~ SavesPerGame + SavePct':<40} {ols4.rsquared:>8.4f}     {ols4.rsquared_adj:>8.4f}     {'(multi)':>8}")

# ============================================================
# PART D: THRESHOLD SCAN ON SAVES PER GAME
# ============================================================
print("\n" + "=" * 60)
print("PART D: THRESHOLD SCAN ON SAVES PER GAME")
print("=" * 60)

sg_vals = np.sort(df["SavesPerGame"].unique())
thresholds = [(sg_vals[i] + sg_vals[i+1]) / 2 for i in range(len(sg_vals) - 1)]

results = []

for thresh in thresholds:
    # Low workload (below threshold) should predict winning
    low = df[df["SavesPerGame"] < thresh]
    high = df[df["SavesPerGame"] >= thresh]

    if len(low) == 0 or len(high) == 0:
        continue

    counts = np.array([low["WinningRecord"].sum(), high["WinningRecord"].sum()])
    nobs = np.array([len(low), len(high)])

    try:
        z_stat, z_p = proportions_ztest(count=counts, nobs=nobs, alternative="larger")
    except Exception:
        z_stat, z_p = np.nan, np.nan

    contingency = pd.crosstab(df["SavesPerGame"] < thresh, df["WinningRecord"])
    try:
        chi2, chi_p, _, _ = chi2_contingency(contingency)
    except Exception:
        chi2, chi_p = np.nan, np.nan

    try:
        fisher_or, fisher_p = fisher_exact(contingency, alternative="greater")
    except Exception:
        fisher_or, fisher_p = np.nan, np.nan

    results.append({
        "Threshold": thresh,
        "Low_n": len(low), "High_n": len(high),
        "Low_win_rate": low["WinningRecord"].mean(),
        "High_win_rate": high["WinningRecord"].mean(),
        "Rate_diff": low["WinningRecord"].mean() - high["WinningRecord"].mean(),
        "Low_avg_wpct": low["WinPct"].mean(),
        "High_avg_wpct": high["WinPct"].mean(),
        "Z_stat": z_stat, "Z_p_value": z_p,
        "Chi_p": chi_p, "Fisher_p": fisher_p
    })

threshold_df = pd.DataFrame(results)
threshold_df = threshold_df.sort_values(
    by=["Z_p_value", "Rate_diff"], ascending=[True, False]
).reset_index(drop=True)

print("\n=== Threshold Scan Results ===")
print(threshold_df.to_string(index=False))

best = threshold_df.iloc[0]
best_thresh = best["Threshold"]

print(f"\n=== Best SavesPerGame Threshold ===")
print(f"Threshold: {best_thresh:.3f}")
print(f"Below (low workload): {best['Low_win_rate']:.3f} win rate "
      f"({int(best['Low_n'])} teams), avg WinPct = {best['Low_avg_wpct']:.3f}")
print(f"Above (high workload): {best['High_win_rate']:.3f} win rate "
      f"({int(best['High_n'])} teams), avg WinPct = {best['High_avg_wpct']:.3f}")
print(f"Z-test p:     {best['Z_p_value']:.4f}")
print(f"Chi-square p: {best['Chi_p']:.4f}")
print(f"Fisher p:     {best['Fisher_p']:.4f}")

# ============================================================
# PART E: LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("PART E: LOGISTIC REGRESSION")
print("=" * 60)

# Grouped indicator
df["LowWorkload"] = (df["SavesPerGame"] < best_thresh).astype(int)
X_grp = sm.add_constant(df["LowWorkload"])
y = df["WinningRecord"]

grp1 = df[df["LowWorkload"] == 1]["WinningRecord"]
grp0 = df[df["LowWorkload"] == 0]["WinningRecord"]
perfect_sep = (
    grp1.nunique() == 1 and grp0.nunique() == 1
    and grp1.iloc[0] != grp0.iloc[0]
)

if perfect_sep:
    print("\n=== Note: Perfect Separation on LowWorkload ===")
    print("Using L2-penalized logistic regression.\n")
    logit_grp = sm.Logit(y, X_grp).fit_regularized(method="l1", alpha=0.5, disp=False)
    print("=== Penalized Logistic Regression: WinningRecord ~ LowWorkload ===")
    print(f"  const       = {logit_grp.params.iloc[0]:.4f}")
    print(f"  LowWorkload = {logit_grp.params.iloc[1]:.4f}")
    df["PredProb_Grouped"] = logit_grp.predict(X_grp)
else:
    try:
        logit_grp = sm.Logit(y, X_grp).fit(disp=False)
        print("\n=== Logistic Regression: WinningRecord ~ LowWorkload ===")
        print(logit_grp.summary())
        df["PredProb_Grouped"] = logit_grp.predict(X_grp)
    except Exception:
        logit_grp = sm.Logit(y, X_grp).fit_regularized(method="l1", alpha=0.5, disp=False)
        print("\n=== Penalized Logistic Regression: WinningRecord ~ LowWorkload ===")
        for name, coef in zip(X_grp.columns, logit_grp.params):
            print(f"  {name:<15} = {coef:.4f}")
        df["PredProb_Grouped"] = logit_grp.predict(X_grp)

# Continuous logistic
X_cont = sm.add_constant(df["SavesPerGame"])
try:
    logit_cont = sm.Logit(y, X_cont).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ SavesPerGame (Continuous) ===")
    print(logit_cont.summary())
    df["PredProb_Cont"] = logit_cont.predict(X_cont)
except Exception:
    print("\n=== Logistic Regression: WinningRecord ~ SavesPerGame (Continuous) ===")
    print("Standard MLE did not converge; using penalized regression.")
    logit_cont = sm.Logit(y, X_cont).fit_regularized(method="l1", alpha=0.5, disp=False)
    print(f"  const        = {logit_cont.params.iloc[0]:.4f}")
    print(f"  SavesPerGame = {logit_cont.params.iloc[1]:.4f}")
    df["PredProb_Cont"] = logit_cont.predict(X_cont)

# ============================================================
# PART F: DEFENSIVE PROFILE ANALYSIS
# Workload vs Individual Quality (Save%)
# ============================================================
print("\n" + "=" * 60)
print("PART F: DEFENSIVE PROFILE ANALYSIS")
print("=" * 60)

avg_workload = df["SavesPerGame"].mean()
avg_savepct = df["SavePct"].mean()

print(f"\nLeague average saves/game:  {avg_workload:.2f}")
print(f"League average save%:      {avg_savepct:.3f}")

def assign_defensive_profile(row):
    low_wl = row["SavesPerGame"] < avg_workload
    high_sv = row["SavePct"] >= avg_savepct
    if low_wl and high_sv:
        return "Elite System"
    elif low_wl and not high_sv:
        return "Protected but Porous"
    elif not low_wl and high_sv:
        return "Busy but Reliable"
    else:
        return "Exposed"

df["DefensiveProfile"] = df.apply(assign_defensive_profile, axis=1)

print("\n=== Teams by Defensive Profile ===")
print(df.sort_values(["DefensiveProfile", "WinPct"], ascending=[True, False])[
    ["Team", "SavesPerGame", "SavePct", "GA/G", "ShutoutsPerGame",
     "DefensiveProfile", "WinPct", "WinningRecord"]
].to_string(index=False))

# Profile summary
print("\n=== Defensive Profile Summary ===")
profile_summary = df.groupby("DefensiveProfile").agg(
    Teams=("Team", "count"),
    Avg_WinPct=("WinPct", "mean"),
    Avg_Workload=("SavesPerGame", "mean"),
    Avg_SavePct=("SavePct", "mean"),
    Avg_GA=("GA/G", "mean"),
    WinningRecords=("WinningRecord", "sum")
).reset_index()
print(profile_summary.to_string(index=False))

# ============================================================
# PART G: SYSTEM VS INDIVIDUAL - WHICH MATTERS MORE?
# ============================================================
print("\n" + "=" * 60)
print("PART G: SYSTEM QUALITY vs INDIVIDUAL QUALITY")
print("=" * 60)

# Compare: does workload (system) predict winning AFTER controlling
# for save% (individual quality)?
print("\n--- Multivariate Model: WinPct ~ SavesPerGame + SavePct ---")
print(f"R-squared: {ols4.rsquared:.4f} (Adj: {ols4.rsquared_adj:.4f})")
print(f"SavesPerGame coefficient: {ols4.params['SavesPerGame']:.4f} "
      f"(p = {ols4.pvalues['SavesPerGame']:.4f})")
print(f"SavePct coefficient:      {ols4.params['SavePct']:.4f} "
      f"(p = {ols4.pvalues['SavePct']:.4f})")

# Predicted win percentages
df["PredWinPct_Workload"] = ols1.predict(sm.add_constant(df["SavesPerGame"]))
df["PredWinPct_SavePct"] = ols2.predict(sm.add_constant(df["SavePct"]))

print("\n=== Predicted Win Percentages by Model ===")
print(df.sort_values("SavesPerGame")[
    ["Team", "SavesPerGame", "SavePct", "WinPct",
     "PredWinPct_Workload", "PredWinPct_SavePct"]
].to_string(index=False))

# ============================================================
# PART H: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PART H: FINAL SUMMARY")
print("=" * 60)

print("\n=== Complete Dataset ===")
print(df.sort_values("SavesPerGame")[
    ["Team", "SavesPerGame", "SavePct", "GA/G", "ShutoutsPerGame",
     "DefensiveProfile", "W", "L", "T", "WinPct", "WinningRecord"]
].to_string(index=False))

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Correlation comparison
r_wl, p_wl = pearsonr(df["SavesPerGame"], df["WinPct"])
r_sv, p_sv = pearsonr(df["SavePct"], df["WinPct"])
r_ga, p_ga = pearsonr(df["GA/G"], df["WinPct"])
r_sh, p_sh = pearsonr(df["ShutoutsPerGame"], df["WinPct"])

print(f"\n--- Correlation Comparison ---")
print(f"  SavesPerGame vs WinPct:    r = {r_wl:.4f} (p = {p_wl:.4f})")
print(f"  SavePct vs WinPct:         r = {r_sv:.4f} (p = {p_sv:.4f})")
print(f"  GA/G vs WinPct:            r = {r_ga:.4f} (p = {p_ga:.4f})")
print(f"  ShutoutsPerGame vs WinPct: r = {r_sh:.4f} (p = {p_sh:.4f})")

print(f"\n--- OLS Comparison ---")
print(f"  SavesPerGame R-sq:    {ols1.rsquared:.4f} (p = {ols1.pvalues['SavesPerGame']:.4f})")
print(f"  SavePct R-sq:         {ols2.rsquared:.4f} (p = {ols2.pvalues['SavePct']:.4f})")
print(f"  GA/G R-sq:            {ols3.rsquared:.4f} (p = {ols3.pvalues['GA/G']:.4f})")
print(f"  ShutoutsPerGame R-sq: {ols5.rsquared:.4f} (p = {ols5.pvalues['ShutoutsPerGame']:.4f})")

print(f"\n--- Threshold ---")
print(f"  Best Sa/G threshold: {best_thresh:.3f}")
print(f"  Z-test p = {best['Z_p_value']:.4f}, Fisher p = {best['Fisher_p']:.4f}")

print(f"\n--- Key Insight ---")
print(f"  SavesPerGame explains {ols1.rsquared:.1%} of win% variance")
print(f"  SavePct explains {ols2.rsquared:.1%} of win% variance")
print(f"  Workload is a {'stronger' if ols1.rsquared > ols2.rsquared else 'weaker'} "
      f"predictor than individual save quality")