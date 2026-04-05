# Research Question:
# Does disciplinary efficiency matter? Is there a relationship between
# yellow card rate (YC/G) and win percentage, and do teams that commit
# more fouls but receive fewer cards outperform teams with high card rates?
#
# Definitions:
#   Foul rate        = Fouls per game (F/G)
#   Yellow card rate = Yellow cards per game (YC/G)
#   Card efficiency  = YC / Fouls (proportion of fouls that draw a card)
#                      Lower = more "disciplined" or "smart" fouling
#
# Methods:
#   - Pearson and Spearman correlation (YC/G vs WinPct, F/G vs WinPct,
#     Card Efficiency vs WinPct)
#   - OLS linear regression: WinPct ~ YC/G, WinPct ~ F/G,
#     WinPct ~ CardEfficiency, and multivariate models
#   - Threshold scan with two-proportion z-test, chi-square, and Fisher's
#     exact test on YC/G and CardEfficiency
#   - Logistic regression with grouped indicator variables
#   - Group comparison: High Fouls / Low Cards vs High Fouls / High Cards

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
fouls_path = "../Data/DA-proj-Team Stats - Fouls_Cards.csv"
results_path = "../Data/DA-proj-Results - Overall.csv"

fouls_df = pd.read_csv(fouls_path)

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
# 3. Merge and compute derived metrics
# ------------------------------------------------------------
df = pd.merge(fouls_df, records_df, on="Team", how="inner")

# Card efficiency: proportion of fouls that draw a yellow card
# Lower = more disciplined fouling (fewer cards per foul committed)
df["CardEfficiency"] = df["YC"] / df["Fo"]

df = df[["Team", "F/G", "YC/G", "RC/G", "Fo", "YC", "RC",
         "CardEfficiency", "W", "L", "T", "WinPct", "WinningRecord"]].copy()

print("=== Team-Level Dataset ===")
print(df.sort_values("CardEfficiency").to_string(index=False))

# ============================================================
# PART A: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PART A: CORRELATION ANALYSIS")
print("=" * 60)

metrics = {
    "F/G (Fouls/Game)": "F/G",
    "YC/G (Yellow Cards/Game)": "YC/G",
    "CardEfficiency (YC/Fouls)": "CardEfficiency"
}

for label, col in metrics.items():
    r_p, p_p = pearsonr(df[col], df["WinPct"])
    r_s, p_s = spearmanr(df[col], df["WinPct"])
    print(f"\n{label} vs WinPct:")
    print(f"  Pearson r  = {r_p:.4f}, p = {p_p:.4f}")
    print(f"  Spearman rho = {r_s:.4f}, p = {p_s:.4f}")

# ============================================================
# PART B: OLS REGRESSION MODELS
# ============================================================
print("\n" + "=" * 60)
print("PART B: OLS REGRESSION MODELS")
print("=" * 60)

# Model 1: WinPct ~ YC/G
print("\n--- Model 1: WinPct ~ YC/G ---")
X1 = sm.add_constant(df["YC/G"])
ols1 = sm.OLS(df["WinPct"], X1).fit()
print(ols1.summary())

# Model 2: WinPct ~ F/G
print("\n--- Model 2: WinPct ~ F/G ---")
X2 = sm.add_constant(df["F/G"])
ols2 = sm.OLS(df["WinPct"], X2).fit()
print(ols2.summary())

# Model 3: WinPct ~ CardEfficiency
print("\n--- Model 3: WinPct ~ CardEfficiency ---")
X3 = sm.add_constant(df["CardEfficiency"])
ols3 = sm.OLS(df["WinPct"], X3).fit()
print(ols3.summary())

# Model 4: WinPct ~ F/G + YC/G (multivariate)
print("\n--- Model 4: WinPct ~ F/G + YC/G ---")
X4 = sm.add_constant(df[["F/G", "YC/G"]])
ols4 = sm.OLS(df["WinPct"], X4).fit()
print(ols4.summary())

# Model 5: WinPct ~ F/G + CardEfficiency (multivariate)
print("\n--- Model 5: WinPct ~ F/G + CardEfficiency ---")
X5 = sm.add_constant(df[["F/G", "CardEfficiency"]])
ols5 = sm.OLS(df["WinPct"], X5).fit()
print(ols5.summary())

# ============================================================
# PART C: THRESHOLD SCAN ON YC/G
# ============================================================
print("\n" + "=" * 60)
print("PART C: THRESHOLD SCAN ON YC/G (Yellow Cards Per Game)")
print("=" * 60)

def run_threshold_scan(df, metric_col, metric_name):
    """Run threshold scan on a given metric column."""
    vals = np.sort(df[metric_col].unique())
    thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals) - 1)]

    results = []
    for thresh in thresholds:
        # For YC/G and CardEfficiency, LOW values should predict winning
        # So we test: teams BELOW threshold vs teams ABOVE threshold
        low = df[df[metric_col] < thresh]
        high = df[df[metric_col] >= thresh]

        if len(low) == 0 or len(high) == 0:
            continue

        low_wins = low["WinningRecord"].sum()
        high_wins = high["WinningRecord"].sum()

        counts = np.array([low_wins, high_wins])
        nobs = np.array([len(low), len(high)])

        # Z-test: low group > high group (fewer cards = more wins?)
        try:
            z_stat, z_p = proportions_ztest(count=counts, nobs=nobs, alternative="larger")
        except Exception:
            z_stat, z_p = np.nan, np.nan

        # Chi-square
        contingency = pd.crosstab(
            df[metric_col] < thresh,
            df["WinningRecord"]
        )
        try:
            chi2, chi_p, dof, expected = chi2_contingency(contingency)
        except Exception:
            chi2, chi_p = np.nan, np.nan

        # Fisher's exact
        try:
            fisher_or, fisher_p = fisher_exact(contingency, alternative="greater")
        except Exception:
            fisher_or, fisher_p = np.nan, np.nan

        results.append({
            "Threshold": thresh,
            "Low_n": len(low),
            "High_n": len(high),
            "Low_win_rate": low["WinningRecord"].mean(),
            "High_win_rate": high["WinningRecord"].mean(),
            "Rate_diff": low["WinningRecord"].mean() - high["WinningRecord"].mean(),
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

    print(f"\n=== Threshold Scan: {metric_name} ===")
    print(threshold_df.to_string(index=False))

    best = threshold_df.iloc[0]
    print(f"\nBest {metric_name} threshold: {best['Threshold']:.4f}")
    print(f"Below threshold win rate: {best['Low_win_rate']:.3f} ({int(best['Low_n'])} teams)")
    print(f"Above threshold win rate: {best['High_win_rate']:.3f} ({int(best['High_n'])} teams)")
    print(f"Z-statistic:          {best['Z_stat']:.4f}")
    print(f"One-sided z p-value:  {best['Z_p_value']:.4f}")
    print(f"Chi-square p-value:   {best['Chi_p_value']:.4f}")
    print(f"Fisher exact p-value: {best['Fisher_p_value']:.4f}")

    return best

best_ycg = run_threshold_scan(df, "YC/G", "YC/G")

# ============================================================
# PART D: THRESHOLD SCAN ON CARD EFFICIENCY
# ============================================================
print("\n" + "=" * 60)
print("PART D: THRESHOLD SCAN ON CARD EFFICIENCY (YC/Fouls)")
print("=" * 60)

best_ce = run_threshold_scan(df, "CardEfficiency", "CardEfficiency")

# ============================================================
# PART E: LOGISTIC REGRESSION WITH GROUPED INDICATORS
# ============================================================
print("\n" + "=" * 60)
print("PART E: LOGISTIC REGRESSION")
print("=" * 60)

# --- YC/G grouped indicator ---
best_ycg_thresh = best_ycg["Threshold"]
df["LowYCG"] = (df["YC/G"] < best_ycg_thresh).astype(int)

X_ycg = sm.add_constant(df["LowYCG"])
y = df["WinningRecord"]

# Check for perfect separation
low_grp = df[df["LowYCG"] == 1]["WinningRecord"]
high_grp = df[df["LowYCG"] == 0]["WinningRecord"]
perf_sep_ycg = (
    low_grp.nunique() == 1 and high_grp.nunique() == 1
    and low_grp.iloc[0] != high_grp.iloc[0]
)

if perf_sep_ycg:
    print("\n=== Note: Perfect Separation on YC/G ===")
    print("Using L2-penalized logistic regression.\n")
    logit_ycg = sm.Logit(y, X_ycg).fit_regularized(method="l1", alpha=0.5, disp=False)
    print("=== Penalized Logistic Regression: WinningRecord ~ LowYCG ===")
    print(f"  const = {logit_ycg.params.iloc[0]:.4f}")
    print(f"  LowYCG = {logit_ycg.params.iloc[1]:.4f}")
    df["PredProb_YCG"] = logit_ycg.predict(X_ycg)
else:
    try:
        logit_ycg = sm.Logit(y, X_ycg).fit(disp=False)
        print("\n=== Logistic Regression: WinningRecord ~ LowYCG ===")
        print(logit_ycg.summary())
        df["PredProb_YCG"] = logit_ycg.predict(X_ycg)
    except Exception:
        logit_ycg = sm.Logit(y, X_ycg).fit_regularized(method="l1", alpha=0.5, disp=False)
        print("\n=== Penalized Logistic Regression: WinningRecord ~ LowYCG ===")
        for name, coef in zip(X_ycg.columns, logit_ycg.params):
            print(f"  {name:<10} = {coef:.4f}")
        df["PredProb_YCG"] = logit_ycg.predict(X_ycg)

# --- CardEfficiency grouped indicator ---
best_ce_thresh = best_ce["Threshold"]
df["LowCE"] = (df["CardEfficiency"] < best_ce_thresh).astype(int)

X_ce = sm.add_constant(df["LowCE"])

low_ce_grp = df[df["LowCE"] == 1]["WinningRecord"]
high_ce_grp = df[df["LowCE"] == 0]["WinningRecord"]
perf_sep_ce = (
    low_ce_grp.nunique() == 1 and high_ce_grp.nunique() == 1
    and low_ce_grp.iloc[0] != high_ce_grp.iloc[0]
)

if perf_sep_ce:
    print("\n=== Note: Perfect Separation on CardEfficiency ===")
    print("Using L2-penalized logistic regression.\n")
    logit_ce = sm.Logit(y, X_ce).fit_regularized(method="l1", alpha=0.5, disp=False)
    print("=== Penalized Logistic Regression: WinningRecord ~ LowCE ===")
    print(f"  const = {logit_ce.params.iloc[0]:.4f}")
    print(f"  LowCE = {logit_ce.params.iloc[1]:.4f}")
    df["PredProb_CE"] = logit_ce.predict(X_ce)
else:
    try:
        logit_ce = sm.Logit(y, X_ce).fit(disp=False)
        print("\n=== Logistic Regression: WinningRecord ~ LowCE ===")
        print(logit_ce.summary())
        df["PredProb_CE"] = logit_ce.predict(X_ce)
    except Exception:
        logit_ce = sm.Logit(y, X_ce).fit_regularized(method="l1", alpha=0.5, disp=False)
        print("\n=== Penalized Logistic Regression: WinningRecord ~ LowCE ===")
        for name, coef in zip(X_ce.columns, logit_ce.params):
            print(f"  {name:<10} = {coef:.4f}")
        df["PredProb_CE"] = logit_ce.predict(X_ce)

# --- Continuous logistic: WinningRecord ~ YC/G ---
X_ycg_cont = sm.add_constant(df["YC/G"])
try:
    logit_ycg_cont = sm.Logit(y, X_ycg_cont).fit(disp=False)
    print("\n=== Logistic Regression: WinningRecord ~ YC/G (Continuous) ===")
    print(logit_ycg_cont.summary())
    df["PredProb_YCG_Cont"] = logit_ycg_cont.predict(X_ycg_cont)
except Exception:
    print("\n=== Logistic Regression: WinningRecord ~ YC/G (Continuous) ===")
    print("Standard MLE did not converge; using penalized regression.")
    logit_ycg_cont = sm.Logit(y, X_ycg_cont).fit_regularized(
        method="l1", alpha=0.5, disp=False
    )
    print(f"  const = {logit_ycg_cont.params.iloc[0]:.4f}")
    print(f"  YC/G  = {logit_ycg_cont.params.iloc[1]:.4f}")
    df["PredProb_YCG_Cont"] = logit_ycg_cont.predict(X_ycg_cont)

# ============================================================
# PART F: TACTICAL FOULING ANALYSIS
# High Fouls / Low Cards vs High Fouls / High Cards
# ============================================================
print("\n" + "=" * 60)
print("PART F: TACTICAL FOULING ANALYSIS")
print("=" * 60)

avg_fg = df["F/G"].mean()
avg_ce = df["CardEfficiency"].mean()

print(f"\nLeague average fouls/game:     {avg_fg:.2f}")
print(f"League average card efficiency: {avg_ce:.4f}")

df["HighFouls"] = (df["F/G"] >= avg_fg).astype(int)
df["HighCardRate"] = (df["CardEfficiency"] >= avg_ce).astype(int)

def assign_tactical_group(row):
    if row["HighFouls"] == 1 and row["HighCardRate"] == 0:
        return "Smart Foulers"
    elif row["HighFouls"] == 1 and row["HighCardRate"] == 1:
        return "Undisciplined"
    elif row["HighFouls"] == 0 and row["HighCardRate"] == 0:
        return "Clean & Disciplined"
    else:
        return "Low Fouls / High Cards"

df["TacticalGroup"] = df.apply(assign_tactical_group, axis=1)

print("\n=== Teams by Tactical Group ===")
print(df.sort_values(["TacticalGroup", "WinPct"], ascending=[True, False])[
    ["Team", "F/G", "YC/G", "CardEfficiency", "TacticalGroup", "WinPct", "WinningRecord"]
].to_string(index=False))

# Group summary
print("\n=== Tactical Group Summary ===")
group_summary = df.groupby("TacticalGroup").agg(
    Teams=("Team", "count"),
    Avg_WinPct=("WinPct", "mean"),
    Avg_FG=("F/G", "mean"),
    Avg_YCG=("YC/G", "mean"),
    Avg_CardEff=("CardEfficiency", "mean"),
    WinningRecords=("WinningRecord", "sum")
).reset_index()
print(group_summary.to_string(index=False))

# Direct comparison: Smart Foulers vs Undisciplined
smart = df[df["TacticalGroup"] == "Smart Foulers"]
undisciplined = df[df["TacticalGroup"] == "Undisciplined"]

if len(smart) > 0 and len(undisciplined) > 0:
    print("\n=== Direct Comparison: Smart Foulers vs Undisciplined ===")
    print(f"Smart Foulers ({len(smart)} teams): "
          f"{', '.join(smart['Team'].tolist())}")
    print(f"  Winning-record rate: {smart['WinningRecord'].mean():.3f}, "
          f"Avg WinPct: {smart['WinPct'].mean():.3f}")
    print(f"Undisciplined ({len(undisciplined)} teams): "
          f"{', '.join(undisciplined['Team'].tolist())}")
    print(f"  Winning-record rate: {undisciplined['WinningRecord'].mean():.3f}, "
          f"Avg WinPct: {undisciplined['WinPct'].mean():.3f}")

    # Z-test: Smart Foulers > Undisciplined
    counts_tf = np.array([
        smart["WinningRecord"].sum(),
        undisciplined["WinningRecord"].sum()
    ])
    nobs_tf = np.array([len(smart), len(undisciplined)])

    try:
        z_tf, p_tf = proportions_ztest(count=counts_tf, nobs=nobs_tf, alternative="larger")
        print(f"\n  Z-test (Smart > Undisciplined): z = {z_tf:.4f}, p = {p_tf:.4f}")
    except Exception:
        z_tf, p_tf = np.nan, np.nan
        print(f"\n  Z-test could not be computed.")

    # Fisher's exact on 2x2
    contingency_tf = pd.crosstab(
        df[df["TacticalGroup"].isin(["Smart Foulers", "Undisciplined"])]["TacticalGroup"] == "Smart Foulers",
        df[df["TacticalGroup"].isin(["Smart Foulers", "Undisciplined"])]["WinningRecord"]
    )
    try:
        f_or, f_p = fisher_exact(contingency_tf, alternative="greater")
        print(f"  Fisher's exact (Smart > Undisciplined): OR = {f_or:.4f}, p = {f_p:.4f}")
    except Exception:
        f_or, f_p = np.nan, np.nan

# ============================================================
# PART G: SUMMARY TABLES
# ============================================================
print("\n" + "=" * 60)
print("PART G: SUMMARY TABLES")
print("=" * 60)

print("\n=== Full Dataset with All Metrics ===")
print(df.sort_values("WinPct", ascending=False)[
    ["Team", "F/G", "YC/G", "CardEfficiency", "TacticalGroup",
     "W", "L", "T", "WinPct", "WinningRecord"]
].to_string(index=False))

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Correlation summary
print("\n--- Correlation Summary ---")
for label, col in metrics.items():
    r_p, p_p = pearsonr(df[col], df["WinPct"])
    sig = "significant" if p_p < 0.05 else "not significant"
    direction = "positive" if r_p > 0 else "negative"
    print(f"  {label}: r = {r_p:.4f} (p = {p_p:.4f}), {direction}, {sig}")

# OLS summary
print(f"\n--- OLS Summary ---")
print(f"  WinPct ~ YC/G:            R-sq = {ols1.rsquared:.4f}, "
      f"YC/G p = {ols1.pvalues['YC/G']:.4f}")
print(f"  WinPct ~ F/G:             R-sq = {ols2.rsquared:.4f}, "
      f"F/G p = {ols2.pvalues['F/G']:.4f}")
print(f"  WinPct ~ CardEfficiency:  R-sq = {ols3.rsquared:.4f}, "
      f"CE p = {ols3.pvalues['CardEfficiency']:.4f}")
print(f"  WinPct ~ F/G + YC/G:      R-sq = {ols4.rsquared:.4f}, Adj R-sq = {ols4.rsquared_adj:.4f}")
print(f"    F/G p = {ols4.pvalues['F/G']:.4f}, YC/G p = {ols4.pvalues['YC/G']:.4f}")
print(f"  WinPct ~ F/G + CardEff:   R-sq = {ols5.rsquared:.4f}, Adj R-sq = {ols5.rsquared_adj:.4f}")
print(f"    F/G p = {ols5.pvalues['F/G']:.4f}, CE p = {ols5.pvalues['CardEfficiency']:.4f}")

# Threshold summary
print(f"\n--- Threshold Summary ---")
print(f"  Best YC/G threshold: {best_ycg['Threshold']:.4f}")
print(f"    Below: {best_ycg['Low_win_rate']:.3f} win rate, "
      f"Above: {best_ycg['High_win_rate']:.3f} win rate")
print(f"    Z p = {best_ycg['Z_p_value']:.4f}, Chi-sq p = {best_ycg['Chi_p_value']:.4f}, "
      f"Fisher p = {best_ycg['Fisher_p_value']:.4f}")
print(f"  Best CardEfficiency threshold: {best_ce['Threshold']:.4f}")
print(f"    Below: {best_ce['Low_win_rate']:.3f} win rate, "
      f"Above: {best_ce['High_win_rate']:.3f} win rate")
print(f"    Z p = {best_ce['Z_p_value']:.4f}, Chi-sq p = {best_ce['Chi_p_value']:.4f}, "
      f"Fisher p = {best_ce['Fisher_p_value']:.4f}")

# Tactical fouling conclusion
if len(smart) > 0 and len(undisciplined) > 0:
    print(f"\n--- Tactical Fouling ---")
    print(f"  Smart Foulers (high fouls, low card rate): "
          f"avg WinPct = {smart['WinPct'].mean():.3f}")
    print(f"  Undisciplined (high fouls, high card rate): "
          f"avg WinPct = {undisciplined['WinPct'].mean():.3f}")
    if not np.isnan(p_tf):
        print(f"  Z-test p = {p_tf:.4f}, Fisher p = {f_p:.4f}")