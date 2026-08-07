"""
(VIX/100)^2  vs  21-day forward realized variance, 2004 -> present
Run locally: pip install yfinance pandas numpy matplotlib
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- fetch ---
spx = yf.Ticker("^GSPC").history(start="2004-01-01", auto_adjust=False)["Close"]
vix = yf.Ticker("^VIX").history(start="2004-01-01", auto_adjust=False)["Close"]

spx.index = spx.index.tz_localize(None)
vix.index = vix.index.tz_localize(None)

if spx.empty or vix.empty:
    raise RuntimeError("Download returned empty data — check internet connection / ticker validity.")

df = pd.DataFrame({"SPX": spx, "VIX": vix}).dropna()

# --- log returns, no de-meaning (per estimator convention) ---
df["r"] = np.log(df["SPX"] / df["SPX"].shift(1))

# --- forward realized variance over next 21 trading days, annualized ---
# RV_t = (252/21) * sum_{i=1}^{21} r_{t+i}^2
df["RV_fwd"] = (
    df["r"].shift(-1)
    .rolling(21)
    .apply(lambda x: np.sum(x**2), raw=True)
    .shift(-20) * (252 / 21)
)
# note: rolling(21) with shift(-1)+shift(-20) aligns the 21-day window
# to start the day AFTER t, matching VIX's forward-looking horizon

# The last 21 rows have no future window to sum over — they MUST stay NaN.
# This assertion guards against a future edit accidentally back-filling them.
assert df["RV_fwd"].tail(21).isna().all(), (
    "Last 21 rows of RV_fwd are not all NaN — forward window has been "
    "back-filled or misaligned. Fix before trusting any downstream stats."
)

df["VIX2"] = (df["VIX"] / 100) ** 2

plot_df = df[["VIX2", "RV_fwd"]].dropna()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(plot_df.index, plot_df["VIX2"], label=r"$(VIX_t/100)^2$", lw=0.9, color="#1f77b4")
ax.plot(plot_df.index, plot_df["RV_fwd"], label="21-day forward RV", lw=0.9, color="#d62728", alpha=0.75)
ax.set_ylabel("Annualized variance")
ax.set_title("Implied vs. Forward Realized Variance, S&P 500 (2004–present)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("vrp_chart.png", dpi=150)
plt.savefig("fig1_iv_vs_rv.png", dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# VRP analysis: variance units, vol points, distribution, regimes
# ============================================================

# --- variance-unit VRP ---
plot_df["VRP_var"] = plot_df["VIX2"] - plot_df["RV_fwd"]

# --- vol-point VRP: VIX (in %) minus realized vol (in %) ---
plot_df["RV_vol_fwd"] = np.sqrt(plot_df["RV_fwd"]) * 100
plot_df["VIX_level"] = np.sqrt(plot_df["VIX2"]) * 100
plot_df["VRP_vol"] = plot_df["VIX_level"] - plot_df["RV_vol_fwd"]

# Guard: VRP_vol must be a difference of two independently-computed vols,
# never sqrt(VIX2 - RV_fwd) applied to a pre-subtracted variance (undefined
# whenever VRP_var < 0 — a classic silent bug if this helper gets reused).
assert np.allclose(
    plot_df["VRP_vol"],
    np.sqrt(plot_df["VIX2"]) * 100 - np.sqrt(plot_df["RV_fwd"]) * 100,
    equal_nan=True,
), "VRP_vol is not a clean difference of vols — check for a shared-helper regression."

def summary_stats(s):
    return pd.Series({
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "skew": s.skew(),
        "pct_positive": (s > 0).mean() * 100,
    })

print("=" * 60)
print("VRP summary — variance units (VIX/100)^2 - RV_fwd")
print("=" * 60)
print(summary_stats(plot_df["VRP_var"]))

print()
print("=" * 60)
print("VRP summary — vol points, VIX - RV_vol_fwd")
print("=" * 60)
print(summary_stats(plot_df["VRP_vol"]))

# --- histogram: distribution shape (mass at small positive, long left tail) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(plot_df["VRP_var"], bins=100, color="#2ca02c", alpha=0.75)
axes[0].axvline(0, color="black", lw=0.8, ls="--")
axes[0].set_title("VRP distribution — variance units")
axes[0].set_xlabel(r"$(VIX_t/100)^2 - RV_{fwd}$")
axes[0].grid(alpha=0.3)

axes[1].hist(plot_df["VRP_vol"], bins=100, color="#9467bd", alpha=0.75)
axes[1].axvline(0, color="black", lw=0.8, ls="--")
axes[1].set_title("VRP distribution — vol points")
axes[1].set_xlabel(r"$VIX_t - RV^{vol}_{fwd}$")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vrp_histogram.png", dpi=150)
plt.show()

# --- sub-period regime table ---
regimes = [
    ("Pre-2008",     "2004-01-01", "2007-11-30"),
    ("GFC",          "2007-12-01", "2009-06-30"),   # NBER recession dates
    ("2009-2019",    "2009-07-01", "2019-12-31"),
    ("Covid",        "2020-01-01", "2021-12-31"),
    ("Post-2022",    "2022-01-01", plot_df.index.max().strftime("%Y-%m-%d")),
]

rows = []
for name, start, end in regimes:
    mask = (plot_df.index >= start) & (plot_df.index <= end)
    sub = plot_df.loc[mask]
    if len(sub) == 0:
        continue
    rows.append({
        "Regime": name,
        "N days": len(sub),
        "Mean VRP (var)": sub["VRP_var"].mean(),
        "Mean VRP (vol pts)": sub["VRP_vol"].mean(),
        "% positive": (sub["VRP_var"] > 0).mean() * 100,
    })

regime_table = pd.DataFrame(rows).set_index("Regime")
print()
print("=" * 60)
print("VRP by regime")
print("=" * 60)
print(regime_table.round(3))
regime_table.to_csv("vrp_regime_table.csv")
 