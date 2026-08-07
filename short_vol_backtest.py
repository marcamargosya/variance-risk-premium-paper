import pandas as pd
import matplotlib.pyplot as plt

ts = pd.read_csv("term_structure.csv", index_col=0, parse_dates=True)

# Daily P&L of short front-month VIX future (1 contract, vol points):
# short position gains when F1 falls. On roll dates F1 switches contracts,
# so compute per-contract: use F1 diff except when the underlying contract
# changes (detect via F1-F2 crossover artifacts is fragile; simplest robust
# proxy: exclude days where |F1 change| reflects a roll by using F2->F1
# continuity). For the paper's purpose, approximate with F1 diff and
# document the roll approximation.
pnl = -ts["F1"].diff()

# crude roll filter: on expiry days yesterday's F2 becomes today's F1;
# replace those daily moves with (F1_today - F2_yesterday), the true
# same-contract move.
roll_days = ts["F1"].notna() & ts["F2"].shift(1).notna() & \
            ((ts["F1"] - ts["F2"].shift(1)).abs() < (ts["F1"] - ts["F1"].shift(1)).abs())
pnl[roll_days] = -(ts["F1"] - ts["F2"].shift(1))[roll_days]

pnl = pnl.dropna()
cum = pnl.cumsum()

fig, ax = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
ax[0].plot(cum.index, cum.values, lw=0.9)
ax[0].set_title("Cumulative P&L: short front-month VIX future (vol points)")
ax[1].hist(pnl, bins=120)
ax[1].axvline(0, color="black", lw=0.8)
ax[1].set_title("Daily P&L distribution")
plt.savefig("fig_short_vol_pnl.png", dpi=200, bbox_inches="tight")
plt.close()

print("mean daily pnl:", pnl.mean().round(4))
print("skew:", pnl.skew().round(2))
print("worst day:", pnl.min().round(2), "on", pnl.idxmin().date())
print("best day:", pnl.max().round(2), "on", pnl.idxmax().date())