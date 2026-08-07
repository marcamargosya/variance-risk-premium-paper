import pandas as pd
import matplotlib.pyplot as plt

ts = pd.read_csv("term_structure.csv", index_col=0, parse_dates=True)


# ---- Figure A: curve snapshots (calm vs crisis) ----
fig, ax = plt.subplots(figsize=(9, 5))
for date, label, color in [
    ("2017-07-21", "Calm (Jul 2017) — contango", "tab:blue"),
    ("2020-03-16", "Crisis (Mar 2020) — backwardation", "tab:red"),
]:
    row = ts.loc[date, ["F1","F2","F3","F4","F5","F6"]]
    ax.plot(range(1, 7), row.values, marker="o", label=label, color=color)
ax.set_xlabel("Contract rank (months to expiry)")
ax.set_ylabel("VIX futures settle")
ax.set_title("VIX futures term structure: contango vs. backwardation")
ax.legend()
plt.savefig("fig_term_structure_snapshots.png", dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure B: slope time series ----
slope = ts["F2"] - ts["F1"]
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(slope.index, slope.values, lw=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("F2 − F1 (vol points)")
ax.set_title("VIX futures slope: contango (>0) vs. backwardation (<0)")
plt.savefig("fig_term_structure_slope.png", dpi=200, bbox_inches="tight")
plt.close()

print("saved both figures")
print("share of days in contango:", (slope > 0).mean().round(3))