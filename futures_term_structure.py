"""
Stitch CFE VX futures contract CSVs into a term-structure table.
Rows = trade date, columns = F1 (front month), F2, F3, ... (ranked by expiry).
Monthly contracts only (weekly VX futures are filtered out).
"""
import glob
import os
import pandas as pd

DATA_DIR = "data"


def monthly_expiries(start="2013-01-01", end="2027-12-31"):
    """Approximate monthly VIX futures expiries: 30 days before the
    3rd Friday of the FOLLOWING month (so iterate months and step back)."""
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        fri = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")[2]
        out.append((fri - pd.Timedelta(days=30)).date())
    return set(out)


EXP_SET = monthly_expiries()


def is_monthly_vx(path):
    """VX_YYYY-MM-DD.csv is monthly iff its filename date is a standard
    monthly expiry (±1 day for holiday shifts)."""
    fname = os.path.basename(path)
    d = pd.Timestamp(fname[3:13]).date()
    return any(abs((d - e).days) <= 3 for e in EXP_SET)


# ------------------------------------------------------------------
# 1. Load every contract CSV, tag with its expiry
# ------------------------------------------------------------------
all_contracts = []

for path in sorted(glob.glob(f"{DATA_DIR}/*.csv")):
    fname = os.path.basename(path)
    is_vx = fname.startswith("VX_")
    

    # Skip weekly VX contracts
    if is_vx and not is_monthly_vx(path):
        continue

    # Some CBOE files have a disclaimer paragraph before the real header row.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Trade Date")), None)
    if header_idx is None:
        print(f"SKIPPING (no header found): {path}")
        continue

    df = pd.read_csv(path, skiprows=header_idx)
    df["Trade Date"] = pd.to_datetime(df["Trade Date"])
    df = df[(df["Settle"] > 0) & (df["Settle"] < 100)]
    if df.empty:
        continue

    if is_vx:
        # Filename IS the expiry date — works for expired AND live contracts.
        expiry = pd.Timestamp(fname[3:13])
    else:
        # CFE-era file (2007-2013): all long expired, so expiry is simply
        # the contract's own last trade date.
        expiry = df["Trade Date"].max()

    df["expiry"] = expiry
    df["source_file"] = path
    all_contracts.append(df[["Trade Date", "expiry", "Settle", "source_file"]])

# ------------------------------------------------------------------
# 2. Concatenate; drop duplicate (date, expiry) pairs from the
#    2013 CFE/VX overlap year
# ------------------------------------------------------------------
long_df = pd.concat(all_contracts, ignore_index=True)
long_df["exp_month"] = long_df["expiry"].dt.to_period("M")
long_df = long_df.drop_duplicates(subset=["Trade Date", "exp_month"], keep="first")
print(f"Loaded {long_df['source_file'].nunique()} contracts, {len(long_df)} rows")

# ------------------------------------------------------------------
# 3. Rank contracts by expiry per trade date; drop expired-on-the-day rows
# ------------------------------------------------------------------
long_df = long_df[long_df["expiry"] > long_df["Trade Date"]]
long_df = long_df.sort_values(["Trade Date", "expiry"])
long_df["rank"] = long_df.groupby("Trade Date")["expiry"].rank(method="first").astype(int)
long_df["F_label"] = "F" + long_df["rank"].astype(str)

# ------------------------------------------------------------------
# 4. Pivot to wide table
# ------------------------------------------------------------------
term_structure = long_df.pivot_table(
    index="Trade Date", columns="F_label", values="Settle", aggfunc="first"
)
term_structure = term_structure[sorted(term_structure.columns, key=lambda x: int(x[1:]))]
term_structure = term_structure.sort_index()

print(term_structure.head(5))
print(term_structure.tail(5))
term_structure.to_csv("term_structure.csv")
print("Saved to term_structure.csv")

# ------------------------------------------------------------------
# 5. Sanity check: adjacent monthly expiries ~1 month apart
# ------------------------------------------------------------------
expiry_by_rank = long_df.pivot_table(
    index="Trade Date", columns="F_label", values="expiry", aggfunc="first"
)
gap_days = (expiry_by_rank["F2"] - expiry_by_rank["F1"]).dt.days
bad_gaps = gap_days[(gap_days > 45) | (gap_days < 20)]
if bad_gaps.empty:
    print("Gap check PASSED: all F2-F1 expiry gaps within 20-45 days.")
else:
    print(f"Gap check FAILED on {len(bad_gaps)} dates")
    print("failures per year:")
    print(bad_gaps.groupby(bad_gaps.index.year).size())
    print("example dates:", [d.date() for d in bad_gaps.index[:5]])
KNOWN_GAP_DATES = set(bad_gaps.index)
known_2013_seam = all(
    pd.Timestamp("2013-03-15") <= d <= pd.Timestamp("2013-04-17")
    for d in KNOWN_GAP_DATES
)
assert bad_gaps.empty or known_2013_seam, "Unexpected gaps beyond the known Mar-2013 seam"
if known_2013_seam and not bad_gaps.empty:
    print(f"Accepted known data limitation: {len(bad_gaps)} dates in Mar 2013 "
          "where the May-2013 contract has no valid settle (CFE file absent, "
          "VX file zero-filled). Slope series will be NaN there.")