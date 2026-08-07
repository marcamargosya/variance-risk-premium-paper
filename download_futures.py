"""
Stitch individual CFE VX futures contract CSVs into a term-structure table.
Rows = trade date, columns = F1 (front month), F2, F3, ... (ranked by expiry).
"""
import glob
import pandas as pd

DATA_DIR = "data"

# ------------------------------------------------------------------
# 1. Load every contract CSV, tag with its expiry
# ------------------------------------------------------------------
all_contracts = []

for path in sorted(glob.glob(f"{DATA_DIR}/*.csv")):
    # Some CBOE files have a disclaimer paragraph before the real header row.
    # Find the actual header line (starts with "Trade Date") and skip to it.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Trade Date")), None)
    if header_idx is None:
        print(f"SKIPPING (no header found): {path}")
        continue

    df = pd.read_csv(path, skiprows=header_idx)
    df["Trade Date"] = pd.to_datetime(df["Trade Date"])

    # Keep only rows with a real settle price (drops any pre-listing junk, if present)
    df = df[df["Settle"] > 0]

    if df.empty:
        continue

    # Expiry = the contract's own last trade date (the settlement/expiration row)
    expiry = df["Trade Date"].max()
    df["expiry"] = expiry
    df["source_file"] = path

    all_contracts.append(df[["Trade Date", "expiry", "Settle", "source_file"]])

# ------------------------------------------------------------------
# 2. Concatenate into one long dataframe
# ------------------------------------------------------------------
long_df = pd.concat(all_contracts, ignore_index=True)
long_df = long_df[long_df["Settle"] > 0]  # redundant safety filter, cheap to keep

print(f"Loaded {long_df['source_file'].nunique()} contracts, {len(long_df)} total rows")

# ------------------------------------------------------------------
# 3. Rank contracts by expiry, per trade date
# ------------------------------------------------------------------
long_df = long_df.sort_values(["Trade Date", "expiry"])

long_df["rank"] = long_df.groupby("Trade Date")["expiry"].rank(method="first").astype(int)
long_df["F_label"] = "F" + long_df["rank"].astype(str)

# ------------------------------------------------------------------
# 4. Pivot to wide term-structure table
# ------------------------------------------------------------------
term_structure = long_df.pivot_table(
    index="Trade Date",
    columns="F_label",
    values="Settle",
    aggfunc="first",  # should never actually collide; 'first' just guards against dupes
)

# reorder columns F1, F2, F3... numerically instead of alphabetically (F10 before F2, etc.)
term_structure = term_structure[
    sorted(term_structure.columns, key=lambda x: int(x[1:]))
]

term_structure = term_structure.sort_index()

# Sanity check: with full monthly coverage, adjacent contracts should expire
# roughly 1 month apart. A larger gap means a monthly contract is missing.
expiry_by_rank = long_df.pivot_table(index="Trade Date", columns="F_label", values="expiry", aggfunc="first")
if "F1" in expiry_by_rank.columns and "F2" in expiry_by_rank.columns:
    gap_days = (expiry_by_rank["F2"] - expiry_by_rank["F1"]).dt.days
    bad_gaps = gap_days[(gap_days > 45) | (gap_days < 20)]
    assert bad_gaps.empty, (
        f"F2-F1 expiry gap outside ~1-month range on {len(bad_gaps)} dates "
        f"(e.g. {bad_gaps.index[0].date()}: {bad_gaps.iloc[0]} days) — "
        f"a monthly contract is likely missing from data/."
    )

print(term_structure.head(10))
print(term_structure.tail(10))

term_structure.to_csv("term_structure.csv")
print("Saved to term_structure.csv")