import requests, os
import pandas as pd

os.makedirs("data", exist_ok=True)
base = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/"

dates = (pd.date_range("2013-01-01", "2026-12-31", freq="W-TUE")
         .union(pd.date_range("2013-01-01", "2026-12-31", freq="W-WED"))
         .union(pd.date_range("2013-01-01", "2026-12-31", freq="W-THU")))

ok = 0
for d in dates:
    name = f"VX_{d.date()}.csv"
    if os.path.exists(f"data/{name}"):
        continue  # already downloaded
    try:
        r = requests.get(base + name, timeout=10)
        if r.status_code == 200 and len(r.content) > 500:
            open(f"data/{name}", "wb").write(r.content)
            ok += 1
            print("ok", name)
    except requests.RequestException:
        pass
print(f"new files: {ok}")