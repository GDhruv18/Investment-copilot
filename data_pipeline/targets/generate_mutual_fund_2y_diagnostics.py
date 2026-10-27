from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("data/clean/mutual_fund_nav_clean.parquet")
OUTPUT_FILE = Path("data/targets/mutual_fund_2y_diagnostics.parquet")
EXPECTED_SCHEMES = 100
EXPECTED_START = pd.Timestamp("2013-01-01")
EXPECTED_END = pd.Timestamp("2026-10-22")


class RangeDrawdownTree:
    def __init__(self, values):
        n = len(values)
        size = 1
        while size < n:
            size *= 2
        self.size = size
        self.mx = np.full(2 * size, -np.inf)
        self.mn = np.full(2 * size, np.inf)
        self.dd = np.zeros(2 * size)
        self.mx[size:size+n] = values
        self.mn[size:size+n] = values
        for p in range(size - 1, 0, -1):
            self._pull(p)

    def _pull(self, p):
        l, r = 2*p, 2*p+1
        self.mx[p] = max(self.mx[l], self.mx[r])
        self.mn[p] = min(self.mn[l], self.mn[r])
        cross = self.mn[r] / self.mx[l] - 1 if np.isfinite(self.mx[l]) and np.isfinite(self.mn[r]) else 0
        self.dd[p] = min(self.dd[l], self.dd[r], cross)

    @staticmethod
    def merge(a, b):
        if a is None: return b
        if b is None: return a
        amax, amin, add = a
        bmax, bmin, bdd = b
        return max(amax, bmax), min(amin, bmin), min(add, bdd, bmin/amax - 1)

    def query(self, left, right):
        left += self.size
        right += self.size
        a = b = None
        while left <= right:
            if left & 1:
                a = self.merge(a, (self.mx[left], self.mn[left], self.dd[left]))
                left += 1
            if not (right & 1):
                b = self.merge((self.mx[right], self.mn[right], self.dd[right]), b)
                right -= 1
            left //= 2
            right //= 2
        return self.merge(a, b)


def validate_input(df):
    required = {"scheme_code", "nav"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # The clean mutual-fund NAV file uses `date` as its date column.
    # Keep `nav_date` as the internal name so the rest of this script
    # remains consistent with the PostgreSQL source schema.
    df = df.copy()
    if "nav_date" not in df.columns:
        if "date" in df.columns:
            df["nav_date"] = df["date"]
        else:
            raise ValueError(
                "Could not find the NAV date column. Expected either "
                "`nav_date` or `date`."
            )

    df["scheme_code"] = df["scheme_code"].astype(str)
    df["nav_date"] = pd.to_datetime(df["nav_date"], errors="coerce").dt.normalize()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    if df["nav_date"].isna().any() or df["nav"].isna().any() or (df["nav"] <= 0).any():
        raise ValueError("Invalid dates or non-positive/missing NAV values.")
    dup = df.duplicated(["scheme_code", "nav_date"]).sum()
    if dup:
        raise ValueError(f"Duplicate scheme/date rows: {dup}")
    return df.sort_values(["scheme_code", "nav_date"]).reset_index(drop=True)


def calculate_scheme(group):
    group = group.sort_values("nav_date").reset_index(drop=True)
    dates = group["nav_date"].to_numpy(dtype="datetime64[ns]")
    nav = group["nav"].to_numpy(float)
    n = len(group)

    target = (pd.Series(pd.to_datetime(dates)) + pd.DateOffset(years=2)).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(dates, target, side="left")
    valid = pos < n

    future_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    future_nav = np.full(n, np.nan)
    years = np.full(n, np.nan)
    idx = np.flatnonzero(valid)
    if len(idx):
        j = pos[idx]
        future_date[idx] = dates[j]
        future_nav[idx] = nav[j]
        days = (future_date[idx] - dates[idx]).astype("timedelta64[D]").astype(float)
        years[idx] = days / 365.25

    ann_ret = np.full(n, np.nan)
    vol = np.full(n, np.nan)
    dd = np.full(n, np.nan)
    ra = np.full(n, np.nan)

    daily = np.full(n, np.nan)
    if n > 1:
        daily[1:] = nav[1:] / nav[:-1] - 1

    ok = np.isfinite(daily)
    ps = np.zeros(n+1)
    ps2 = np.zeros(n+1)
    pc = np.zeros(n+1, dtype=np.int64)
    for k in range(n):
        ps[k+1], ps2[k+1], pc[k+1] = ps[k], ps2[k], pc[k]
        if ok[k]:
            ps[k+1] += daily[k]
            ps2[k+1] += daily[k]**2
            pc[k+1] += 1

    tree = RangeDrawdownTree(nav)

    for i in idx:
        j = pos[i]
        if years[i] > 0:
            ann_ret[i] = (future_nav[i] / nav[i]) ** (1 / years[i]) - 1

        a, b = i + 1, j + 1
        count = pc[b] - pc[a]
        if count >= 2:
            total = ps[b] - ps[a]
            total2 = ps2[b] - ps2[a]
            mean = total / count
            variance = max((total2 - count * mean * mean) / (count - 1), 0)
            vol[i] = np.sqrt(variance) * np.sqrt(252)

        dd[i] = tree.query(i, j)[2]
        if np.isfinite(ann_ret[i]) and np.isfinite(vol[i]) and vol[i] > 0:
            ra[i] = ann_ret[i] / vol[i]

    out = group.copy()
    out["target_date_2y"] = pd.to_datetime(target)
    out["future_date_2y"] = pd.to_datetime(future_date)
    out["future_nav_2y"] = future_nav
    out["actual_years_2y"] = years
    out["annualized_return_2y"] = ann_ret
    out["annualized_volatility_2y"] = vol
    out["max_drawdown_2y"] = dd
    out["risk_adjusted_return_2y"] = ra
    return out


def distribution(s, name):
    s = s.dropna()
    if s.empty:
        print(f"\n{name}: no valid observations")
        return
    q = s.quantile([.01,.05,.10,.25,.50,.75,.90,.95,.99])
    print(f"\n{name}: count={len(s):,}, mean={s.mean():.6f}, std={s.std():.6f}, min={s.min():.6f}")
    for p, v in q.items():
        print(f"  {p:.0%}: {v:.6f}")
    print(f"  max: {s.max():.6f}")


def main():
    print("="*70)
    print("MUTUAL FUND 2Y DIAGNOSTICS")
    print("="*70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = validate_input(pd.read_parquet(INPUT_FILE))
    print(f"Rows: {len(df):,}")
    print(f"Schemes: {df['scheme_code'].nunique():,}")
    print(f"Date range: {df.nav_date.min().date()} -> {df.nav_date.max().date()}")

    if df["scheme_code"].nunique() != EXPECTED_SCHEMES:
        raise ValueError("Expected exactly 100 selected schemes.")
    if df.nav_date.min() != EXPECTED_START or df.nav_date.max() != EXPECTED_END:
        raise ValueError("Unexpected clean NAV date range.")

    parts = []
    schemes = df["scheme_code"].nunique()
    for k, (scheme, g) in enumerate(df.groupby("scheme_code", sort=True), 1):
        print(f"[{k:03d}/{schemes:03d}] {scheme} ({len(g):,} rows)")
        parts.append(calculate_scheme(g))

    out = pd.concat(parts, ignore_index=True).sort_values(["scheme_code","nav_date"]).reset_index(drop=True)
    valid = out.future_date_2y.notna()

    print("\n"+"="*70)
    print("2Y OUTCOME SUMMARY")
    print("="*70)
    print(f"Rows: {len(out):,}")
    print(f"Valid 2Y outcomes: {valid.sum():,}")
    print(f"Missing 2Y outcomes: {(~valid).sum():,}")
    print(f"Coverage: {valid.mean():.2%}")

    cohort = out.loc[valid].groupby("nav_date").scheme_code.nunique()
    print(f"Cohort dates: {len(cohort):,}; min={cohort.min() if len(cohort) else 0}, median={cohort.median() if len(cohort) else 0:.0f}, max={cohort.max() if len(cohort) else 0}")

    distribution(out.annualized_return_2y, "ANNUALIZED 2Y RETURN")
    distribution(out.annualized_volatility_2y, "ANNUALIZED 2Y VOLATILITY")
    distribution(out.max_drawdown_2y, "2Y MAX DRAWDOWN")
    distribution(out.risk_adjusted_return_2y, "RAW RISK-ADJUSTED RETURN")

    print("\nEXTREME COUNTS")
    r, v, d, ra = out.annualized_return_2y, out.annualized_volatility_2y, out.max_drawdown_2y, out.risk_adjusted_return_2y
    for x in [.5,1,2,5]: print(f"Return > {x:.0%}: {(r>x).sum():,}")
    for x in [-.25,-.5,-.75]: print(f"Return < {x:.0%}: {(r<x).sum():,}")
    for x in [.01,.02,.05,.10]: print(f"Volatility < {x:.0%}: {(v<x).sum():,}")
    for x in [.5,.75,1]: print(f"Volatility > {x:.0%}: {(v>x).sum():,}")
    for x in [-.01,-.05,-.10]: print(f"Drawdown > {x:.0%}: {(d>x).sum():,}")
    for x in [-.25,-.5,-.75]: print(f"Drawdown < {x:.0%}: {(d<x).sum():,}")
    for x in [5,10,20]: print(f"Risk-adjusted > {x}: {(ra>x).sum():,}")
    print(f"Risk-adjusted < 0: {(ra<0).sum():,}")

    all_components = out[["annualized_return_2y","annualized_volatility_2y","max_drawdown_2y","risk_adjusted_return_2y"]].dropna()
    print("\nCOMPONENT CORRELATIONS — PEARSON")
    print(all_components.corr().round(4).to_string())
    print("\nCOMPONENT CORRELATIONS — SPEARMAN")
    print(all_components.corr(method="spearman").round(4).to_string())

    print("\nTop annualized-return observations:")
    print(out.nlargest(10, "annualized_return_2y")[["scheme_code","nav_date","future_date_2y","annualized_return_2y","annualized_volatility_2y","max_drawdown_2y"]].to_string(index=False))
    print("\nWorst annualized-return observations:")
    print(out.nsmallest(10, "annualized_return_2y")[["scheme_code","nav_date","future_date_2y","annualized_return_2y","annualized_volatility_2y","max_drawdown_2y"]].to_string(index=False))
    print("\nHighest-volatility observations:")
    print(out.nlargest(10, "annualized_volatility_2y")[["scheme_code","nav_date","annualized_return_2y","annualized_volatility_2y","max_drawdown_2y"]].to_string(index=False))
    print("\nWorst-drawdown observations:")
    print(out.nsmallest(10, "max_drawdown_2y")[["scheme_code","nav_date","annualized_return_2y","annualized_volatility_2y","max_drawdown_2y"]].to_string(index=False))

    # Final validation.
    checks = {
        "row count preserved": len(out) == len(df),
        "scheme count preserved": out.scheme_code.nunique() == 100,
        "duplicate scheme/date": out.duplicated(["scheme_code","nav_date"]).sum() == 0,
        "target-date violations": not ((out.future_date_2y.notna()) & (out.future_date_2y < out.target_date_2y)).any(),
        "future before current": not ((out.future_date_2y.notna()) & (out.future_date_2y < out.nav_date)).any(),
        "invalid elapsed years": not ((out.actual_years_2y.notna()) & (out.actual_years_2y <= 0)).any(),
        "invalid annualized return": not (out.annualized_return_2y.dropna().map(np.isfinite) == False).any(),
        "invalid volatility": not (out.annualized_volatility_2y.dropna().map(np.isfinite) == False).any(),
        "invalid drawdown": not (out.max_drawdown_2y.dropna().map(np.isfinite) == False).any(),
        "drawdown outside [-1,0]": not ((out.max_drawdown_2y.dropna() > 1e-12) | (out.max_drawdown_2y.dropna() < -1-1e-12)).any(),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_FILE, index=False)
    reload = pd.read_parquet(OUTPUT_FILE)

    print("\nFINAL VALIDATION")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} - {name}")
    print(f"Reload rows: {len(reload):,}")
    print(f"Reload schemes: {reload.scheme_code.nunique():,}")

    if not all(checks.values()) or len(reload) != len(out) or reload.scheme_code.nunique() != 100:
        raise RuntimeError("Final validation failed.")

    print("\nFINAL STATUS: PASS")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
