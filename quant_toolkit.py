#!/usr/bin/env python3
"""quant_toolkit: one program combining price generation, backtests,
parameter screening and scenario simulation.  Educational only -- not
investment advice, and synthetic data says nothing about real markets.

Install:  pip install numpy pandas

Commands
  generate   create a price series (gbm | ou | bridge) and save to CSV
  sample     draw samples from a probability distribution
  backtest   run one strategy (ma | meanrev | pairs) on CSV or synthetic data
  screen     tune MA windows on a training split, test on holdout, bootstrap
             a future-price distribution, print a rule-based label
  pipeline   do everything: get prices -> all strategies -> screen -> report

Examples
  python3 quant_toolkit.py pipeline --model gbm --points 5000
  python3 quant_toolkit.py pipeline --csv prices.csv --column Close
  python3 quant_toolkit.py pipeline --csv pair.csv --column A --column-b B
  python3 quant_toolkit.py backtest --strategy meanrev --model ou
  python3 quant_toolkit.py generate --model gbm --points 100000 --out p.csv
  python3 quant_toolkit.py sample --distribution student-t --df 5
"""
import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Price generators (from stochastic_processes.py, mean_reversion.py, pairs)
# --------------------------------------------------------------------------


def simulate_gbm(start, mu, sigma, duration, points, seed):
    rng = np.random.default_rng(seed)
    dt = duration / (points - 1)
    inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=points - 1)
    return start * np.exp(np.r_[0.0, np.cumsum(inc)])


def simulate_ou(start, mean, speed, sigma, duration, points, seed):
    rng = np.random.default_rng(seed)
    dt = duration / (points - 1)
    shocks = rng.normal(size=points - 1)
    x = np.empty(points)
    x[0], target = np.log(start), np.log(mean)
    for i in range(1, points):
        x[i] = x[i - 1] + speed * (target - x[i - 1]) * dt + sigma * np.sqrt(dt) * shocks[i - 1]
    return np.exp(x)


def make_bridge(start, end, n, seed):
    """Noisy path pinned to exact start/end prices."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    noise = rng.normal(0, 0.01, n).cumsum()
    log_p = np.log(start) + t * (np.log(end) - np.log(start)) + noise - t * noise[-1]
    p = np.exp(log_p)
    p[0], p[-1] = start, end
    return p


def make_pair(n, sa, ea, sb, eb, seed):
    """Two prices whose log spread mean-reverts."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    a_noise = rng.normal(0, 0.008, n).cumsum()
    log_a = np.log(sa) + t * (np.log(ea) - np.log(sa)) + a_noise - t * a_noise[-1]
    spread = np.zeros(n)
    innov = rng.normal(0, 0.004, n)
    for i in range(1, n):
        spread[i] = 0.98 * spread[i - 1] + innov[i]
    spread -= t * spread[-1]
    base = np.log(sb / sa) + t * (np.log(eb / ea) - np.log(sb / sa))
    log_b = log_a + base + spread
    log_b[0], log_b[-1] = np.log(sb), np.log(eb)
    return np.exp(log_a), np.exp(log_b)


def synthetic_prices(a) -> np.ndarray:
    n = a.points
    if n < 250:
        sys.exit("error: need at least 250 points")
    duration = n / a.periods_per_year  # keeps dt = one period
    if a.model == "gbm":
        return simulate_gbm(a.start, a.mu, a.vol, duration, n, a.seed)
    if a.model == "ou":
        return simulate_ou(a.start, a.mean, a.speed, a.vol, duration, n, a.seed)
    return make_bridge(a.start, a.end, n, a.seed)


def load_column(path, col):
    raw = pd.read_csv(path)
    if col not in raw.columns:
        sys.exit(f"error: column {col!r} not found; available: {list(raw.columns)}")
    return pd.to_numeric(raw[col], errors="coerce")


def get_prices(a) -> np.ndarray:
    if a.csv:
        s = load_column(a.csv, a.column).dropna()
        p = s.to_numpy(float)
    else:
        p = synthetic_prices(a)
    if len(p) < 250 or not np.isfinite(p).all() or (p <= 0).any():
        sys.exit("error: need at least 250 finite, positive prices")
    return p


# --------------------------------------------------------------------------
# Shared backtest engine + metrics
# --------------------------------------------------------------------------


def apply_positions(asset_ret, signal, cost_bps):
    """signal[t] is decided with data up to t's close and held during t+1.
    Returns net per-period returns and the position actually held."""
    held = np.r_[0.0, signal[:-1]]
    turnover = np.abs(np.diff(held, prepend=0.0))
    net = held * asset_ret - turnover * cost_bps / 1e4
    return net, held, turnover


def simple_returns(prices):
    return np.r_[0.0, prices[1:] / prices[:-1] - 1.0]


@dataclass
class Result:
    name: str
    returns: np.ndarray
    turnover: np.ndarray
    ppy: int

    @property
    def equity(self):
        return np.cumprod(1 + self.returns)

    def summary(self) -> dict:
        eq = self.equity
        sd = self.returns.std(ddof=1)
        return {
            "strategy": self.name,
            "total_return": eq[-1] - 1,
            "max_drawdown": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
            "sharpe": float(np.sqrt(self.ppy) * self.returns.mean() / sd) if sd > 0 else 0.0,
            "position_changes": int((self.turnover > 0).sum()),
        }


def sharpe(r, ppy):
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(np.sqrt(ppy) * r.mean() / sd) if sd > 0 else 0.0


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def ma_signal(prices, fast, slow):
    s = pd.Series(prices)
    f, sl = s.rolling(fast).mean(), s.rolling(slow).mean()
    return ((f > sl) & sl.notna()).astype(float).to_numpy()


def run_ma(prices, fast, slow, fee_bps, ppy):
    if not 1 <= fast < slow:
        sys.exit("error: need 1 <= fast < slow")
    r, _, t = apply_positions(simple_returns(prices), ma_signal(prices, fast, slow), fee_bps)
    return Result(f"ma_crossover({fast}/{slow})", r, t, ppy)


def zscore_state_machine(z, entry, exit_):
    """Long/short/flat state from a z-score series (signal known at t)."""
    sig = np.zeros(len(z))
    state = 0
    for i, v in enumerate(z):
        if not np.isfinite(v):
            state = 0
        elif state == 0:
            state = 1 if v <= -entry else (-1 if v >= entry else 0)
        elif abs(v) <= exit_:
            state = 0
        sig[i] = state
    return sig


def check_z(entry, exit_):
    if entry <= exit_ or exit_ < 0:
        sys.exit("error: require entry-z > exit-z >= 0")


def run_meanrev(prices, window, entry, exit_, fee_bps, ppy):
    check_z(entry, exit_)
    s = pd.Series(prices)
    z = ((s - s.rolling(window).mean()) / s.rolling(window).std(ddof=0).replace(0, np.nan)).to_numpy()
    sig = zscore_state_machine(z, entry, exit_)
    r, _, t = apply_positions(simple_returns(prices), sig, fee_bps)
    return Result(f"mean_reversion(w={window})", r, t, ppy)


def run_pairs(pa, pb, lookback, entry, exit_, fee_bps, ppy):
    check_z(entry, exit_)
    la, lb = pd.Series(np.log(pa)), pd.Series(np.log(pb))
    beta = (la.rolling(lookback).cov(lb) / lb.rolling(lookback).var(ddof=0)).replace([np.inf, -np.inf], np.nan)
    spread = la - beta * lb
    z = ((spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std(ddof=0).replace(0, np.nan)).to_numpy()
    sig = zscore_state_machine(z, entry, exit_)
    # previous bar's hedge ratio defines the pair return (no look-ahead)
    hb = beta.shift(1).fillna(1.0).to_numpy()
    pair_ret = (simple_returns(pa) - hb * simple_returns(pb)) / (1.0 + np.abs(hb))
    r, _, t = apply_positions(pair_ret, sig, 2.0 * fee_bps)  # cost on both legs
    return Result(f"pairs_trading(lb={lookback})", r, t, ppy)


# --------------------------------------------------------------------------
# Screener (from investment_screener.py)
# --------------------------------------------------------------------------


def screen(prices, a):
    n = len(prices)
    split = int(n * 0.7)
    ret = simple_returns(prices)
    best = None
    for fast in (5, 10, 20, 30):
        for slow in (40, 60, 100, 150):
            if fast >= slow or split <= slow + 10:
                continue
            sig = ma_signal(prices[:split], fast, slow)
            r, _, _ = apply_positions(ret[:split], sig, a.fee_bps)
            sc = sharpe(r, a.periods_per_year)
            if best is None or sc > best[2]:
                best = (fast, slow, sc)
    if best is None:
        sys.exit("error: not enough data to optimise MA windows")
    fast, slow, train_sharpe = best

    # signal uses the whole history (so MAs are warm), score only the holdout
    sig = ma_signal(prices, fast, slow)
    r, _, t = apply_positions(ret, sig, a.fee_bps)
    test = Result("holdout", r[split:], t[split:], a.periods_per_year)
    ts = test.summary()

    recent = ret[1:][-min(a.lookback, n - 1):]
    rng = np.random.default_rng(a.seed + 1)
    sampled = rng.choice(recent, size=(a.simulations, a.horizon), replace=True)
    fut = prices[-1] * np.prod(1 + sampled, axis=1)
    q05, q50, q95 = np.quantile(fut, [0.05, 0.5, 0.95])
    med_ret = q50 / prices[-1] - 1

    if med_ret >= a.min_expected_return and ts["total_return"] > 0 and ts["max_drawdown"] >= -a.max_test_drawdown:
        label = "BUY_CANDIDATE"
    elif med_ret < 0 and ts["total_return"] < 0:
        label = "AVOID / REVIEW"
    else:
        label = "HOLD / INSUFFICIENT EDGE"
    return dict(fast=fast, slow=slow, train_sharpe=train_sharpe, holdout=ts,
                q05=q05, q50=q50, q95=q95, median_return=med_ret,
                label=label, future=fut)


def print_screen(s, last, a):
    print(f"MA windows tuned on first 70%: fast={s['fast']} slow={s['slow']} (train Sharpe {s['train_sharpe']:.2f})")
    h = s["holdout"]
    print(f"Holdout return {h['total_return']:.2%}, max drawdown {h['max_drawdown']:.2%}")
    print(f"Bootstrap {a.horizon} periods from {last:.4f}: median {s['q50']:.4f} "
          f"(5%-95%: {s['q05']:.4f} to {s['q95']:.4f})")
    print(f"Rule-based label: {s['label']}   (mechanical filter, NOT advice)")


# --------------------------------------------------------------------------
# Distribution sampler (from probability_distributions.py)
# --------------------------------------------------------------------------


def sample(a):
    rng = np.random.default_rng(a.seed)
    n, d = a.points, a.distribution
    if d == "normal":
        v = rng.normal(a.mean, a.std, n)
    elif d == "lognormal":
        v = rng.lognormal(a.mean, a.std, n)
    elif d == "uniform":
        v = rng.uniform(a.low, a.high, n)
    elif d == "exponential":
        v = rng.exponential(a.scale, n)
    else:
        v = a.mean + a.std * rng.standard_t(a.df, n)
    if a.scale_to_range:
        lo, hi = v.min(), v.max()
        v = np.full(n, (a.min_price + a.max_price) / 2) if hi == lo else \
            a.min_price + (v - lo) * (a.max_price - a.min_price) / (hi - lo)
    return v


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_source(p):
    g = p.add_argument_group("price source")
    g.add_argument("--csv", help="CSV of real prices (otherwise synthetic)")
    g.add_argument("--column", default="Close")
    g.add_argument("--model", choices=["gbm", "ou", "bridge"], default="gbm")
    g.add_argument("--points", type=int, default=5000)
    g.add_argument("--start", type=float, default=100.0)
    g.add_argument("--end", type=float, default=110.0, help="bridge model only")
    g.add_argument("--mu", type=float, default=0.05, help="GBM annual drift")
    g.add_argument("--vol", type=float, default=0.20, help="annual volatility")
    g.add_argument("--mean", type=float, default=100.0, help="OU long-run level")
    g.add_argument("--speed", type=float, default=2.0, help="OU reversion speed")
    g.add_argument("--periods-per-year", type=int, default=252)
    g.add_argument("--seed", type=int, default=7)


def add_strategy_args(p):
    g = p.add_argument_group("strategy")
    g.add_argument("--fast", type=int, default=20)
    g.add_argument("--slow", type=int, default=50)
    g.add_argument("--window", type=int, default=50, help="mean-reversion window")
    g.add_argument("--lookback", type=int, default=100, help="pairs window")
    g.add_argument("--entry-z", type=float, default=2.0)
    g.add_argument("--exit-z", type=float, default=0.5)
    g.add_argument("--fee-bps", type=float, default=5.0)
    g.add_argument("--column-b", help="second price column in --csv (enables pairs)")


def add_screen_args(p):
    g = p.add_argument_group("screener")
    g.add_argument("--horizon", type=int, default=252)
    g.add_argument("--simulations", type=int, default=10_000)
    g.add_argument("--boot-lookback", dest="lookback_boot", type=int, default=252)
    g.add_argument("--min-expected-return", type=float, default=0.05)
    g.add_argument("--max-test-drawdown", type=float, default=0.30)


def get_pair(a, prices):
    if a.csv and a.column_b:
        b = load_column(a.csv, a.column_b)
        df = pd.DataFrame({"a": load_column(a.csv, a.column), "b": b}).dropna()
        return df.a.to_numpy(float), df.b.to_numpy(float)
    if a.csv:
        return None
    return make_pair(a.points, a.start, a.end, 0.8 * a.start, 0.8 * a.end, a.seed)


def fmt(rows):
    df = pd.DataFrame(rows)
    for c in ("total_return", "max_drawdown"):
        df[c] = df[c].map("{:.2%}".format)
    df["sharpe"] = df["sharpe"].map("{:.2f}".format)
    return df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate"); add_source(g); g.add_argument("--out", default="prices.csv")

    s = sub.add_parser("sample")
    s.add_argument("--distribution", choices=["normal", "lognormal", "uniform", "exponential", "student-t"], default="normal")
    s.add_argument("--points", type=int, default=100_000)
    s.add_argument("--mean", type=float, default=100.0)
    s.add_argument("--std", type=float, default=10.0)
    s.add_argument("--low", type=float, default=50.0)
    s.add_argument("--high", type=float, default=150.0)
    s.add_argument("--scale", type=float, default=20.0)
    s.add_argument("--df", type=float, default=5.0)
    s.add_argument("--scale-to-range", action="store_true")
    s.add_argument("--min-price", type=float, default=50.0)
    s.add_argument("--max-price", type=float, default=150.0)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="distribution_samples.csv")

    b = sub.add_parser("backtest"); add_source(b); add_strategy_args(b)
    b.add_argument("--strategy", choices=["ma", "meanrev", "pairs"], required=True)
    b.add_argument("--out", default="backtest_results.csv")

    sc = sub.add_parser("screen"); add_source(sc); add_strategy_args(sc); add_screen_args(sc)
    sc.add_argument("--out", default="screen_simulations.csv")

    pl = sub.add_parser("pipeline"); add_source(pl); add_strategy_args(pl); add_screen_args(pl)
    pl.add_argument("--outdir", default=".")

    a = ap.parse_args()
    if hasattr(a, "lookback_boot"):
        a.boot = a.lookback_boot
    ppy = getattr(a, "periods_per_year", 252)

    if a.cmd == "generate":
        p = get_prices(a)
        pd.DataFrame({"Close": p}).to_csv(a.out, index=False)
        print(f"{len(p):,} prices {p[0]:.4f} -> {p[-1]:.4f} saved to {a.out}")

    elif a.cmd == "sample":
        v = sample(a)
        pd.DataFrame({"observation": np.arange(1, len(v) + 1), "value": v}).to_csv(a.out, index=False)
        print(f"{a.distribution}: mean {v.mean():.4f}, sd {v.std(ddof=1):.4f}, "
              f"min/median/max {v.min():.4f}/{np.median(v):.4f}/{v.max():.4f}; saved to {a.out}")

    elif a.cmd == "backtest":
        if a.strategy == "pairs":
            pair = get_pair(a, None)
            if pair is None:
                sys.exit("error: pairs needs --column-b with --csv")
            res = run_pairs(*pair, a.lookback, a.entry_z, a.exit_z, a.fee_bps, ppy)
        else:
            p = get_prices(a)
            res = run_ma(p, a.fast, a.slow, a.fee_bps, ppy) if a.strategy == "ma" else \
                run_meanrev(p, a.window, a.entry_z, a.exit_z, a.fee_bps, ppy)
        pd.DataFrame({"strategy_return": res.returns, "turnover": res.turnover, "equity": res.equity}).to_csv(a.out, index=False)
        print(fmt([res.summary()])); print(f"saved to {a.out}")

    elif a.cmd == "screen":
        p = get_prices(a)
        a.lookback = a.lookback_boot
        out = screen(p, a)
        print_screen(out, p[-1], a)
        pd.DataFrame({"simulation": np.arange(1, len(out["future"]) + 1), "future_value": out["future"]}).to_csv(a.out, index=False)

    elif a.cmd == "pipeline":
        import os
        os.makedirs(a.outdir, exist_ok=True)
        p = get_prices(a)
        print(f"Data: {len(p):,} prices, {p[0]:.4f} -> {p[-1]:.4f}"
              f" ({'CSV' if a.csv else 'synthetic ' + a.model})\n")
        results = [run_ma(p, a.fast, a.slow, a.fee_bps, ppy),
                   run_meanrev(p, a.window, a.entry_z, a.exit_z, a.fee_bps, ppy)]
        pair = get_pair(a, p)
        if pair is not None:
            results.append(run_pairs(*pair, a.lookback, a.entry_z, a.exit_z, a.fee_bps, ppy))
        else:
            print("(pairs skipped: pass --column-b with --csv to enable)\n")
        bh = float(p[-1] / p[0] - 1)
        print("Full-sample backtests (net of costs):")
        print(fmt([r.summary() for r in results]))
        print(f"Buy & hold over same period: {bh:.2%}\n")
        eq = pd.DataFrame({r.name: r.equity for r in results})
        eq.to_csv(f"{a.outdir}/pipeline_equity.csv", index=False)

        a.lookback = a.lookback_boot
        out = screen(p, a)
        print_screen(out, p[-1], a)
        pd.DataFrame({"simulation": np.arange(1, len(out["future"]) + 1), "future_value": out["future"]}).to_csv(
            f"{a.outdir}/pipeline_simulations.csv", index=False)
        print(f"\nSaved equity curves and simulations to {a.outdir}/")
        print("Educational output only -- not investment advice or a prediction.")


if __name__ == "__main__":
    main()
