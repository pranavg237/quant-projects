import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
TICKER = "SPY"
START = "2018-01-01"
END = "2024-01-01"
SHORT_WINDOW = 50
LONG_WINDOW = 200

# --- Pull data ---
df = yf.download(TICKER, start=START, end=END, auto_adjust=True)
df = df[["Close"]].copy()

# --- Compute moving averages ---
df["MA_short"] = df["Close"].rolling(SHORT_WINDOW).mean()
df["MA_long"]  = df["Close"].rolling(LONG_WINDOW).mean()

# --- Generate signals ---
# 1 = hold long, 0 = out of market
df["signal"] = 0
df.loc[df["MA_short"] > df["MA_long"], "signal"] = 1

# Signal changes only on crossover days
df["position"] = df["signal"].diff()

# --- Calculate returns ---
df["market_return"]   = df["Close"].pct_change()
df["strategy_return"] = df["market_return"] * df["signal"].shift(1)  # shift: no lookahead

df["market_cumulative"]   = (1 + df["market_return"]).cumprod()
df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()

# --- Print summary ---
total_market   = df["market_cumulative"].iloc[-1] - 1
total_strategy = df["strategy_cumulative"].iloc[-1] - 1
print(f"Buy & Hold Return:  {total_market:.2%}")
print(f"Strategy Return:    {total_strategy:.2%}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df["Close"], label="Price", alpha=0.7)
ax1.plot(df["MA_short"], label=f"MA {SHORT_WINDOW}", alpha=0.8)
ax1.plot(df["MA_long"],  label=f"MA {LONG_WINDOW}",  alpha=0.8)

# Mark buy/sell signals
buys  = df[df["position"] ==  1]
sells = df[df["position"] == -1]
ax1.scatter(buys.index,  buys["Close"],  marker="^", color="green", zorder=5, label="Buy")
ax1.scatter(sells.index, sells["Close"], marker="v", color="red",   zorder=5, label="Sell")
ax1.set_title(f"{TICKER} MA Crossover ({SHORT_WINDOW}/{LONG_WINDOW})")
ax1.legend()

ax2.plot(df["market_cumulative"],   label="Buy & Hold")
ax2.plot(df["strategy_cumulative"], label="MA Strategy")
ax2.set_title("Cumulative Returns")
ax2.legend()

plt.tight_layout()
plt.savefig("backtest.png", dpi=150)
plt.show()