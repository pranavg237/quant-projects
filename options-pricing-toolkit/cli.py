"""
Command-line pricer / Greeks calculator.

Examples:
  # price + Greeks for a call
  python3 cli.py price --S 230 --K 235 --dte 2 --r 0.045 --sigma 0.55 --type call

  # solve implied vol from an observed market price
  python3 cli.py iv --S 230 --K 235 --dte 2 --r 0.045 --type call --price 3.10
"""
import argparse

from black_scholes import price, greeks, implied_vol


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Black-Scholes options calculator")
    sub = p.add_subparsers(dest="cmd", required=True)

    price_p = sub.add_parser("price", help="Price an option and show its Greeks")
    price_p.add_argument("--S", type=float, required=True, help="Underlying price")
    price_p.add_argument("--K", type=float, required=True, help="Strike price")
    price_p.add_argument("--dte", type=float, required=True, help="Days to expiry")
    price_p.add_argument("--r", type=float, default=0.045, help="Risk-free rate (default 0.045)")
    price_p.add_argument("--q", type=float, default=0.0, help="Dividend yield (default 0)")
    price_p.add_argument("--sigma", type=float, required=True, help="Annualized IV, e.g. 0.55")
    price_p.add_argument("--type", dest="option_type", choices=["call", "put"], required=True)

    iv_p = sub.add_parser("iv", help="Solve implied vol from an observed market price")
    iv_p.add_argument("--S", type=float, required=True)
    iv_p.add_argument("--K", type=float, required=True)
    iv_p.add_argument("--dte", type=float, required=True)
    iv_p.add_argument("--r", type=float, default=0.045)
    iv_p.add_argument("--q", type=float, default=0.0)
    iv_p.add_argument("--price", type=float, required=True, dest="market_price")
    iv_p.add_argument("--type", dest="option_type", choices=["call", "put"], required=True)

    return p


def main():
    args = build_parser().parse_args()
    T = args.dte / 365.0

    if args.cmd == "price":
        px = price(args.S, args.K, T, args.r, args.sigma, args.option_type, args.q)
        g = greeks(args.S, args.K, T, args.r, args.sigma, args.option_type, args.q)
        print(f"price = {px:.4f}")
        print(g)

    elif args.cmd == "iv":
        iv = implied_vol(args.market_price, args.S, args.K, T, args.r, args.option_type, args.q)
        g = greeks(args.S, args.K, T, args.r, iv, args.option_type, args.q)
        print(f"implied_vol = {iv:.4f} ({iv*100:.2f}%)")
        print(g)


if __name__ == "__main__":
    main()
