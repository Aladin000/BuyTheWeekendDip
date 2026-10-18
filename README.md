# Buy the Weekend Dip?

Does Bitcoin recover after a sharp hourly drop? And does it matter whether that drop happens at the weekend or after a quiet 24 hours of trading?

This project explores those questions using hourly BTC/USDT spot data from Binance. It compares four simple trading rules, with fees and slippage included.

The [report](report.pdf) explains the research and findings. The [notebook](notebook.ipynb) walks through the data, analysis and backtest, with the results already displayed.

## What I found

The weekend rule gained 1.88% during development but lost 11.74% in the later test period. All four rules lost money in that test period after costs. The statistical tests also found no reliable support for the four main predictions about rebounds, weekend timing and earlier trading activity.

| Trading rule | 2020–2023 net return | Jan 2024–Jul 2026 net return |
| :--- | ---: | ---: |
| Every qualifying drop | -57.47% | -52.57% |
| Weekend drops | +1.88% | -11.74% |
| Drops after low trading activity | -6.37% | -11.91% |
| Weekend drops after low activity | -10.10% | -8.02% |

These are cumulative returns for each period, including a 0.10% fee and 0.02% slippage on each buy and sell. The combined rule made only 15 trades in the test period, so its results need particular care. The report includes trade counts, Sharpe ratios, drawdowns and checks using different costs and settings.

## How it works

A drop qualifies when its hourly return is more than 2.5 standard deviations below its average over the previous 30 days. A 24-hour cooldown limits repeated signals from the same sell-off. Trading activity is measured using the USDT value traded in the preceding 24 hours, excluding the drop itself.

Each rule buys at the next hourly open and sells after 12 hours. It uses the full account balance, holds one position at a time and keeps USDT between trades. Weekend means Saturday or Sunday in UTC.

The rules were fixed before outcomes were inspected. Development covers 2020–2023; January 2024 to July 2026 is the separate historical test period. The full definitions are in the [research specification](docs/PROJECT_SPEC.md) and [configuration](config/project.yaml).

## Data

The repository includes 82 monthly archives from [Binance Public Data](https://github.com/binance/binance-public-data), covering October 2019 to July 2026. The final three months of 2019 provide the history needed for the first calculations.

The code verifies the original files against Binance's checksums and handles the timestamp format change in 2025. Eight rows with irregular closing times are excluded and recorded. Missing hours are kept as gaps; prices are never filled in. The original downloads are preserved in `data/raw/`.

## Run the notebook

Tested on macOS with **Python 3.12.13**. Use Python 3.12 and install the supplied dependencies:

```bash
git clone https://github.com/Aladin000/BuyTheWeekendDip.git
cd BuyTheWeekendDip
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
```

Open `notebook.ipynb` in your notebook editor, select `.venv/bin/python` as the kernel, then restart it and run all cells. Use the project folder as the working directory and keep its folder structure intact.

A full run takes about two minutes on the tested machine. It uses the included data, needs no API key or further downloads, and checks that all 56 rebuilt outputs match the saved results exactly. Other platforms have not been tested.

## Project files

| Path | Contents |
| :--- | :--- |
| `report.pdf` | Research report |
| `notebook.ipynb` | Executed analysis and backtest |
| `data/raw/` | Original data archives and checksums |
| `src/` | Data processing, statistical analysis and backtest code |
| `config/` | Research settings and data-quality review decisions |
| `outputs/runs/` | Saved tables, charts and run records |
| `docs/PROJECT_SPEC.md` | Detailed research design |
| `tests/` | Automated checks |
| `report/` | LaTeX source and generated report tables |

## Check the project

With the environment activated, run:

```bash
python -m pytest
python -m src.pipeline config-check
```

The tests cover data integrity, time alignment, event selection, statistical calculations, trading costs and portfolio accounting.

To regenerate the report's numerical LaTeX tables from the verified saved results:

```bash
python -m src.pipeline report
```

## Limits

This study covers one asset, exchange and stablecoin pair. Trading volume measures completed activity, not the orders available to buy or sell. The observed relationships do not establish cause and effect.

Fees and slippage are assumptions, and market impact is not modelled. Some groups contain few events. This is a historical test, and the findings do not show a reliable trading edge after the assumed costs.
