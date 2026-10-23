# Project Specification

**Project:** Buy the Weekend Dip?  
**Subtitle:** Bitcoin Reversals After Extreme Sell-Offs: Weekend Timing or Market Activity?  
**Specification version:** 1.5

This document describes the project's research design and computational definitions. The methodological choices were fixed before outcome inspection; the text presents those choices for the completed project.

## 1. Purpose

This project investigates the short-term behaviour of Bitcoin after unusually large negative hourly returns. It asks whether:

1. prices reverse after extreme hourly sell-offs;
2. reversal is stronger for weekend shocks;
3. reversal is associated with low pre-shock trading activity;
4. weekend timing contains information beyond observable activity and other pre-shock conditions;
5. negative and positive shocks behave asymmetrically;
6. any identified relationship has held-out economic value after costs.

The project does not assume that weekend liquidity is thin, that investors overreact, or that a profitable strategy exists.

Technical terms used below:

- **event:** an accepted extreme-price-move observation;
- **activity proxy:** a measurable stand-in for market activity, not direct liquidity;
- **held out:** historical data kept unseen while choices are made;
- **HC3:** standard errors designed to be safer when residual variance is uneven;
- **Holm correction:** a procedure that controls false positives across H1–H4;
- **month-block bootstrap:** resampling whole calendar months to preserve within-month dependence.

## 2. Research design

The study combines an event study with a trading backtest on one market. The event study tests whether post-shock returns are associated with weekend timing and preceding market activity. The backtest evaluates whether the same event definitions support an economic trading rule after explicit costs.

Primary parameters and secondary sensitivity checks were specified before outcome inspection. Development and historical held-out samples are analysed separately. Statistical evidence and net trading performance answer distinct questions; a positive association alone does not establish a profitable strategy.

## 3. Central argument

Previous research separately documents short-horizon Bitcoin reversal, time-varying cryptocurrency trading conditions, and activity-dependent predictability. These findings do not establish a weekend-overreaction effect.

The project therefore tests whether extreme Bitcoin sell-offs are followed by short-term reversal and whether an apparent weekend difference survives after pre-shock market activity is considered.

The intended distinction is:

\[
\text{calendar label: weekend}
\quad\text{versus}\quad
\text{observable state: pre-shock market activity}
\]

If the common-sample weekend coefficient becomes smaller after activity is included, that descriptive change is consistent with activity explaining part of the raw weekend association. Its magnitude and uncertainty must be shown; no arbitrary “material” cutoff is used. This is not causal proof. If the weekend coefficient remains positive and H4 is supported under the locked rule, weekend timing contains measured incremental association not captured by the included activity variable and controls.

## 4. Research questions

### RQ1 - Baseline reversal

Are extreme negative hourly BTC/USDT returns followed by positive cumulative returns over the next 12 hours?

### RQ2 - Weekend timing

Are post-sell-off returns different after weekend shocks compared with weekday shocks?

### RQ3 - Market activity

Are post-sell-off returns different when the shock is preceded by relatively low trading activity?

### RQ4 - Incremental weekend information

Does weekend timing retain explanatory information after controlling for pre-shock activity, shock magnitude, recent volatility, and time of day?

### RQ5 - Asymmetry

Does post-event behaviour following extreme negative shocks differ from behaviour following comparable positive shocks?

### RQ6 - Economic significance

Can a simple rule based on the relationship produce held-out performance after explicit fees and slippage?

## 5. Hypotheses

The hypotheses are tested separately. Every pre-specified analysis is retained regardless of whether another hypothesis is supported.

### H1 - General reversal

\[
H_1:\ E(Y_i^{12}\mid i\text{ is an eligible accepted negative shock})>0
\]

where \(Y_i^{12}\) is the 12-hour post-event log return for eligible negative event \(i\).

Null:

\[
H_{0,1}:\ E(Y_i^{12})\leq0
\]

### H2 - Weekend difference

\[
H_2:\ E(Y_i^{12}\mid Weekend_i=1)>
E(Y_i^{12}\mid Weekend_i=0)
\]

Null: the weekend-minus-weekday difference is non-positive.

### H3 - Low-activity difference

The primary H3 test is:

\[
H_3:\ \beta_A>0
\]

in Model 2, where a higher `LowActivityScore` means lower preceding activity. The bottom-third comparison has the same directional expectation,

\[
E(Y_i^{12}\mid LowActivity_i=1)>
E(Y_i^{12}\mid LowActivity_i=0),
\]

but is descriptive corroboration rather than a second H3 p-value. Null: \(\beta_A\leq0\).

### H4 - Incremental weekend information

Weekend timing is hypothesised to retain a positive association with 12-hour post-shock returns after adding pre-shock activity and controls.

The primary H4 test is \(\beta_W>0\) in Model 2. Null: \(\beta_W\leq0\).

The change in the weekend coefficient between the weekend-only and activity-controlled models is also reported. Attenuation is consistent with a market-state explanation but is not proof of mediation or causality.

### H5 - Shock asymmetry

Post-event behaviour following negative shocks differs from post-event behaviour following comparable positive shocks.

This is a two-sided asymmetry hypothesis. Positive shocks are a comparison test, not a perfect placebo.

### Decision rules

- H1-H4 are directional, one-sided tests fixed before outcome inspection.
- Report two-sided 95% confidence intervals even when the pre-specified test is one-sided.
- Within each period, apply Holm's step-down correction once to the four raw p-values from Model 0 H1, Model 1 H2, and Model 2 H3-H4.
- If a planned coefficient/test is not estimable because of zero variation, rank deficiency, or insufficient residual degrees of freedom, display its raw and adjusted p-values as unavailable and label that hypothesis inconclusive. For the sole purpose of retaining the pre-specified four-test Holm family for other estimable hypotheses, insert a conservative value of 1.0 in that test's family slot; never shrink the family after seeing feasibility or results.
- Use \(\alpha=0.05\). A hypothesis is labelled **supported in held-out data** only when the estimate has the hypothesised sign and its held-out Holm-adjusted p-value is below 0.05.
- Development-period tests are reported as development evidence, not confirmatory evidence. H5 is secondary, two-sided, and outside the H1-H4 Holm family.
- A non-significant result is labelled unsupported or inconclusive, not accepted as proof of the null.

## 6. Falsification and interpretation

Evidence against the proposed weekend-dip story includes:

- no positive average return following extreme negative shocks;
- no weekend-weekday difference;
- no weekend coefficient after activity and controls;
- an effect driven by very few observations;
- instability across periods;
- failure in the held-out period;
- disappearance after plausible execution delay or costs.

Possible conclusions:

| Evidence | Permitted conclusion |
|---|---|
| Reversal exists; weekend survives controls | Weekend contains incremental information beyond measured activity |
| Raw weekend difference disappears after activity | Activity helps explain the apparent weekend association |
| Activity matters; weekend does not | Reversal is state-dependent but not specifically a weekend anomaly |
| Weekend matters; activity does not | Calendar association exists; proposed activity explanation is unsupported |
| General reversal only | Sell-offs reverse, but neither conditioning variable explains cross-event variation |
| Development effect fails held out | The relationship is unstable or sample-specific |
| No reversal | Prior findings do not generalise to this venue, period, or definition |
| Continuation follows shocks | Buying dips is unsupported at the selected horizon |
| Statistical effect disappears after costs | Predictability is not economically exploitable under assumed execution |

Failure to reject a null hypothesis does not establish the absence of an effect when inference is imprecise.

## 7. Data

### 7.1 Primary market

- Exchange: Binance
- Market: spot
- Symbol: BTCUSDT
- Bar frequency: one hour
- Timezone: UTC
- Analysis sample: 2020-01-01 through 2026-07-31
- Development period: 2020-01-01 through 2023-12-31
- Held-out period: 2024-01-01 through 2026-07-31

The held-out period is historical pseudo-out-of-sample evidence, not a live forward test.

Assign dates from the completed candle's source close time after conversion to UTC. Standard Binance hourly candles use half-open intervals. Their source `close_time` is one source unit before the next bar: one millisecond before 2025 and one microsecond from 2025 onward. Section 7.5 defines the deterministic exclusion of the small number of historical source rows that violate this relation.

“Through 2026-07-31” includes the candle opening at 23:00 UTC that day and ending just before 2026-08-01. Preserve the original integer, detected unit, and converted UTC value. Never move a Friday candle into Saturday merely because the next bar opens at midnight.

### 7.2 Warm-up acquisition

Acquire data beginning 2019-10-01 so the January 2020 analysis start can have approximately 90 days of lagged history. Warm-up observations may initialise rolling features but are not analysis events or strategy trades.

If exact window requirements leave any early 2020 observation without full history, mark it ineligible rather than shortening the window.

### 7.3 Required raw fields

- candle open timestamp;
- candle close timestamp;
- open;
- high;
- low;
- close;
- BTC base volume;
- USDT quote volume;
- number of trades;
- taker-buy base and quote volumes, retained but not required in the primary analysis;
- the archive `ignore` field, retained for schema fidelity but never used analytically;
- source-file name, source-row number, original timestamp integers, and detected timestamp unit in staging metadata.

### 7.4 Data-source requirements

- Primary source: the official Binance Public Data monthly spot-kline archive at `data.binance.vision`.
- Download the BTCUSDT `1h` monthly ZIP archives from 2019-10 through 2026-07 inclusive, using the documented path pattern `data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM.zip`.
- Download and verify every companion `.CHECKSUM` file before accepting an archive. Also record a local SHA-256 digest for every raw file and for the completed manifest.
- Preserve the original ZIP files, checksum files, and manifest unchanged. Extracted/concatenated data are staging artifacts, not replacements for raw files.
- Do not silently mix the bulk archive with REST data. An unavailable official archive or checksum prevents acceptance of the snapshot; no fallback source is included in the specified design.
- Record source URL, access date, symbol, market, interval, file period, source timezone, schema, and timestamp-unit rule.
- Binance spot archive timestamps are milliseconds before 2025-01-01 and microseconds from 2025-01-01. Detect the unit by integer magnitude, verify it against the file period, and reject mixed or unexpected units within a monthly file.
- Parse the exact documented 12-column kline schema. Permit at most one first-row header only when all 12 cells exactly match the documented field names; record its presence and remove it as metadata. Any other apparent header, schema drift, extra column, or non-parseable row fails source validation.
- The public archive requires no credentials, and raw data are preserved unchanged.

### 7.5 Data-quality rules

- At most one row per expected UTC hour. Reconcile observed timestamps against the complete expected hourly calendar; every absent hour is an explicit gap and segment boundary, never a silently fabricated row.
- Strict chronological ordering.
- Each `open_time` must be aligned to a whole UTC hour. A row is accepted for analysis only when its converted `close_time` is exactly one source quantum before `open_time + 1 hour`.
- Preserve but exclude any otherwise parseable row with a non-standard `close_time` relation. Record its source evidence and exclusion reason, count its open hour as an explicit gap, and start a new segment after the gap. Do not alter its timestamps, impute it, or replace it from REST. Wrong units, unaligned opens, duplicates, schema drift, non-parseable values, and violated finite/OHLC invariants fail source validation.
- No return across a missing-hour gap.
- Prices must be finite and positive.
- Volumes must be finite and non-negative; trade counts must be non-negative integer-compatible values.
- Require \(High\geq\max(Open,Close)\) and \(Low\leq\min(Open,Close)\).
- Remove incomplete final candles.
- Require contiguous history for rolling features.
- Create a deterministic `contiguous_segment_id` that increments whenever adjacent `open_time` values are not exactly one hour apart. Calculate returns and all rolling features within segment only.
- Do not impute, forward-fill, interpolate, or synthesize missing OHLCV/trade observations.
- Duplicate `open_time` values fail source validation; neither the first nor the last duplicate is silently retained.
- Preserve row counts and exclusion counts at raw-file, staging, cleaned, feature-eligible, event, model, and trade stages.
- Manually audit the 20 largest absolute valid hourly returns, the 20 largest absolute valid contiguous gaps between the previous close and current open, and every zero-base-volume, zero-quote-volume, or zero-trade row. Record the reviewed timestamp, flag, source row, disposition, and reason; the audit must not remove genuine extreme market moves.
- Exact previous-close/current-open equality is not required because the fields can represent distinct last and first trades. Only the ranked 20 largest absolute log gaps require individual review. A large return, ranked open gap, or zero-activity candle is a retain-and-review diagnostic, not an exclusion rule. No discretionary row deletion is allowed.

### 7.6 Reproducibility

Core transformations are implemented in importable functions. The notebook executes those functions and presents the results. Persisted artifacts record the run identifier, creation time, specification and configuration versions and checksums, input checksums, code identity, row count, schema and output checksum. Row additions, exclusions and eligibility changes reconcile between consecutive stages.

Raw files and accepted analytical outputs are immutable. Outputs are stored by run identifier. A downstream stage rejects an artifact whose required configuration, input, schema or recorded identity does not match. Validation failures stop dependent analysis; declared diagnostics remain visible in the recorded evidence.

Sorting is deterministic. Stochastic calculations use the integer seed `20260901` with NumPy `Generator(PCG64)`, and the NumPy version is recorded. The default absolute and relative tolerance for numerical hand fixtures and zero-coefficient checks is `1e-12`, unless a test specifies a stricter exact comparison. Analytical tables and figures are generated from code, without manual spreadsheet or notebook-only corrections.

## 8. Variables

### 8.1 Hourly return

For completed candle \(t\):

\[
r_t=\ln\left(\frac{Close_t}{Close_{t-1}}\right)
\]

\(r_t\) is undefined when the previous candle is not the immediately preceding UTC hour.

### 8.2 Rolling shock z-score

Use the previous 720 valid, contiguous hourly returns:

\[
z_t=
\frac{r_t-\mu_{t-720:t-1}}
{\sigma_{t-720:t-1}}
\]

The current return is excluded from the rolling mean and standard deviation.

The sample standard deviation uses \(ddof=1\), matching the pandas rolling standard deviation convention and the output metadata.

### 8.3 Candidate shocks

Primary negative candidate:

\[
z_t<-2.5
\]

Positive comparison candidate:

\[
z_t>2.5
\]

The z-score is a relative extremeness measure. It is not assigned a Gaussian tail probability.

### 8.4 Event de-clustering

Candidates are processed chronologically:

- Accept the first chronological candidate that satisfies `shock_eligible_hour`; activity, RV, future-bar availability, and outcome values play no role in acceptance.
- Reject any later same-sign candidate occurring less than 24 hours after the last accepted same-sign event.
- A candidate exactly 24 hours later is eligible.
- Construct negative and positive event sets separately.

Apply cooldown to chronological candidates before checking future data. Once accepted, an event stays accepted even if a later model cannot use it because a covariate or outcome is missing. Record model eligibility separately. This prevents future availability from changing event identity.

Run cooldown continuously from warm-up through the analysis cutoff. Do not reset it at 2020-01-01 or at the 2024 split. A prior warm-up or development candidate may suppress a later candidate within 24 clock hours. Measure elapsed time from converted close timestamps, not row counts.

The primary choice retains the first candidate in a cluster. Retaining the most extreme candidate in a completed cluster is a retrospective event-study diagnostic only because it requires later within-cluster information. It must not generate a backtest, execution metric, or claim of tradability.

### 8.5 Weekend

Classify using the shock candle’s completed UTC close timestamp:

\[
Weekend_t=
\begin{cases}
1,&\text{Saturday or Sunday UTC}\\
0,&\text{Monday through Friday UTC}
\end{cases}
\]

Weekend refers to shock timing, not necessarily the entire 12-hour holding period. For a weekend event with a complete 12-hour horizon, define `crosses_into_monday_12h=1` when at least one held candle from \(t+1\) through \(t+12\) has a completed close timestamp on Monday UTC. Define the complementary weekend-event group as fully weekend-contained. Do not apply this comparison to weekday events.

### 8.6 Pre-shock activity

Let \(Q_t\) be USDT quote volume during candle \(t\). Define trailing 24-hour activity:

\[
A_t=\sum_{j=1}^{24}Q_{t-j}
\]

The shock-hour volume \(Q_t\) is excluded.

Compare \(A_t\) with the immediately preceding 2,160 hourly trailing-activity values:

\[
P_t=
\frac{1}{2160}
\sum_{k=1}^{2160}
\mathbf{1}(A_{t-k}\leq A_t)
\]

Require the complete historical reference window.

The reference values are exactly \(A_{t-1},\ldots,A_{t-2160}\). They must correspond to consecutive hourly candles in the same contiguous segment as \(t\); do not skip a missing or invalid value and reach farther back to replace it. Because every \(A\) value itself requires 24 prior quote-volume observations, primary activity eligibility requires uninterrupted quote-volume data from \(t-2184\) through \(t-1\).

The empirical-percentile tie convention is locked as “less than or equal to,” exactly as shown in the formula. The denominator is always 2,160 for an eligible observation; no shortened reference window is allowed.

Define:

\[
LowActivityScore_t=1-P_t
\]

and:

\[
LowActivity_t=\mathbf{1}\left(P_t<\frac{1}{3}\right)
\]

Primary regressions use the continuous score; descriptive groups and strategy filters use the bottom-third indicator.

Quote volume measures market activity, not direct liquidity.

### 8.7 Recent realised volatility

Primary pre-shock volatility control:

\[
RV24_t=
\sqrt{\sum_{j=1}^{24}r_{t-j}^{2}}
\]

It excludes the shock return.

### 8.8 Shock magnitude

For negative shocks:

\[
ShockMagnitude_t=-z_t
\]

Higher values represent more extreme negative standardised returns.

### 8.9 Primary event outcome

For accepted shock candle \(t\):

\[
Y_t^{12}=
\ln\left(\frac{Close_{t+12}}{Close_t}\right)
\]

The complete 12-hour horizon must be available and contiguous.

Secondary outcomes:

\[
Y_t^6,\quad Y_t^{24}
\]

The 12-hour outcome is primary. Six and 24 hours are robustness horizons.

An outcome is contiguous only when every intervening hourly candle from \(t+1\) through the endpoint is present. Availability of the two endpoint prices alone is insufficient.

## 9. Feasibility audit

The feasibility audit precedes post-event return inspection and contains counts only:

- candidate negative shocks;
- accepted negative shocks;
- candidate and accepted positive shocks;
- events by year;
- events in development and held-out periods;
- weekend and weekday events;
- low-activity and other events;
- all four weekend/activity cells;
- events per 1,000 eligible weekend and weekday hours.

Use two denominators rather than one ambiguous “eligible hour”:

- `shock_eligible_hour`: inside the relevant analysis split with a valid gap-aware return and complete contiguous 720-return shock window;
- `full_feature_eligible_hour`: shock-eligible and also has complete contiguous primary activity and RV24 history, including all immediately preceding 2,160 activity-reference values.

Report negative accepted events per 1,000 shock-eligible weekend and weekday hours. Use the full-feature denominator for activity-cell rates. Neither denominator requires a future outcome. Accepted-event counts are shown before and after horizon/model/strategy eligibility filters so missing data are never hidden.

The count-only audit may calculate timestamp-only structural flags indicating whether a required future bar exists, is contiguous, and remains in the split. It must not calculate, expose, inspect, or summarise any future price ratio, return, trade P&L, or outcome value.

Thresholds are independent of group balance. Small subgroups retain their original definitions and are labelled descriptive or exploratory when reliable inference is unavailable.

## 10. Descriptive and event-study analysis

For accepted negative events, report:

- sample size;
- shock raw return and z-score;
- pre-shock activity percentile;
- pre-shock realised volatility;
- mean and median post-event return;
- standard deviation and quantiles;
- hit rate;
- confidence interval;
- largest winners and losers;
- events by year and group.

Plot the average cumulative log-return path through 24 hours. Use one fixed, balanced sample whose full path is contiguous and contained in the split. Show its event count and note that it may be smaller than the primary 12-hour sample. The primary test still uses the largest eligible 12-hour sample.

Also report the mean, median, and count of 12-hour returns for eligible non-candidate hours (`z_t >= -2.5`) with complete, split-contained horizons. This overlapping market-drift reference is context only. It is not a matched comparison and does not replace H1.

Compare:

- all negative events;
- weekend versus weekday;
- low activity versus other activity;
- weekend and low activity versus remaining events.

For raw group comparisons, estimate the difference in means through the equivalent intercept-plus-group-indicator regression, report HC3 inference, and show a calendar-month block-bootstrap interval alongside medians and distribution summaries. Do not rely on a normality assumption or a single rank test alone.

Use the pre-specified calendar-month block bootstrap for the principal robust group-comparison interval. Do not add a simple label-permutation analysis: weekend labels and market regimes are not exchangeable over calendar time, and the block bootstrap already supplies the declared robust comparison.

## 11. Regression analysis

The unit of analysis is an accepted negative-shock event.

### Model 0 - Baseline reversal

\[
Y_i^{12}=\alpha+\varepsilon_i
\]

H1 tests whether \(\alpha>0\).

This is an absolute rebound test. The deterministic market-drift reference uses overlapping hours and has no pre-specified formal comparison, so H1 does not establish incremental predictability relative to ordinary hours. A positive event mean close to or below that reference indicates only a positive post-shock return.

### Model 1 - Raw weekend association

\[
Y_i^{12}=\alpha+\beta_W Weekend_i+\varepsilon_i
\]

H2 tests whether \(\beta_W>0\).

### Model 2 - Weekend versus activity

\[
Y_i^{12}=
\alpha+
\beta_W Weekend_i+
\beta_A LowActivityScore_i+
\theta_1 ShockMagnitude_i+
\theta_2 RV24_i+
\delta^\top TimeBlock_i+
\varepsilon_i
\]

Use four pre-defined six-hour UTC blocks:

- 00:00-05:59;
- 06:00-11:59;
- 12:00-17:59;
- 18:00-23:59.

Include three indicator columns and omit one pre-declared reference block to avoid the dummy-variable trap. Use 00:00-05:59 UTC as the reference category.

H3 tests whether \(\beta_A>0\). H4 tests whether \(\beta_W>0\). Model 1 uses its maximal eligible 12-hour event sample. For the descriptive attenuation calculation only, re-estimate Model 1 on the exact complete-case rows used by Model 2 (`Model 1-CC`), then compare its weekend coefficient with Model 2. Report both the absolute change and

\[
Attenuation=\frac{\beta_{W,Model\ 1-CC}-\beta_{W,Model\ 2}}{|\beta_{W,Model\ 1-CC}|}.
\]

Report the percentage as undefined when the Model 1-CC coefficient is zero within the stored numerical tolerance. A negative value means amplification rather than attenuation. Never compare the maximal-sample Model 1 coefficient directly with Model 2 to claim attenuation, and never describe this comparison as causal mediation.

### Model 3 - Interaction

\[
Y_i^{12}=
\alpha+
\beta_W Weekend_i+
\beta_A LowActivityScore_i+
\beta_{WA}(Weekend_i\times LowActivityScore_i)+
\theta_1 ShockMagnitude_i+
\theta_2 RV24_i+
\delta^\top TimeBlock_i+
\varepsilon_i
\]

The interaction is always secondary. Interpret its size only if every weekend-by-activity cell has at least 10 events and complete-case \(N\) is at least ten times the number of Model 3 coefficients. Otherwise show an estimable result but label it exploratory and underpowered.

### Inference

- Estimate Models 0-3 separately in development and held-out samples. Held-out H1-H4 results determine the confirmatory labels; development results provide context. Do not generate a pooled full-sample regression for the submitted core project.
- Report coefficient estimates, HC3 standard errors, two-sided 95% confidence intervals, and the pre-specified one-sided p-values for H1-H4. Use finite-sample t reference probabilities with residual degrees of freedom and record the software convention.
- Use a calendar-month block pairs bootstrap as a robustness check: from the event-containing UTC months in the relevant split, sample the original number of month blocks with replacement, carry every event in each selected block, assign a synthetic block copy identifier, and refit the model. Target 10,000 valid replicates with the fixed project seed and a maximum of 20,000 attempted draws. Report attempted, valid, empty, and rank-deficient counts and percentile intervals. If fewer than 10,000 valid replicates are obtained, label the bootstrap robustness inconclusive rather than silently changing the model. The HC3/Holm result remains primary.
- Report raw and Holm-adjusted p-values for the primary H1-H4 family.
- Preserve four Holm slots even when a planned test is unestimable, using the locked conservative placeholder rule; expose the unestimable status rather than displaying the placeholder as an observed raw p-value.
- Report sample size for every model.
- Diagnose residuals, influence, multicollinearity, and coefficient stability.
- Flag Cook's distance above \(4/N\) and report leave-one-flagged-event-out coefficient ranges; never delete an influential genuine event automatically. Report VIF/condition diagnostics without selecting controls according to their effect on significance.
- Do not interpret coefficients causally.

## 12. Positive-shock comparison

Apply the same:

- \(z_t>2.5\) threshold;
- 24-hour same-sign cooldown;
- weekend and activity definitions;
- 12-hour primary outcome;
- descriptive and controlled comparison structure.

Compare direction, magnitude, event counts, and shock severity with negative events. For the formal H5 comparison, stack eligible positive and negative events and estimate a pre-specified pooled sign model in each split:

\[
Y_i^{12}=\alpha+\gamma Positive_i+\beta_W Weekend_i+\beta_A LowActivityScore_i+\lambda |z_i|+\theta RV24_i+\delta^\top TimeBlock_i+\varepsilon_i
\]

where \(Positive_i=1\) for positive shocks. H5 tests \(\gamma\neq0\) using a two-sided HC3 test. Also report sign-specific means and models so that a difference is not mistaken for reversal in either sign. H5 is secondary and is not included in the H1-H4 Holm family.

Sign-specific cooldowns can allow positive and negative events within one 24-hour episode. Count overlapping 12-hour windows across signs. For H5 only, rebuild events with one any-sign 24-hour cooldown and repeat the comparison. Do not replace the primary sign-specific event sets.

## 13. Trading strategy

### 13.1 Signal

Generate a signal only after an accepted negative-shock candle is complete.

### 13.2 Strategy variants

Report all pre-specified variants:

1. All-shock benchmark: trade every accepted negative shock.
2. Weekend strategy: trade accepted weekend negative shocks.
3. Low-activity strategy: trade accepted negative shocks with \(P_t<1/3\).
4. Combined strategy: trade accepted negative shocks that are both weekend and low activity.

The weekend strategy is the primary candidate. The all-shock strategy is the principal conditional benchmark. All four variants are retained, without retrospective selection of the best performer.

Run each strategy in a separate account starting at 1.0. Compute every variant, even with few trades. If the held-out combined strategy completes fewer than 20 trades, show it but label annualised performance descriptive and highly unstable.

The all-shock and weekend variants require an accepted negative event plus valid contiguous execution bars; they do not require an activity percentile. The low-activity and combined variants additionally require the primary activity feature. Every accepted event that fails a strategy-specific rule remains in the event audit with an explicit rejection reason.

### 13.3 Execution

- Enter long at the next hourly candle open after the completed signal candle.
- Hold for 12 completed hours.
- Exit at the corresponding hourly close.
- Use one position at a time.
- Allocate 100% of current strategy capital to each accepted trade.
- Use no leverage.
- Use no short position.
- Use no stop-loss or take-profit.
- Remain in USDT between positions.

The next-hour open is an approximation to immediate post-signal execution. Test a one-hour-delayed entry as execution robustness.

For delayed execution, enter at the open one complete hour after the primary entry and hold for the same 12 completed hours from the delayed entry. This changes execution timing only, not the signal or event definition.

With candle \(t\) as the completed shock candle, the exact primary indices are:

\[
Entry=Open_{t+1},\qquad Exit=Close_{t+12}
\]

so the held candles are \(t+1,\ldots,t+12\). The delayed indices are:

\[
Entry_{delay}=Open_{t+2},\qquad Exit_{delay}=Close_{t+13}
\]

“Next-hour open” means the next bar, with no extra one-hour delay. A signal skipped because a position is open stays in the event study and is logged as a backtest rejection. Primary 12-hour trades should not overlap under the 24-hour cooldown, but still enforce this rule in robustness runs.

Require the entry bar and every held bar to be contiguous with the signal bar. If the next bar is missing, no synthetic fill or later substitute entry is allowed.

### 13.4 Costs

Primary assumptions:

- trading fee \(f=0.001\), or 10 basis points per side;
- slippage \(s=0.0002\), or 2 basis points per side.

Slippage is a reduced-form haircut for spread, latency, and small-order execution uncertainty. It is an assumed scenario rather than an estimate from historical order-book data. The backtest makes no institutional-capacity claim; market impact beyond the declared scenarios is not modelled.

Pre-declared cost scenarios:

| Scenario | Fee per side | Slippage per side | Approximate round trip |
|---|---:|---:|---:|
| Low | 5 bp | 1 bp | 12 bp |
| Primary | 10 bp | 2 bp | 24 bp |
| High | 10 bp | 10 bp | 40 bp |

Exact net returns use the multiplicative formula below rather than subtracting the approximate round-trip number.

For entry open \(O\) and exit close \(C\):

\[
BuyFill=O(1+s)
\]

\[
SellFill=C(1-s)
\]

The net wealth factor is:

\[
G_{net}=(1-f)^2\frac{SellFill}{BuyFill}
\]

and:

\[
R_{net}=G_{net}-1
\]

Report gross, primary net, alternative-cost, and break-even-cost results. Taxes are excluded.

### 13.5 Portfolio accounting

- Use arithmetic net returns for compounding.
- Start from a declared normalised capital, such as 1.0.
- Maintain an hourly marked-to-market equity curve.
- No trade may extend beyond the data cutoff.
- Record entry, exit, signal time, filter values, gross return, each cost component, and net return.

At entry, units are \(W_{pre}(1-f)/BuyFill\). At that bar's `open_time`, mark the position using the raw open so the entry cost appears immediately. During the trade, value equity at each observed hourly close. At exit, replace the asset mark with \(W_{post}=units\times SellFill\times(1-f)\). Apply the exit cost once.

Build the equity index on the split's complete UTC hourly calendar. Cash wealth may stay unchanged across a missing market-data hour; label that row `cash_calendar_mark`. Never create an invested asset price. This accounting rule does not permit filling OHLCV data.

### 13.6 Performance metrics

Report separately for development and held-out periods:

- trade count;
- average and median trade return;
- hit rate;
- cumulative gross return;
- cumulative net return;
- annualised return;
- annualised volatility;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- percentage of time invested;
- largest winner and loser;
- break-even symmetric all-in cost per side.

Compute the principal Sharpe ratio from daily marked-to-market strategy returns using 365 days and a disclosed zero risk-free-rate assumption. Report the instability of annualised metrics for sparse strategies.

Set net equity to 1.0 at the inclusive UTC split boundary. For each calendar day, use the last UTC hourly mark and keep unchanged cash days. Compute the first daily return against the initial 1.0 mark:

\[
Sharpe=\sqrt{365}\frac{\overline{R_d}}{s(R_d)}
\]

with sample standard deviation. Report Sharpe as undefined when daily volatility is zero. Annualised volatility is \(s(R_d)\sqrt{365}\). With \(D\) equal to exact elapsed UTC days between the configured inclusive split start and exclusive split end, annualised return is

\[
CAGR=\left(\frac{W_{end}}{W_{start}}\right)^{365/D}-1,
\]

This is calendar-time annualisation, not trade-frequency extrapolation. The exclusive split ends are 2024-01-01 00:00 UTC and 2026-08-01 00:00 UTC.

Calculate maximum drawdown from the hourly net equity curve, including the initial mark. Notional turnover is \(\sum_j(units_j BuyFill_j+units_j SellFill_j)\) divided by mean calendar day-end net equity. Exclude the synthetic initial mark from that mean.

Use net arithmetic returns for the main average trade, median trade, and hit rate; show gross values separately. A primary trade has 12 hours of exposure, from `open_time[t+1]` to the exclusive boundary one hour after `open_time[t+12]`. Divide total non-overlapping invested hours by exact split hours. Use the net curve for Sharpe, volatility, and drawdown.

For a zero-trade strategy, return, CAGR, drawdown, turnover, exposure, and annualised volatility are zero. Average/median trade return, hit rate, largest winner/loser, Sharpe, and break-even cost are not applicable. Do not display undefined statistics as zero.

For break-even cost, replace fee and slippage with one symmetric all-in cost per side, \(c\). Each trade factor is \((1-c)^2(C/O)\). Solve for the non-negative \(c\) that makes terminal split wealth 1.0 and report basis points per side. With no trades, report not applicable. If gross wealth is at most 1.0, report no positive break-even cost.

### 13.7 Benchmarks

- cash/USDT with zero assumed yield;
- BTC buy-and-hold, clearly identified as having different exposure;
- all-shock strategy;
- weekday-shock counterpart.

Buy-and-hold enters at each split's first candle open and exits at its final candle close, with gross returns and the same cost cases. It is an exposure benchmark rather than a risk-matched comparison. Benchmarks were specified before outcome inspection; no outcome-dependent benchmark selection is included.

## 14. Development and held-out protocol

### Development period

The 2020 to 2023 sample supports implementation checks, event-construction validation, count-only feasibility assessment, model verification and the pre-specified robustness analyses. Primary parameters are not optimised for development Sharpe.

### Pre-evaluation record

Before the first held-out evaluation, the specification, configuration, data, code and development outputs were recorded with their checksums, core validation was completed, and the evaluation was authorised. Saved run logs and manifests identify the accepted records and document execution attempts and corrections.

### Held-out period

Use 2024-July 2026 once for the final historical out-of-sample evaluation. Earlier observations may initialise lagged rolling features. Held-out outcomes may not influence model or strategy selection.

Keep each signal and its complete analysed horizon inside one split. Exclude a development event if its primary 12-hour window crosses into 2024. Apply the same rule separately to every alternative horizon and delayed trade. Exclude held-out results that cross the July 2026 cutoff. Reset strategy equity to 1.0 at each split start.

Implementation corrections and held-out execution attempts are documented in the saved run logs and manifests. Accepted historical outputs are preserved, and poor performance is not treated as a programming error.

Primary research conclusions for H1-H4 and primary economic conclusions come from the held-out split. Development and held-out estimates must both be shown. Do not pool the two periods after unblinding; the separate estimates provide the required comparison without adding a discretionary analysis.

## 15. Robustness hierarchy

Robustness changes one family at a time; the design contains no Cartesian search over alternative settings.

### Essential robustness

1. Shock thresholds: \(-2.0\), primary \(-2.5\), and \(-3.0\).
2. Horizons: 6, primary 12, and 24 hours.
3. Activity proxy: quote volume primary; number of trades secondary.
4. Execution: next-hour open primary; one-hour-delayed entry secondary.
5. Costs: low, primary, and high scenarios.
6. Inference: HC3 primary; month-block bootstrap secondary.
7. Positive shocks: symmetric \(+2.5\) comparison.

For each alternative threshold or cooldown, rebuild candidates and accepted events from scratch. Do not filter the primary event table to create an alternative set. For another holding horizon, change the exit consistently but keep signal timing and the one-position rule.

Change one robustness family at a time; do not build a grid. Horizon checks apply to both event outcomes and the matching fixed-hold strategy. The trade-count activity check rebuilds its trailing sum, percentile, score, and bottom-third flag with the same lag rules.

### Selected diagnostic robustness

- 12- and 48-hour cooldowns;
- calendar-year stability;
- leave-one-event-out and leave-one-year-out results;
- fully weekend-contained versus Monday-crossing events.

These diagnostics were frozen before development or held-out outcome inspection and must all be reported regardless of result. Secondary analyses must not replace the primary specification.

For 12-/48-hour cooldown checks, rebuild sign-specific events and rerun the primary 12-hour summaries, Models 0–2, and all four strategies. Keep every other setting fixed.

For leave-one-out checks, keep primary event IDs fixed; removing an event must not release a suppressed candidate. Report the resulting ranges for the H1 mean, H2 weekend coefficient, Model 2 activity/weekend coefficients, and weekend-strategy net return. Do this by event and by shock-close year. The report may summarise full generated outputs as ranges and changes in signs or conclusions.

The core project excludes the empirical lower-tail shock, most-extreme cluster rule, other activity windows, base-volume analysis, external stress classification, label permutation, and pooled full-sample regression. Any additional analysis introduced after outcome inspection would be exploratory and would not replace a confirmatory test.

## 16. Scope exclusions

The primary study excludes:

- machine learning;
- multiple primary assets or exchanges;
- order-book reconstruction;
- options or other derivatives-market data;
- causal estimation;
- live execution;
- leverage;
- dynamic position sizing;
- strategy parameter optimisation;
- complete news classification.

## 17. Computational efficiency

The notebook benchmarks rolling-feature construction and calendar-month bootstrap regression. The activity percentile is checked against a naive recount over \(n\) rows and a reference window of \(w\) values, whose time cost is \(O(nw)\). The implemented sorted window gives identical ranks with lower measured runtime.

Repeated bootstrap regression is the measured computational bottleneck. For \(S\) estimates, \(B\) bootstrap draws, \(N\) events and \(k\) regressors, approximate time cost is \(O(SBNk^2)\), with working memory \(O(Nk+B)\) per bootstrap. The event samples and small model matrices keep this practical. Measured timings and the implementation's scaling are reported in the notebook and report.

## 18. Risk interpretation

The strategy is concentrated in one volatile asset, one exchange, and one stablecoin pair. A plausible bad scenario is a sell-off caused by durable new information, followed by continued decline, elevated slippage, and no reversal.

Other risks include:

- sparse trades;
- dependence on a few large winners;
- regime change;
- USDT deviations;
- exchange-specific disruption;
- inability to execute at the observed next open;
- underestimated costs;
- continued crash risk during the fixed holding period.

## 19. Analytical outputs

Generated tables cover data quality, outcome-free feasibility counts and event incidence, event descriptives, weekend and activity comparisons, primary regressions, positive-shock comparisons, development versus held-out strategy metrics, cost sensitivity, robustness, and risk concentration. Group and model results include their sample sizes and uncertainty where applicable.

Figures show prices and accepted events, average post-event return paths, outcome distributions by weekend and activity, gross and net equity, drawdown, and calendar-year stability. The standalone report summarises the evidence; the notebook, downloaded data snapshot and pipeline code provide its reproducible basis.

## 20. Interpretation

The models describe associations conditional on an extreme shock. A smaller weekend coefficient after activity controls is a descriptive change rather than evidence of causal mediation. Quote volume and trade count measure completed trading activity, without directly identifying liquidity, spreads or order-book depth.

The held-out evaluation uses historical data and is described as pseudo-out-of-sample evidence. Statistical support requires the pre-specified sign and multiplicity-adjusted decision rule; an unsupported test does not establish that an effect is absent. Economic conclusions use held-out net performance under the declared execution and cost assumptions. Null, negative, unstable and cost-sensitive results remain part of the evidence.
