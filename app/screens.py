"""The four dashboard screens (P9-02 … P9-05).

Each is a plain function taking no arguments, rendering into the current page.
`dashboard.py` owns routing, the masthead and the disclaimer, so no screen can
render without them.

The screens differ in audience and are laid out accordingly. *Today's alerts*
and *Ticker detail* are worked by an analyst deciding in about thirty seconds
whether something deserves a closer look, so they lead with the alert and put
the reasoning beside it. *Evaluation* is read by an examiner, so it leads with
the comparison and states the measurement rules on the page rather than
assuming them.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data, ui

HOUR = 3600
DAY = 86400

#: How each feature is put into words, with the comparison that makes it mean
#: something: "4.2x its own normal", never a bare "4.2".
_REASON = {
    "volume_z": lambda v: f"vol {v:+.1f} sd vs normal",
    "ret_rel_4h": lambda v: f"{v:+.1%} vs SPY 4h",
    "ret_rel_24h": lambda v: f"{v:+.1%} vs SPY 24h",
    "ret_4h": lambda v: f"{v:+.1%} 4h",
    "ret_24h": lambda v: f"{v:+.1%} 24h",
    "ret_120h": lambda v: f"{v:+.1%} 120h",
    "volatility": lambda v: f"vol'y {v:.2%}/h",
    "days_since_last_8k": lambda v: f"8-K {v:.0f}d ago",
    "days_since_last_earnings": lambda v: f"results {v:.0f}d ago",
    "hours_since_news": lambda v: f"news {v / 24:.1f}d ago",
    "news_count_24h": lambda v: f"{v:.0f} articles 24h",
    "trading_hours_to_close": lambda v: f"{v:.1f}h to close",
}

#: Reading order for a triage analyst, as SLOTS rather than a flat list: one
#: reason per idea, first available alternative wins.
#:
#: `UI-context.md` rule 1 names four things that must be in the row — volume
#: multiple, benchmark-relative move, hours since the last 8-K, news coverage —
#: and a flat list could not deliver them. It ran volume_z, ret_rel_4h,
#: ret_rel_24h, ret_4h and stopped at four, so on every alert that has a
#: benchmark-relative move the four slots went to volume plus three restatements
#: of the same move, and 8-K recency never appeared at all. Grouping the returns
#: into one "the move" slot spends each of the four on a different idea.
#:
#: (When this was written, 513 of 2,033 logged alerts carried a
#: benchmark-relative move. That share has since risen past half as the price
#: snapshot caught up, which changes how often the fallback below is reached
#: but not the reason the slots are grouped. `data.feature_coverage` computes
#: the current figure; the caption on screen reads it from there.)
_SLOTS = (
    ("volume_z",),
    # The move, benchmark-relative where we have it. `ret_rel_*` is missing
    # whenever a bar was scored past the benchmark's own newest bar, so the raw
    # return is the fallback rather than a second slot. The share varies with
    # how far the benchmark's data extends — see `data.feature_coverage`.
    ("ret_rel_4h", "ret_rel_24h", "ret_4h", "ret_24h"),
    ("days_since_last_8k",),
    ("hours_since_news", "news_count_24h"),
    # Beyond the four rule 1 requires, for the expanded view.
    ("ret_120h",),
    ("volatility",),
    ("trading_hours_to_close",),
    ("days_since_last_earnings",),
)


def _reasons(row: pd.Series, limit: int = 4) -> list[str]:
    """The features that fired this alert, in words with their units.

    Rule 1: reasons travel WITH the alert, in the row, never behind a click. A
    number with no reason attached is a black box, and the point of the screen
    is that a human can sanity-check it in half a minute.
    """
    out = []
    for slot in _SLOTS:
        for key in slot:
            if key in row.index and pd.notna(row.get(key)):
                out.append(_REASON[key](row[key]))
                break                      # one reason per idea, not three
        if len(out) >= limit:
            break
    return out or ["no features recorded for this alert"]


def _outcome(row: pd.Series) -> tuple[str, str]:
    """(state, words) for one alert — four states, not three.

    The state is read from the column `data.alerts_with_outcomes` derives from
    the alert's own timestamp against the answerable edge. It used to be read
    from the join alone, which conflated "the window has not closed" with "this
    database holds no outcome row" and so labelled 1,236 alerts pending whose
    windows had closed weeks earlier.
    """
    state = row.get("outcome_state")
    if not isinstance(state, str) or state not in data.OUTCOME_WORDS:
        state = "unscored"
    return state, data.OUTCOME_WORDS[state]


# --------------------------------------------------------------------------
# P9-02 — today's alerts
# --------------------------------------------------------------------------
def _queue_stats(view: pd.DataFrame, all_rows: pd.DataFrame) -> None:
    """The four numbers a triage analyst needs before anything else.

    Eye-tracking work on dashboards is consistent that the top-left carries
    most of the attention and that four to six figures above the fold is the
    limit before they stop being read. So this row answers "what is in my
    queue right now", not "how is the system configured" — the alert budget
    and universe size are real (rule 6) but they are context for a result, not
    the first question a queue poses. They now sit in the strip below.
    """
    split = data.split_hit_rates(view)
    resolved = split["resolved"]
    hours = data.window_hours()
    # Alerts on known-bad bars stay listed but feed no figure here: a 101×
    # "strongest" read off a day of inflated volumes is the data, not a stock.
    bad = int(view["incident"].sum()) if "incident" in view else 0
    view = data.counted(view)
    strongest = view.apply(
        lambda r: ui.strength(r["score"], r["threshold"], r["detector"])[2], axis=1).max() \
        if not view.empty else float("nan")
    state = (view["outcome_state"] if "outcome_state" in view
             else pd.Series("unscored", index=view.index))
    open_n = int((state == "open").sum())
    unscored_n = int((state == "unscored").sum())

    c = st.columns(5)
    # No `delta` here: Streamlit renders one with a directional arrow, and an
    # arrow beside "of N logged" reads as a trend when it is a denominator.
    c[0].metric(f"In view · of {ui.num(len(all_rows))}", ui.num(len(view) + bad),
                help="Alerts matching the filters above, out of every alert "
                     "ever logged."
                     + (f" {ui.num(bad)} of them sit on bars inside a recorded "
                        f"data incident and are left out of the other four "
                        f"figures." if bad else ""))
    c[1].metric("Awaiting outcome", ui.num(open_n),
                help=f"The {hours}-hour window (wall-clock) has not closed yet, "
                     f"so these cannot be graded. They are excluded from the "
                     f"hit rate rather than counted as misses.")
    c[2].metric("Not scored", ui.num(unscored_n),
                help="The window closed, but no outcome is on record for the "
                     "alert — neither in live-log/outcomes.csv nor in this "
                     "database. Not a miss and not pending — an "
                     "answer nobody has looked up. Kept as its own count so "
                     "the hit rate's denominator is not mistaken for the set "
                     "of alerts that could be graded.")
    c[3].metric("Strongest", f"{strongest:.1f}×" if pd.notna(strongest) else "—",
                help="Highest multiple of its own alert threshold in this view. "
                     "NOT a probability.")
    # Rule 7: unscheduled is the headline, never the pooled figure. Results
    # announcements are published weeks in advance and would carry the number.
    c[4].metric("Unscheduled hit rate",
                ui.pct(split["unscheduled_rate"], 1)
                if split["unscheduled_rate"] is not None else "—",
                help=(f"{split['unscheduled']} of {resolved} graded alerts were "
                      f"followed by an UNSCHEDULED 8-K within {hours} hours; "
                      f"{split['scheduled']} by a scheduled one (a results "
                      f"announcement). One alert can be followed by both. Any "
                      f"8-K at all, routine ones included, is "
                      f"{ui.pct(split['pooled'], 1)}, which this project does "
                      f"not report as one number." if resolved else
                      "No alert in this view has both a closed window and an "
                      "outcome on record, so there is no rate to quote. An "
                      "empty figure is reported rather than a flattering one."))


def _volume_trend(df: pd.DataFrame) -> None:
    """Alert volume over time — named in the triage literature as a core metric.

    It is the fastest way to see the thing a count cannot show: whether today
    is unusual, and whether a step in the series is the market or a change we
    made. This log has one such step by construction — coverage widened from
    400 tickers to 1,500 on 2026-09-07, when the scheduled job went live — and
    a reader who cannot see it would mistake it for a signal. That is a fixed
    historical fact about the series, not a figure that drifts.
    """
    if df.empty:
        return
    if "episode_start" in df:
        df = df[df["episode_start"]]
    day = pd.to_datetime(df["ts_utc"], unit="s", utc=True).dt.floor("D")
    counts = day.value_counts().sort_index()
    if len(counts) < 2:
        return
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(),
                           marker_color=ui.SERIES))
    st.plotly_chart(ui.chart(fig, 120, "alerts"), width="stretch")


def alerts_today() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.section("No alerts yet")
        st.info("The alert log is empty. The system flags roughly **2 per stock "
                "per month by design**, so an empty queue is a normal state "
                "rather than a failure.")
        return

    newest = int(df["ts_utc"].max())
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        scope = st.radio("Window", ["Latest session", "Last 7 days", "All"],
                         horizontal=True, label_visibility="collapsed")
    # Default ON: a stock that stays elevated writes one alert per bar, and a
    # queue of seven identical rows for one stock is not seven things to look
    # at. The repeats are still in the log and one click away.
    new_only = c4.toggle("New flags only", value=True,
                         help="Hide repeat alerts on a stock already flagged by "
                              "the same detector within its "
                              f"{data.config()['decision']['horizon_hours']}-bar "
                              "episode. Every repeat stays in the log.")
    which = c2.selectbox("Detector",
                         ["All detectors"] + sorted(df["detector"].unique()),
                         label_visibility="collapsed")
    state = c3.selectbox("Outcome",
                         ["All outcomes", "8-K followed", "No 8-K",
                          "Window open", "Not scored"],
                         label_visibility="collapsed")

    # "Latest session" means the newest session with at least one alert on
    # sound bars. A session made entirely of incident rows is a queue of
    # nothing to triage; it is still under "Last 7 days" and "All".
    clean = data.counted(df)
    latest = int(clean["ts_utc"].max()) if not clean.empty else newest
    skipped = sorted({ui.short_utc(t)[:6] for t in
                      df.loc[df["ts_utc"] >= latest - latest % DAY + DAY, "ts_utc"]})
    cutoff = {"Latest session": latest - (latest % DAY),
              "Last 7 days": newest - 7 * DAY, "All": 0}[scope]
    view = df[df["ts_utc"] >= cutoff]
    if scope == "Latest session":
        view = view[view["ts_utc"] < cutoff + DAY]
        if skipped:
            ui.note(f"**Showing the newest session on sound bars.** Every alert "
                    f"after it ({', '.join(skipped)}) sits inside a recorded "
                    f"data incident; switch to *Last 7 days* to see them.")
    hidden = 0
    if new_only and "episode_start" in view:
        hidden = int((~view["episode_start"]).sum())
        view = view[view["episode_start"]]
    if which != "All detectors":
        view = view[view["detector"] == which]
    if state != "All outcomes":
        # Filter on the derived state, not on `filed`: "window open" and "not
        # scored" both show a missing `filed` and are different answers.
        want = {"8-K followed": "filed", "No 8-K": "none",
                "Window open": "open", "Not scored": "unscored"}[state]
        view = view[view["outcome_state"] == want]

    _queue_stats(view, df)
    if view.empty:
        st.info("No alerts match these filters. The system flags roughly 2 per "
                "stock per month by design.")
        return

    # Sort ONCE, on the source frame, so a selected table row maps back to its
    # alert by position. Sorting the rendered table separately would silently
    # open the wrong alert the moment the two orders diverged.
    view = view.assign(_mult=view.apply(
        lambda r: ui.strength(r["score"], r["threshold"], r["detector"])[2], axis=1)
    ).assign(_bad=view["incident"] if "incident" in view else False
    ).sort_values(["_bad", "_mult"], ascending=[True, False]
    ).reset_index(drop=True)

    table = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Strength": ui.strength(r["score"], r["threshold"], r["detector"])[1],
        "× thresh": round(r["_mult"], 1) if pd.notna(r["_mult"]) else None,
        "Detector": r["detector"],
        "Bar (UTC)": ui.short_utc(r["ts_utc"]),
        "Outcome": ("Data incident — not counted" if r.get("incident")
                    else _outcome(r)[1]),
        # Rule 1: the reasons travel WITH the alert, never behind a click.
        "Why it fired": " · ".join(_reasons(r)),
    } for _, r in view.iterrows()])

    ui.section(f"{len(table):,} alerts",
               "Strongest first — a work queue, not an index. Click any column "
               "to sort, or a row to open it below."
               + (f" {hidden:,} repeat alerts on stocks already flagged are "
                  f"hidden; switch off *New flags only* to see them."
                  if hidden else ""))
    if "incident" in view and view["incident"].any():
        for inc in data.incidents():
            ui.note(f"**Data incident ({inc['label']}) — these alerts are shown "
                    f"but counted in no rate or budget.** {inc['reason']}")
    picked = st.dataframe(
        table, width="stretch", hide_index=True, height=430,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "× thresh": st.column_config.NumberColumn(
                "× thresh", format="%.1f×", width="small",
                help="How far above its own threshold this score sat. NOT a "
                     "probability — these detectors emit raw statistics."),
            "Why it fired": st.column_config.TextColumn(width="large"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Strength": st.column_config.TextColumn(width="small"),
        })

    # Master-detail: the row is the summary, selecting it reveals the depth,
    # without losing the queue position that a page change would cost.
    sel = picked.selection.rows if picked and picked.selection else []
    if sel and sel[0] < len(view):
        _alert_detail(view.iloc[sel[0]])
    else:
        dist = data.strength_distribution(df)
        # Read from the log rather than written down: these figures were once
        # literals here and were wrong within a fortnight of being typed.
        where = (f"the median alert sits at {dist['median']:.1f}× its "
                 f"threshold and the 90th percentile at {dist['p90']:.1f}×"
                 if dist["median"] is not None else
                 "no alert in this log carries a usable threshold")
        st.caption(f"**Strength bands come from the observed distribution**, "
                   f"not round numbers: across the {ui.num(dist['n'])} logged "
                   f"rule-detector alerts, {where}. A learned policy's alert "
                   f"is shown as *Policy flag* instead — its score is a "
                   f"probability pressed against 1.0, not a multiple of a "
                   f"rule. Extreme ≥10×, Strong ≥4×, Elevated ≥2×. "
                   f"Select a row above to open it.")

    # Rule 1 names four reasons. Two of them are not always available, and
    # saying so once is better than leaving a reader to wonder which alerts are
    # missing an explanation and why.
    cov = data.feature_coverage(df)
    # Both counts are read off the log. They move every time the monitor runs —
    # the benchmark-relative share in particular went from a minority to a
    # majority as the price snapshot caught up — so a written-down figure here
    # is a figure that will be wrong by the next presentation.
    news_line = (
        "**News coverage is absent from every live alert** — "
        "`features.include_news_coverage` is off in the config, because "
        "whether the news channel helps is the Phase 8 experiment and the "
        "live monitor runs the same arm every earlier phase ran."
        if not cov["news"] else
        f"News coverage is recorded on {ui.num(cov['news'])} of "
        f"{ui.num(cov['total'])} alerts ({ui.pct(cov['news_pct'], 0)}).")
    st.caption(
        "**Why some rows show fewer than four reasons.** Rule 1 asks for four: "
        "volume multiple, benchmark-relative move, hours since the last 8-K, "
        f"and news coverage. {news_line} And the benchmark-relative move is "
        f"recorded on {ui.num(cov['benchmark_relative'])} of "
        f"{ui.num(cov['total'])} logged alerts "
        f"({ui.pct(cov['benchmark_relative_pct'], 0)}); the rest were scored "
        "on bars past the benchmark's own newest bar, and show the raw return "
        "instead. Neither gap is filled with a fabricated value.")

    with st.expander("Alert volume over time — is today unusual?"):
        _volume_trend(df)
        st.caption("New flags per day, repeats excluded. One step in this "
                   "series is ours, not the market's: coverage widened from 400 "
                   "tickers to the full 1,500 on 2026-09-07, so flags per day "
                   "rises there by construction. A second, from 2026-09-25, is "
                   "the learned policy joining the two rules."
                   + " Recorded data incidents — bad bars, not a busy "
                   "market: " + "; ".join(i["label"] for i in data.incidents())
                   + ".")


def _alert_detail(r: pd.Series) -> None:
    """One alert opened in place — the master-detail half of the queue."""
    _, words, mult = ui.strength(r["score"], r["threshold"], r["detector"])
    with st.container(border=True):
        a, b = st.columns([1, 3])
        a.metric(r["ticker"], ui.times(mult), delta=words, delta_color="off")
        a.caption(f"{r['detector']} · {_outcome(r)[1]}")
        b.markdown(f"**Bar** {ui.utc(r['ts_utc'])}  \n"
                   f"**Noticed** {ui.utc(r['raised_utc'], False)}  \n"
                   f"**Score** {r['score']:.4f} against a threshold of "
                   f"{r['threshold']:.4f}")
        b.markdown("**Why it fired** — " + " · ".join(_reasons(r, limit=8)))
        st.caption("Open **Ticker detail** for the price chart, the volume "
                   "z-score band and this company's filing history.")


# --------------------------------------------------------------------------
# P9-03 — ticker detail
# --------------------------------------------------------------------------
def ticker_detail() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.note("No alerts to inspect yet.")
        return

    # Open on something worth reading: the strongest rule alert of the newest
    # session on sound bars, not whichever ticker sorts first alphabetically.
    tickers = sorted(df["ticker"].unique())
    first = None
    clean = data.counted(df)
    if not clean.empty:
        newest = int(clean["ts_utc"].max())
        recent = clean[clean["ts_utc"] >= newest - newest % DAY]
        mult = recent.apply(lambda r: ui.strength(
            r["score"], r["threshold"], r["detector"])[2], axis=1)
        first = recent.loc[mult.fillna(-1).idxmax()]
    c1, c2 = st.columns([1, 2])
    ticker = c1.selectbox("Ticker", tickers, index=tickers.index(
        first["ticker"]) if first is not None else 0)
    rows = df[df["ticker"] == ticker].sort_values("ts_utc", ascending=False)
    # Keyed by alert, not by hour: two detectors often fire on the same bar,
    # and an hour-keyed list kept only one of them.
    label = {r.alert_id: f"{ui.utc(r.ts_utc, False)} · {r.detector}"
             for r in rows.itertuples()}
    ids = list(label)
    pick = c2.selectbox("Flagged hour", ids, format_func=label.get,
                        index=ids.index(first["alert_id"])
                        if first is not None and first["alert_id"] in label else 0)
    row = rows[rows["alert_id"] == pick].iloc[0]
    flagged = int(row["ts_utc"])

    key, words, mult = ui.strength(row["score"], row["threshold"], row["detector"])
    state, _ = _outcome(row)
    s = st.columns(4)
    ui.stat(s[0], "Ticker", ticker, row["detector"])
    ui.stat(s[1], "Strength", ui.times(mult),
            f"{words} — {ui.times(mult)} threshold" if mult == mult else
            f"{words} — a learned policy's P(FLAG), not a multiple of a rule")
    ui.stat(s[2], "Flagged", dt.datetime.fromtimestamp(
        int(flagged), dt.timezone.utc).strftime("%d %b %H:%M"), ui.utc(flagged))
    hours = data.window_hours()
    ui.stat(s[3], "Outcome", {"filed": "8-K followed", "none": "No 8-K",
                              "open": "Pending", "unscored": "Not scored"}[state],
            # Wall-clock, not trading hours. `outcome_window_hours` is elapsed
            # time — a company can file overnight or at a weekend — and calling
            # it trading hours would stretch two days into about seven.
            f"within {hours} hours (wall-clock)")

    live = state == "open"
    if live:
        ui.note("This alert's window is still open, so **nothing after the "
                "flagged hour is shown**. Revealing what happened next would "
                "turn a surveillance tool into a hindsight demo.")

    hi = int(flagged) if live else int(flagged) + hours * HOUR
    lo = int(flagged) - 30 * DAY
    price = data.bars(ticker, lo, hi)

    if not data.db_present():
        # This screen is the one that genuinely needs the database: bars, news
        # and filing history all live there and none of them are in the
        # committed CSV. Say so plainly instead of rendering three empty panels.
        ui.note("**No local database, so the evidence panels below are empty.** "
                "The alert and its recorded features come from the committed "
                "log, but the price bars, the news timeline and the filing "
                "history are read from `paths.db`, which is a rebuildable cache "
                "and is not committed. Rebuild it with the Phase 2 and 3 "
                "collectors, or restore the bootstrap snapshot.")
    elif price.empty:
        ui.note("No price bars stored for this window.")

    if not price.empty:
        ts = pd.to_datetime(price["ts_utc"], unit="s", utc=True)
        marker = dt.datetime.fromtimestamp(int(flagged), dt.timezone.utc)
        sev = ui.MARKER

        # Once the window has closed and an 8-K followed, its t0 is drawn
        # too: the bars after the flag are already shown then, and the filing
        # is what they lead up to. Never while the window is open.
        filed_at = None
        if not live and state == "filed" and pd.notna(row.get("t0_utc")):
            filed_at = dt.datetime.fromtimestamp(int(row["t0_utc"]),
                                                 dt.timezone.utc)

        def marks(f: go.Figure) -> None:
            f.add_vline(x=marker, line_dash="dash", line_color=sev, line_width=1.4)
            if filed_at is not None:
                f.add_vline(x=filed_at, line_dash="solid",
                            line_color=ui.CONTROL, line_width=1.4)

        ui.section("Price and volume",
                   "Hourly bars for the 30 days before the flag"
                   + ("" if live else f" and the {hours} hours after it")
                   + ". The dashed line is the flagged hour"
                   + (f"; the solid grey line is when the 8-K that followed "
                      f"became public ({ui.utc(int(row['t0_utc']))})"
                      if filed_at is not None else "") + ". Times are UTC.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=price["close"], mode="lines",
                                 name="close", line=dict(color=ui.SERIES, width=1.5)))
        marks(fig)
        st.plotly_chart(ui.chart(fig, 230, "close"), width="stretch")

        vol = go.Figure()
        vol.add_trace(go.Bar(x=ts, y=price["volume"], marker_color=ui.DIM))
        marks(vol)
        st.plotly_chart(ui.chart(vol, 150, "volume"), width="stretch")

        # The z-score band, computed with the SAME function the detector used.
        # A lookalike written here could drift from it and would then explain
        # the wrong thing convincingly.
        from src.pipeline.features import volume_zscore

        frame = price.set_index("ts_utc")[["close", "volume"]]
        z = volume_zscore(frame, data.config())["volume_z"]
        if z.notna().any():
            ui.section("Volume z-score",
                       "How unusual each hour's volume is against this stock's "
                       "own trailing normal. The dotted line is the alert "
                       "threshold.")
            band = go.Figure()
            band.add_trace(go.Scatter(x=ts, y=z.to_numpy(), mode="lines",
                                      line=dict(color=ui.SERIES, width=1.5)))
            if pd.notna(row.get("threshold")) and row["detector"] == "volume_zscore":
                band.add_hline(y=float(row["threshold"]), line_dash="dot",
                               line_color=sev, line_width=1.2)
            marks(band)
            st.plotly_chart(ui.chart(band, 165, "standard deviations"),
                            width="stretch")

    ui.section("Why this hour was flagged",
               "Each value beside the same measure's trailing normal for this "
               "ticker over the 30 days before the flag. A number alone means "
               "little — the comparison is what makes it mean something.")
    _feature_table(row, price)

    a, b = st.columns(2)
    with a:
        ui.section("News", "Headlines as published, quoted verbatim with their "
                           "publisher. Some carry analyst language — that is "
                           "the outlet's wording, not this tool's.")
        n = data.news(ticker, lo, hi)
        through = data.news_through()
        if n.empty and through is not None and hi > through:
            # Past the last article on record, an empty panel is a gap in the
            # data and must not be read as a quiet stretch.
            st.caption(f"No articles on record for this window. News was "
                       f"collected for the study window only — through "
                       f"{ui.utc(through, False)} — and is off for the live "
                       f"monitor, so this is a gap in the data, not a quiet "
                       f"stretch.")
        elif n.empty:
            st.caption("No articles in this window. A quiet stretch before a "
                       "move is the interesting shape, not a gap in the data — "
                       "every week of the study window was fetched for every "
                       "in-universe ticker.")
        else:
            for _, art in n.head(10).iterrows():
                st.markdown(
                    f'<div style="margin-bottom:.5rem"><span style="opacity:.65;font-size:.8rem">'
                    f'{ui.utc(art["published_utc"], False)}</span><br>'
                    f'<span style="font-size:.85rem;color:inherit">'
                    f'{art["title"]}</span> '
                    f'<span style="opacity:.65;font-size:.8rem">*{art["source_name"]}*</span></div>',
                    unsafe_allow_html=True)
    with b:
        ui.section("Filing history", "Past 8-K filings with their item codes.")
        f = data.filings(ticker)
        if f.empty:
            st.caption("No 8-K filings on record.")
        else:
            st.dataframe(pd.DataFrame({
                "accepted (UTC)": f["acceptance_utc"].map(lambda t: ui.utc(t, False)),
                "items": f["items"].fillna("—"),
            }), width="stretch", hide_index=True, height=320)


def _feature_table(row: pd.Series, price: pd.DataFrame) -> None:
    """Feature values beside the same measure's trailing normal (P9-03).

    The trailing normal is recomputed from this ticker's own bars over the 30
    days STRICTLY BEFORE the flagged hour — the whole series would put the
    spike inside the baseline it is being judged against, which is the same
    reason `volume_zscore` carries its own `shift(1)`.
    """
    from src.pipeline.features import returns, volume_zscore

    rows = []
    if not price.empty:
        frame = price.set_index("ts_utc")[["close", "volume"]]
        cfg = data.config()
        hist = pd.concat([returns(frame, cfg), volume_zscore(frame, cfg)], axis=1)
        hist = hist[hist.index < int(row["ts_utc"])]

        for col in ("volume_z", "ret_1h", "ret_4h", "ret_24h", "ret_120h"):
            if col not in row.index or pd.isna(row.get(col)) or col not in hist:
                continue
            past = hist[col].dropna()
            if past.empty:
                continue
            fmt = ((lambda v: f"{v:+.2f} sd") if col == "volume_z"
                   else (lambda v: f"{v:+.2%}"))
            rows.append({
                "feature": col,
                "at the flagged hour": fmt(row[col]),
                "trailing median": fmt(past.median()),
                "trailing 5–95%": f"{fmt(past.quantile(.05))} … {fmt(past.quantile(.95))}",
                "percentile": f"{(past < row[col]).mean() * 100:.0f}th",
            })

    # Context features have no price-derived trailing normal. They are shown
    # as-is rather than given a fabricated comparison.
    for col in ("days_since_last_8k", "hours_since_news", "news_count_24h"):
        if col in row.index and pd.notna(row.get(col)):
            rows.append({"feature": col,
                         "at the flagged hour": _REASON[col](row[col]),
                         "trailing median": "—", "trailing 5–95%": "—",
                         "percentile": "—"})

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No features recorded for this alert.")


# --------------------------------------------------------------------------
# P9-04 — evaluation
# --------------------------------------------------------------------------
#: The Phase 10 artifacts, newest FIRST — and "newest" means the 2026-09-10
#: re-run, not either 2026-09-08 file.
#:
#: The two older files are VOID, not merely superseded. They were produced by
#: an evaluation frame that gave a positive episode 48 bars and a quiet window
#: one, scoring a window by its maximum — so a positive had 48 chances to cross
#: the threshold against a quiet bar's one, at the same one-alert cost. On that
#: frame a pure random-noise scorer reached 29.6x lift, beating every tuned
#: detector in the table. Those numbers rank on window length, not detection,
#: and the arithmetic corrections applied to one of them fix three real defects
#: without touching that one. Preferring either over the re-run would put a
#: number on screen that the project's own correction note disowns.
_PHASE10 = (
    ("phase10/FINAL-test-evaluation-r2.csv",
     "**Phase 10 final evaluation on the sealed test set**, re-run once on "
     "2026-09-10 after the evaluation frame was corrected. The seal was opened "
     "with its reason recorded first and closed again afterwards; the decision "
     "is in the project's decision log. These are the results."),
    ("phase10/FINAL-test-evaluation-corrected.csv",
     "⚠ **VOID — shown only because the 2026-09-10 re-run is missing.** This is "
     "the 2026-09-08 run with arithmetic corrections applied. Its frame ranked "
     "on window length rather than detection: pure random noise scored 29.6x on "
     "it. Do not quote these numbers. See `CORRECTION-NOTE.md` beside the file."),
    ("phase10/FINAL-test-evaluation.csv",
     "⚠ **VOID — the original 2026-09-08 file, uncorrected.** Its frame ranked "
     "on window length (pure noise scored 29.6x), AND three arithmetic defects "
     "stand: a `max_precision` ceiling ten rows beat, `degenerate = False` on "
     "all 42 `always_quiet` rows, and rows for item codes `items.exclude` "
     "drops. Do not quote these numbers."),
)

#: Phase 8, the news ablation, on the corrected frame (2026-09-10). The
#: without-news arm IS the canonical validation comparison — news off is the
#: default — and the with-news arm was run on the same 314,694 windows and the
#: same 1,452 events three hours later with nothing in between that touches
#: gradient boosting. The 2026-09-08 pair (`p8-with-news-val.csv`,
#: `p8-without-news-val.csv`) used the void frame and is no longer read.
_P8_WITHOUT = "baseline-comparison-val-r2.csv"
_P8_WITH = "p8-with-news-val-r2.csv"

#: Kept as a standing caption rather than a footnote, because it disqualifies a
#: column that is on screen. Quoted from `CORRECTION-NOTE.md`.
#: Shown ONLY when a void artifact is on screen. It disqualifies the `lift`
#: column, so applying it to the 2026-09-10 re-run — whose frame is the fixed
#: one — would disown a number that is actually sound. Keyed off which file was
#: loaded rather than printed unconditionally.
_LIFT_CAVEAT = (
    "**The `lift vs floor` column in THIS file is not trustworthy.** It was "
    "produced under an evaluation frame that rewarded window length rather "
    "than detection: a positive episode got 48 bars and a quiet window got "
    "one, while a window scores as the maximum over its rows. Measured on that "
    "exact shape, **a scorer made of pure random noise reaches 29.6× lift, "
    "beating every detector in the table**; with both classes the same length "
    "the same noise scores 1.0×. Fixed in `9ada635`, and the sealed set was "
    "re-run on 2026-09-10 — load `FINAL-test-evaluation-r2.csv` instead. The "
    "figures here are kept and labelled rather than deleted."
)

#: The re-run carries a permanent `random_noise` row, so a reader can check the
#: null instead of being asked to trust it. Said on screen, because "the metric
#: is fixed" is a claim and the row is the evidence.
_NULL_NOTE = (
    "**Every row here can be checked against a `random_noise` baseline**, "
    "drawn at several seeds and run through the identical evaluation path. A "
    "scorer that knows nothing must land on the always-quiet floor; if it ever "
    "reads meaningfully above it, the evaluation frame has developed an "
    "asymmetry and no other row means anything until that is explained. This "
    "row exists because an earlier frame gave a 29.6× lift to pure noise and "
    "nothing in the table could reveal it."
)




def evaluation() -> None:
    table = pd.DataFrame()
    source = ""
    for name, caption in _PHASE10:
        table = data.comparison(name)
        if not table.empty:
            source = caption
            break
    if table.empty:
        table = data.comparison(_P8_WITHOUT)
    if table.empty:
        ui.note("No comparison table built yet. Run "
                "`python -m src.baselines.compare --split val --out …`")
        return

    # The base rate is read off the table actually on screen, not written
    # down: it differs between the validation and sealed-test frames (0.461%
    # against 0.319%), so a literal here would contradict the file it is
    # describing the moment either is loaded.
    pooled = table[table["slice"] == "all"]["base_rate"].dropna()
    rate = float(pooled.iloc[0]) if len(pooled) else None
    ui.note(
        "**Plain accuracy is not reported anywhere, by design.** "
        + (f"Only **{rate * 100:.3f}%** of hours in this frame precede an "
           f"event, so a system that always says \"nothing is coming\" is "
           f"**{(1 - rate) * 100:.2f}%** accurate and useless. "
           if rate is not None else
           "Well under one per cent of hours precede an event, so a system "
           "that always says \"nothing is coming\" scores over 99% and is "
           "useless. ")
        + "The function that would compute it raises an error instead. The "
          "headline is precision at the fixed alert budget.")
    ui.note(
        "**Scheduled and unscheduled are never pooled.** Scheduled events — "
        "results announcements — have dates published weeks ahead, so a run-up "
        "before one is far less interesting. Unscheduled events are the real "
        "target, and pooling would let the easy half carry the number.")
    # Only the void files carry the caveat; the re-run's lift is sound.
    void = not ("-r2" in source or "re-run" in source)
    if not void:
        ui.note(_NULL_NOTE)
    else:
        st.warning(_LIFT_CAVEAT)

    if source:
        st.caption(f"Source: {source}")

    c1, c2 = st.columns([1, 2])
    variants = sorted(table["t0_variant"].unique())
    variant = c1.selectbox(
        "t₀ variant", variants,
        index=variants.index("news_adjusted") if "news_adjusted" in variants else 0,
        help="`filing` uses the 8-K acceptance time. `news_adjusted` takes the "
             "earlier of that and the first news article — the honest clock, "
             "and the project's main contribution.")
    with c2:
        sl = st.radio("Slice", ["all", "scheduled", "unscheduled"],
                      horizontal=True)

    view = table[(table["t0_variant"] == variant) & (table["slice"] == sl)]
    view = view.sort_values("precision", ascending=False)

    ui.section("Detector comparison",
               "Precision at the fixed alert budget. `max_precision` is the "
               "ceiling: the budget is SPENT, not capped, so when it exceeds "
               "the number of events even a flawless detector cannot reach 1.0.")
    show = pd.DataFrame({
        "detector": view["baseline"],
        "precision": view["precision"].map(lambda v: ui.pct(v, 3)),
        "ceiling": view["max_precision"].map(lambda v: ui.pct(v, 2)),
        "lift vs floor": view["lift"].map(lambda v: f"{v:.1f}×"),
        "recall": view["recall"].map(lambda v: ui.pct(v, 1)),
        "median lead": view["median_lead_trading_h"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f} h"),
        "alerts": view["n_alerts"].map(ui.num),
    })
    st.dataframe(show, width="stretch", hide_index=True)

    # State the outcome rather than leaving a reader to rank nine rows by eye.
    # This is the project's actual finding and the plan committed to reporting
    # it either way: "if the simple threshold wins, that is a finding".
    if len(view):
        top = view.iloc[0]
        floor = view[view.baseline == "always_quiet"]["precision"]
        lift = (top["precision"] / floor.iloc[0]) if len(floor) and floor.iloc[0] else None
        noise = view[view.baseline.str.startswith("random_noise")]["lift"]
        if not lift:
            said = ""
        elif void:
            # The lift is quoted with its disqualifier attached, not silently.
            # A bare "22.1× the do-nothing floor" reads as the finding, and
            # pure noise scores 29.6× on the void frame.
            said = (f", {lift:.1f}× the do-nothing floor — a figure the caveat "
                    f"above disqualifies until the evaluation frame is "
                    f"recomputed, since pure noise reaches 29.6× on this frame")
        else:
            # On the corrected frame the lift stands; the null beside it is
            # what makes it a finding, so it is quoted in the same sentence.
            said = f", {lift:.1f}× the do-nothing floor" + (
                f" — random noise reaches {noise.min():.1f}–{noise.max():.1f}× "
                f"on this slice" if len(noise) else "")
        st.success(
            f"**{top['baseline']}** leads this slice at "
            f"**{ui.pct(top['precision'], 3)}** precision" + said
            + f", against a ceiling of {ui.pct(top['max_precision'], 2)}. "
            f"**If a simple baseline wins, it is shown winning** — that is the "
            f"finding, not something to hide.")

    _event_study()
    _item_lift(table, variant)

    ui.section("Calibration",
               "When a detector says 70%, is it right about 70% of the time? A "
               "detector can rank well and still be badly calibrated, which "
               "matters when a human decides what to act on.")
    st.dataframe(view[["baseline", "brier", "brier_skill_score", "ece"]],
                 width="stretch", hide_index=True)
    st.caption(
        "**Blank is the honest entry, not a gap.** CUSUM and the volume z-score "
        "emit scores that are **not probabilities** — the evaluation contract "
        "says so — and scoring them with Brier or ECE would invent a "
        "calibration they never claimed. A negative skill score means the "
        "probabilities are worse than always predicting the base rate: these "
        "models rank far better than they calibrate.")

    ui.section("Action distribution",
               "WAIT versus FLAG. A detector that cannot detect shows as "
               "`degenerate` — either it never flags, or its scores are all "
               "one value and so cannot rank.")
    st.dataframe(view[["baseline", "n_wait_hours", "n_flag_hours",
                       "pct_hours_flagged", "pct_windows_alerted", "degenerate"]],
                 width="stretch", hide_index=True)
    st.caption("`pct_hours_flagged` never exceeds ~2% even for a busy detector, "
               "because at most one FLAG is allowed per 48-hour window; "
               "`pct_windows_alerted` is the interpretable one.")

    with_news = data.comparison(_P8_WITH)
    without = data.comparison(_P8_WITHOUT)
    if not with_news.empty and not without.empty:
        _news_ablation(without, with_news, variant)


def _event_study() -> None:
    """The footprint itself: trading around t0, against a matched control.

    Every other figure on this screen scores a detector; this one scores
    nothing, so a reader can see what the detectors are hunting. Read off
    `src.eval.event_study` — train and validation events only, the test set
    untouched — and drawn as small multiples, one per slice, so scheduled and
    unscheduled are never on one line (rule 4). The control is the same stock
    ten sessions earlier at the same time of day: without it the opening
    bar's ordinary volume would pass for a footprint.
    """
    t = data.event_study()
    if t.empty:
        ui.note("No event-study table yet. Build it with "
                "`python -m src.eval.event_study`.")
        return
    t = t[t["bar"].notna()]
    ui.section(
        "The footprint itself — trading around t₀",
        "Every usable train and validation event lined up on its own t₀ (the "
        "news-adjusted clock), hourly bars counted in trading time: bar 0 "
        "contains or follows t₀, bar −1 is the last hour before it. The grey "
        "line is the same stock ten sessions earlier at the same time of day, "
        "which is what \"unusual\" has to beat. Test events are not read.")

    from plotly.subplots import make_subplots

    def facet(col: str, ylab: str, scale: float, fmt: str) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                            horizontal_spacing=0.04,
                            subplot_titles=("Unscheduled — the target",
                                            "Scheduled — results, dated ahead"))
        for c, (sl, colour, dash) in enumerate(
                (("unscheduled", ui.SERIES, "solid"),
                 ("scheduled", ui.SCHED, "dash")), start=1):
            ev = t[t["slice"] == sl].sort_values("bar")
            ctl = t[t["slice"] == f"{sl} control"].sort_values("bar")
            n_ev = int(ev["n_events"].max())
            fig.add_trace(go.Scatter(
                x=ctl["bar"], y=ctl[col] * scale, mode="lines",
                name="same stock, 10 sessions earlier", legendgroup="control",
                showlegend=c == 1,
                line=dict(color=ui.CONTROL, width=1.5, dash="dot"),
                hovertemplate=f"control: %{{y:{fmt}}}<extra></extra>"),
                row=1, col=c)
            fig.add_trace(go.Scatter(
                x=ev["bar"], y=ev[col] * scale, mode="lines",
                name=f"{sl} events ({n_ev:,})",
                line=dict(color=colour, width=2, dash=dash),
                hovertemplate=f"{sl}: %{{y:{fmt}}}<extra></extra>"),
                row=1, col=c)
            fig.add_vline(x=-0.5, line_dash="dash", line_color=ui.MARKER,
                          line_width=1, row=1, col=c)
        fig = ui.chart(fig, 300, ylab, legend=True)
        fig.update_xaxes(title=dict(text="trading-hour bars from t₀",
                                    font=dict(size=11)))
        # Legend under the plots: above them it collides with the panel titles.
        fig.update_layout(margin=dict(t=52, b=84, l=6, r=6),
                          legend=dict(orientation="h", yanchor="top",
                                      y=-0.36, x=0))
        return fig

    st.plotly_chart(facet("share_above_threshold",
                          "% of events with unusual volume", 100, ".1f"),
                    width="stretch")

    def at(sl: str, bar: int, col: str) -> float:
        r = t[(t["slice"] == sl) & (t["bar"] == bar)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    thr = float(t["threshold"].iloc[0])
    u, uc = (at("unscheduled", -1, "share_above_threshold"),
             at("unscheduled control", -1, "share_above_threshold"))
    sch, sc = (at("scheduled", -1, "share_above_threshold"),
               at("scheduled control", -1, "share_above_threshold"))
    u0 = at("unscheduled", 0, "share_above_threshold")
    n = {sl: int(t.loc[t["slice"] == sl, "n_events"].max())
         for sl in ("unscheduled", "unscheduled control", "scheduled",
                    "scheduled control")}
    st.caption(
        f"\"Unusual\" is the volume z-score detector's own threshold "
        f"({thr:.3f} sd). In the last hour before t₀, **{u:.1%} of unscheduled "
        f"events** clear it against {uc:.1%} for the same stocks ten sessions "
        f"earlier — about {u / uc:.1f}× — and {u0:.1%} in the bar that holds "
        f"or follows t₀. Scheduled events: {sch:.1%} against {sc:.1%}. The "
        f"saw-tooth every seven bars is the opening hour, present in the "
        f"control too; read the gap between the lines, not their shape. The "
        f"dashed line is t₀. Events: {n['unscheduled']:,} unscheduled and "
        f"{n['scheduled']:,} scheduled; controls: {n['unscheduled control']:,} "
        f"and {n['scheduled control']:,} — fewer, because a control window "
        f"holding another event of the same stock is dropped.")
    with st.expander("Median volume z-score and price move, same layout"):
        st.plotly_chart(facet("volume_z_median", "median volume z-score (sd)",
                              1, "+.2f"), width="stretch")
        st.plotly_chart(facet("abs_rel_ret_median_pct",
                              "median |1-bar move vs SPY| (%)", 1, ".2f"),
                        width="stretch")
        st.caption("The price move is absolute, because news moves prices both "
                   "ways and a signed average cancels to nothing. It barely "
                   "separates from the control before t₀ and jumps at it: the "
                   "footprint before an announcement is in volume, not price.")


def _item_lift(table: pd.DataFrame, variant: str) -> None:
    """Lift by 8-K item type, beside the random-noise range on the same type.

    The item codes are the answer key's free labels (frozen decision): this
    asks which kinds of event leave a footprint. Types with few events are
    left off rather than drawn — a lift on two events is noise with a decimal
    point — and the hit count is printed beside every dot.
    """
    items = table[(table["t0_variant"] == variant)
                  & table["slice"].str.startswith("item ")]
    if items.empty or "n_positive" not in items:
        return
    floor = int(data.config()["dashboard"]["item_min_events"])
    detectors = [d for d in ("cusum", "volume_zscore", "rl_policy[s43]",
                             "gradient_boosting") if d in set(items["baseline"])]
    if not detectors:
        return
    ui.section("Which kinds of event leave a footprint",
               f"Lift by 8-K item type, for item types with at least {floor} "
               f"events. The grey bar is the range pure random noise reaches "
               f"on the same type: a dot inside it is no finding.")
    who = st.selectbox("Detector", detectors, key="item_detector")
    d = items[items["baseline"] == who]
    d = d[d["n_positive"] >= floor].copy()
    if d.empty:
        st.caption("No item type has enough events in this table.")
        return
    noise = items[items["baseline"].str.startswith("random_noise")]
    rng = noise.groupby("slice")["lift"].agg(["min", "max"])
    d = d.join(rng, on="slice").sort_values("lift")
    d["hits"] = (d["precision"] * d["n_alerts"]).round().astype(int)
    codes = {str(c) for c in data.config()["items"]["scheduled"]}
    d["code"] = d["slice"].str.replace("item ", "", regex=False)
    d["kind"] = d["code"].map(lambda c: "scheduled" if c in codes else "unscheduled")
    names = {str(k): v for k, v in
             (data.config()["items"].get("names") or {}).items()}
    label = (d["code"] + " " + d["code"].map(lambda c: names.get(c, ""))
             + " · " + d["n_positive"].map(lambda n: f"{n:,} events"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=label, x=d["max"] - d["min"], base=d["min"], orientation="h",
        marker_color=ui.DIM, name="random noise range",
        hovertemplate="noise %{base:.2f}–%{x:.2f}<extra></extra>"))
    for kind, colour, sym in (("unscheduled", ui.SERIES, "circle"),
                              ("scheduled", ui.SCHED, "diamond")):
        k = d[d["kind"] == kind]
        if k.empty:
            continue
        fig.add_trace(go.Scatter(
            y=label[k.index], x=k["lift"], mode="markers+text",
            name=f"{who} — {kind}",
            marker=dict(color=colour, size=11, symbol=sym),
            text=k.apply(lambda r: f"{r['lift']:.1f}× · {r['hits']} hits", axis=1),
            textposition="middle right", textfont=dict(size=11),
            hovertemplate="%{x:.2f}× the floor<extra></extra>"))
    fig.add_vline(x=1, line_dash="dot", line_color=ui.CONTROL, line_width=1)
    fig = ui.chart(fig, 110 + 34 * len(d), "", legend=True)
    fig.update_layout(hovermode="closest", barmode="overlay",
                      margin=dict(t=24, b=80, l=6, r=90), legend=dict(orientation="h", yanchor="top", y=-0.3, x=0))
    # Room on the right for the label beside the largest dot.
    fig.update_xaxes(title=dict(text="lift vs the do-nothing floor (×)",
                                font=dict(size=11)),
                     range=[0, float(max(d["lift"].max(), d["max"].max())) * 1.3])
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "One event can carry several item codes, so the types overlap and do "
        "not add up to the slices above. Item 2.02 is results — scheduled, "
        "dated weeks ahead — and is marked apart. The hit count is how many "
        "budgeted alerts landed before an event of that type: a lift built on "
        "a dozen hits moves a lot with one more or one fewer. The dotted line "
        "is 1× — no better than raising nothing.")


def _rate_bars(r: pd.DataFrame, order: list, xlab: str) -> go.Figure:
    """Grouped bars of a live hit rate with its interval, split by kind."""
    fig = go.Figure()
    for kind, colour, pattern in (("unscheduled", ui.SERIES, ""),
                                  ("scheduled", ui.SCHED, "/")):
        k = r[r["kind"] == kind].set_index("group").reindex(order).dropna(
            subset=["n"])
        if k.empty:
            continue
        fig.add_trace(go.Bar(
            x=list(k.index), y=k["rate"] * 100, name=f"followed by {kind} 8-K",
            marker=dict(color=colour, pattern_shape=pattern),
            error_y=dict(type="data", symmetric=False,
                         array=(k["hi"] - k["rate"]) * 100,
                         arrayminus=(k["rate"] - k["lo"]) * 100,
                         color=ui.CONTROL, thickness=1.2, width=4),
            text=[f"{v:.1f}%" for v in k["rate"] * 100],
            textposition="outside", textfont=dict(size=11),
            customdata=k[["hits", "n", "lo", "hi"]].to_numpy(),
            hovertemplate=(f"{kind}: %{{customdata[0]:.0f}} of "
                           f"%{{customdata[1]:,.0f}} (%{{y:.1f}}%%, "
                           f"interval %{{customdata[2]:.1%%}}–"
                           f"%{{customdata[3]:.1%%}})<extra></extra>")))
    fig = ui.chart(fig, 290, "% of new flags", legend=True)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06,
                      hovermode="closest", margin=dict(t=24, b=80, l=6, r=6),
                      legend=dict(orientation="h", yanchor="top", y=-0.3, x=0))
    fig.update_xaxes(title=dict(text=xlab, font=dict(size=11)))
    return fig


def _live_charts(df: pd.DataFrame, hours: int) -> None:
    """Three live charts an analyst can act on, each split and each honest
    about its size: does a stronger alert pay off, is the rate holding up week
    to week, and how much warning did a hit give.

    All three count NEW flags with a closed, graded window, outside every
    data incident (`data.rates_by`). Raw rows would let one sustained anomaly
    carry a whole week — a single stock wrote 31 of one week's 58 graded
    alerts.
    """
    level = float(data.config()["dashboard"]["rate_interval"])
    ui.section(
        "How the live rate behaves",
        f"New flags only — one per stock per episode — with a closed "
        f"{hours}-hour window, outside every data incident. Each bar or point "
        f"carries a {level:.0%} interval (Wilson): the honest width of a rate "
        f"measured on this many alerts.")

    strength = df.apply(lambda r: ui.strength(r["score"], r["threshold"],
                                              r["detector"])[1], axis=1)
    by_band = data.rates_by(df, strength)
    if by_band.empty:
        ui.note("No graded new flags yet, so there is no rate to draw.")
        return
    order = ["Marginal", "Elevated", "Strong", "Extreme", "Policy flag"]
    order = [o for o in order if o in set(by_band["group"])]
    left, right = st.columns(2)
    with left:
        st.markdown("**Does a stronger alert pay off?**")
        st.plotly_chart(_rate_bars(by_band, order, "strength band"),
                        width="stretch")
        n = by_band[by_band["kind"] == "unscheduled"].set_index("group")["n"]
        st.caption(
            "Hit rate by the strength band shown in the queue. "
            + ", ".join(f"{g} {int(n[g]):,}" for g in order if g in n)
            + " new flags. Where the intervals overlap, the bands do not "
            "separate on this much data — a stronger alert is not yet shown to "
            "be a better one, and the queue's order is a reading order, not a "
            "probability.")

    week = pd.to_datetime(df["ts_utc"], unit="s", utc=True).dt.tz_localize(None) \
        .dt.to_period("W-SUN").dt.start_time
    by_week = data.rates_by(df, week)
    min_n = int(data.config()["dashboard"]["week_min_flags"])
    thin = sorted(by_week.loc[by_week["n"] < min_n, "group"].unique())
    by_week = by_week[by_week["n"] >= min_n]
    with right:
        st.markdown("**Is the rate holding up week to week?**")
        fig = go.Figure()
        for kind, colour, dash in (("unscheduled", ui.SERIES, "solid"),
                                   ("scheduled", ui.SCHED, "dash")):
            k = by_week[by_week["kind"] == kind].sort_values("group")
            fig.add_trace(go.Scatter(
                x=list(k["group"]) + list(k["group"])[::-1],
                y=list(k["hi"] * 100) + list(k["lo"] * 100)[::-1],
                mode="lines", fill="toself", fillcolor=ui.BAND,
                line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=k["group"], y=k["rate"] * 100, mode="lines+markers",
                name=f"followed by {kind} 8-K",
                line=dict(color=colour, width=2, dash=dash),
                marker=dict(size=8, color=colour),
                customdata=k[["hits", "n"]].to_numpy(),
                hovertemplate=(f"{kind}: %{{y:.1f}}%% "
                               f"(%{{customdata[0]:.0f}} of "
                               f"%{{customdata[1]:,.0f}})<extra></extra>")))
        fig = ui.chart(fig, 300, "% of new flags", legend=True)
        # The date ticks take two lines, so the legend sits lower here.
        fig.update_layout(margin=dict(t=24, b=96, l=6, r=6),
                          legend=dict(orientation="h", yanchor="top", y=-0.42, x=0))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(title=dict(text="week of the flagged bar (UTC, "
                                         "Monday start)", font=dict(size=11)))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Weekly rate with its interval shaded; a week's swing inside its "
            "band is noise, not drift. "
            + (f"Left off, with fewer than {min_n} graded new flags: weeks of "
               + ", ".join(w.strftime("%Y-%m-%d") for w in thin)
               + " — the monitor covered 400 stocks until 2026-09-07, and the "
               "newest week fills in as its windows close."
               if thin else ""))

    g = data.counted(df)
    if "episode_start" in g:
        g = g[g["episode_start"]]
    g = g[(g["outcome_state"] == "filed") & g["lead_trading_h"].notna()]
    if g.empty or "filed_unscheduled" not in g:
        return
    st.markdown("**How much warning did a hit give?**")
    fig = go.Figure()
    for kind, col, colour, pattern in (
            ("unscheduled", "filed_unscheduled", ui.SERIES, ""),
            ("scheduled", "filed_scheduled", ui.SCHED, "/")):
        k = g[g[col].fillna(0).astype(int) == 1]["lead_trading_h"]
        if k.empty:
            continue
        fig.add_trace(go.Histogram(
            x=k, xbins=dict(start=0, size=1), name=f"{kind} ({len(k):,}, "
            f"median {k.median():.1f} h)",
            marker=dict(color=colour, pattern_shape=pattern),
            opacity=0.85, hovertemplate=f"{kind}: %{{y}} flags<extra></extra>"))
    fig = ui.chart(fig, 260, "new flags", legend=True)
    fig.update_layout(barmode="group", bargap=0.15, hovermode="closest",
                      margin=dict(t=24, b=80, l=6, r=6), legend=dict(orientation="h", yanchor="top", y=-0.3, x=0))
    fig.update_xaxes(title=dict(text="lead time to the first 8-K that "
                                     "followed (trading hours)",
                                font=dict(size=11)))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Trading hours from the flagged bar to the first 8-K that followed, "
        f"for new flags that were hits. The {hours}-hour outcome window is "
        f"wall-clock, so no live lead can exceed about two sessions — the "
        f"longest here is {g['lead_trading_h'].max():.1f} trading hours — and "
        f"this is not comparable to the offline median lead, which is measured "
        f"over a 48-bar window.")


def _news_ablation(without: pd.DataFrame, with_news: pd.DataFrame,
                   variant: str) -> None:
    """Phase 8 on the corrected frame, with the noise floor beside it.

    The first version of this section read the 2026-09-08 pair, scored on the
    frame where pure noise reached 29.6x; on it news looked like +39% on
    unscheduled events. On the 2026-09-10 frame the same comparison is
    gradient boosting moving from about the noise floor to just above it, so
    the random-noise range is printed in the same table: a lift is only a
    finding if it clears that. Every figure is read off the files.
    """
    ui.section("Phase 8 — does the news channel help?",
               "Gradient boosting only: it is the sole baseline reading more "
               "than one column, so it is the only one that can carry this "
               "comparison. Validation, not test, on the corrected frame "
               "(2026-09-10).")
    slices = ["all", "scheduled", "unscheduled"]

    def pick(df, base):
        d = df[(df.t0_variant == variant) & (df.baseline == base)]
        return d.set_index("slice").reindex(slices)

    a, b = pick(without, "gradient_boosting"), pick(with_news, "gradient_boosting")
    noise = with_news[(with_news.t0_variant == variant)
                      & with_news.baseline.str.startswith("random_noise")]
    lo = noise.groupby("slice")["lift"].min().reindex(slices)
    hi = noise.groupby("slice")["lift"].max().reindex(slices)
    hits_a = (a.precision * a.n_alerts).round()
    hits_b = (b.precision * b.n_alerts).round()
    st.dataframe(pd.DataFrame({
        "slice": slices,
        "lift without news": a["lift"].map(lambda v: f"{v:.2f}×").to_numpy(),
        "lift with news": b["lift"].map(lambda v: f"{v:.2f}×").to_numpy(),
        "random noise": [f"{l:.2f}–{h:.2f}×" for l, h in zip(lo, hi)],
        "hits (without → with)": [f"{int(x)} → {int(y)}"
                                  for x, y in zip(hits_a, hits_b)],
        "lead without": a["median_lead_trading_h"].map(lambda v: f"{v:.1f} h").to_numpy(),
        "lead with": b["median_lead_trading_h"].map(lambda v: f"{v:.1f} h").to_numpy(),
    }), width="stretch", hide_index=True)

    clear = [sl for sl in slices if b.loc[sl, "lift"] > hi[sl]]
    # A rough two-count z for the change in hits; the arms share their
    # windows, so this is a guide to size, not a test.
    z = ((hits_b - hits_a) / (hits_a + hits_b).clip(lower=1) ** 0.5).round(1)
    sizes = ", ".join(f"{sl} z ≈ {z[sl]:.1f}" for sl in slices)
    st.caption(
        "**Read the lift against the random-noise column, not the change "
        "between the two.** Gradient boosting sits close to the noise floor "
        "with or without news, so a large percentage change is a handful of "
        "hits — the hits column shows how many. "
        + (f"With news it clears the noise range on: {', '.join(clear)}. "
           if clear else "With news it clears the noise range on no slice. ")
        + f"Clearing that range is not yet a significant gain; the change in "
        f"hits, in rough standard errors: {sizes} — two is the usual bar, and "
        f"several slices were looked at. Lead time "
        "falls with news — press coverage accumulates close to the event, so "
        "it buys confidence at the cost of warning. The 2026-09-08 version of "
        "this table (+39% on unscheduled events) was scored on the frame where "
        "pure noise reached 29.6× and is void.")


# --------------------------------------------------------------------------
# P9-05 — live monitor log
# --------------------------------------------------------------------------
def monitor_log() -> None:
    df = data.alerts_with_outcomes()
    if df.empty:
        ui.note("The live alert log is empty.")
        return

    hours = data.window_hours()
    split = data.split_hit_rates(df)
    cov = data.coverage()

    c = st.columns(4)
    ui.stat(c[0], "Alerts logged", ui.num(cov["logged"]),
            "append-only, hash-chained, committed to git after every run")
    ui.stat(c[1], "Windows closed", ui.num(cov["answerable"]),
            f"{hours} hours (wall-clock) elapsed, so an answer exists")
    ui.stat(c[2], "Graded here", ui.num(cov["graded"]),
            "closed AND with an outcome on record")
    ui.stat(c[3], "Not scored", ui.num(cov["unscored"]),
            "closed, but no outcome on record — an answer nobody looked up")

    # Rule 7, on the live screen as much as the offline one: the unscheduled
    # figure first and at least equal prominence, never a single pooled rate.
    ui.section("Hit rate — split scheduled vs unscheduled",
               f"Of the {cov['graded']:,} alerts with a graded outcome, how "
               f"many were followed by an 8-K within {hours} hours "
               f"(wall-clock). The denominator is the same for both halves: an "
               f"alert followed by nothing is a miss either way, and there is "
               f"no event to attach a slice to.")
    h = st.columns(3)
    ui.stat(h[0], "Unscheduled — the headline",
            ui.pct(split["unscheduled_rate"], 1)
            if split["unscheduled_rate"] is not None else "—",
            f"{split['unscheduled']} of {split['resolved']} graded alerts. "
            f"Genuinely unscheduled disclosures are the target of the whole "
            f"project.")
    ui.stat(h[1], "Scheduled (item "
            + ", ".join(data.config()["items"]["scheduled"]) + ")",
            ui.pct(split["scheduled_rate"], 1)
            if split["scheduled_rate"] is not None else "—",
            f"{split['scheduled']} of {split['resolved']} graded alerts. "
            f"Results announcements, whose dates are published weeks ahead — "
            f"the easy half.")
    ui.stat(h[2], "Pooled (not the headline)",
            ui.pct(split["pooled"], 1) if split["pooled"] is not None else "—",
            f"{split['filed']} of {split['resolved']} followed by any 8-K, "
            f"routine filings included. Shown for completeness and never "
            f"quoted alone: the easy half and the routine filings carry it.")

    ui.note(ui.honest_rate(split, hours))

    if cov.get("incident"):
        reasons = " ".join(f"**{i['label']}:** {i['reason']}"
                           for i in data.incidents())
        ui.note(f"**{cov['incident']:,} logged alerts sit on bars inside a "
                f"recorded data incident and are left out of every rate on "
                f"this screen.** They stay in the log, which is append-only "
                f"and hash-chained. {reasons}")
    quarantined = data.quarantine()
    if not quarantined.empty:
        ui.note(f"**The volume check has quarantined {len(quarantined):,} "
                f"stock-sessions** whose hourly volumes summed to more than "
                f"{data.config()['live']['volume_check']['max_hourly_to_daily']}x "
                f"the vendor's own daily total. Those bars were taken out "
                f"before scoring, so they raised no alerts. Record: "
                f"`live-log/quarantine.csv`.")

    if cov["unscored"]:
        ui.note(
            f"**Some closed windows have no grade on record, and the gap is "
            f"the denominator.** `live-log/alerts.csv` holds "
            f"**{cov['logged']:,}** alerts; **{cov['answerable']:,}** of them "
            f"have a window that closed long enough ago to be answerable, and "
            f"an outcome is on record — in `live-log/outcomes.csv` or this "
            f"database — for **{cov['graded']:,}** of those. The remaining **{cov['unscored']:,}** are shown as *not "
            f"scored*, not as pending: their windows closed, nobody has looked "
            f"the answer up here, and calling that \"still open\" would present "
            f"a rate measured on "
            f"{cov['graded'] / cov['answerable']:.1%} of the gradeable alerts "
            f"as if it were measured on all of them. Run "
            f"`python -m src.live.outcomes` against a database holding the "
            f"filings to close the gap.")

    # Measured off the log, per detector, rather than quoting one detector's
    # count from the day this note was written.
    clusters = data.alert_clusters(df)
    measured = "; ".join(
        f"**{name}** {ui.num(c['alerts'])} alerts span {ui.num(c['clusters'])} "
        f"clusters, {c['per_cluster']:.2f} per cluster"
        for name, c in sorted(clusters.items()))
    ui.note(
        f"**The live rate is not comparable to the offline precision figures.** "
        f"Offline, each event gets one {hours}-bar window and at most one FLAG "
        f"inside it, so precision counts one alert per event-window. Live, "
        f"`window_id` is `live:<ticker>:<ts>` — **every bar is its own window, "
        f"with no dedupe** — so one sustained anomaly writes several alerts and "
        f"several denominator entries. Measured on this log, collapsing each "
        f"ticker's alerts that sit within {hours} hours of each other: "
        f"{measured}. The two numbers answer different questions and neither "
        f"is adjusted to match the other.")

    ui.note(
        "**Append-only, and checkably so — with one gap named.** Every row "
        "carries a hash of itself and of the row before it, so an edit to a "
        "logged row, a row removed from the middle, and a row inserted out of "
        "order are all caught — verify with `python -m src.live.alertlog "
        "--verify`. What the chain does **not** prove is that the log is "
        "complete: nothing anchors the head, so truncating the newest rows or "
        "dropping a whole detector leaves a chain that still verifies. Two "
        "things cover that instead — git's own history of this file, and an "
        "export that refuses to write a log with fewer rows per detector than "
        "the one it would overwrite. The monitor re-scores a rolling 48-bar "
        "window each run and suppresses re-detections by natural key, so a "
        "repeated scan writes nothing.")

    _live_charts(df, hours)

    ui.section("The log", "Bar time and notice time are separate columns on "
                          "purpose: the monitor runs once a day after the "
                          "close, so an alert is noticed later than the hour it "
                          "describes.")
    show = pd.DataFrame({
        "bar (UTC)": df["ts_utc"].map(lambda t: ui.utc(t, False)),
        "noticed (UTC)": df["raised_utc"].map(lambda t: ui.utc(t, False)),
        "ticker": df["ticker"],
        "detector": df["detector"],
        "score": df["score"].map(lambda v: f"{v:.3f}"),
        "threshold": df["threshold"].map(lambda v: f"{v:.3f}"),
        # The log keeps incident rows (append-only) but says what they are,
        # in the same words as the triage queue.
        "outcome": df["outcome_state"].map(data.OUTCOME_WORDS).where(
            ~df["incident"], "Data incident — not counted")
        if "incident" in df else df["outcome_state"].map(data.OUTCOME_WORDS),
        # Rule 7 reaches the table too, not just the figures above it.
        "8-K type": data.outcome_slice(df),
        "lead (trading h)": df.get("lead_trading_h", pd.Series(index=df.index))
                              .map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
    })
    st.dataframe(show, width="stretch", hide_index=True, height=460)
    st.caption(
        "**`lead (trading h)` really is trading hours** — rule 3, the same unit "
        "every other lead-time figure in this project uses — while the "
        f"{hours}-hour outcome window above is wall-clock. They are different "
        "clocks on purpose: a company can file overnight or at a weekend, so "
        "\"did a filing follow within two days\" is a question about elapsed "
        "time, but \"how much warning was there\" is only meaningful in hours "
        "the market was open.")
