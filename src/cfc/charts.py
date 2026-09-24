"""Matplotlib / mplsoccer figures. Every function returns a PNG as bytes."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

INK = "#16213A"
MUTED = "#6B7488"
GRID = "#E3E7EF"
BLUE = "#1C4FB8"      # Chelsea blue
GOLD = "#C39A3A"      # crest gold
RED = "#C2413B"
PAPER = "#FFFFFF"
MOU = "#8C96AE"       # Mourinho period shade

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "legend.frameon": False,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return buf.getvalue()


def _shade_mourinho(ax, n_mou: int):
    ax.axvspan(0.5, n_mou + 0.5, color=MOU, alpha=0.12, lw=0)
    ax.text(n_mou / 2 + 0.5, 0.97, "Mourinho", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=MUTED)
    ax.text((n_mou + 38) / 2 + 0.5, 0.97, "Holland / Hiddink", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=MUTED)


def cumulative_points(c: pd.DataFrame, n_mou: int) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(1, len(c) + 1)
    ax.plot(x, c.points.cumsum(), color=BLUE, lw=2.4, label="Actual points")
    ax.plot(x, c.xpts.cumsum(), color=GOLD, lw=2.0, ls="--", label="Expected points (xPts)")
    _shade_mourinho(ax, n_mou)
    gap = c.xpts.iloc[:n_mou].sum() - c.points.iloc[:n_mou].sum()
    ax.annotate(f"-{gap:.1f} pts vs expected\nafter {n_mou} games", xy=(n_mou, c.points.iloc[:n_mou].sum()),
                xytext=(n_mou + 3, c.points.iloc[:n_mou].sum() - 6), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlim(0.5, len(c) + 0.5)
    ax.set_xlabel("Matchweek")
    ax.set_title("Chelsea 2015/16 — points vs expected points")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    return _png(fig)


def rolling_xg(c: pd.DataFrame, n_mou: int, window: int = 6) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(1, len(c) + 1)
    f = c.npxg.rolling(window, min_periods=3).mean()
    a = c.npxga.rolling(window, min_periods=3).mean()
    ax.plot(x, f, color=BLUE, lw=2.4, label="npxG for")
    ax.plot(x, a, color=RED, lw=2.4, label="npxG against")
    ax.fill_between(x, f, a, where=f >= a, color=BLUE, alpha=0.08, interpolate=True)
    ax.fill_between(x, f, a, where=f < a, color=RED, alpha=0.08, interpolate=True)
    _shade_mourinho(ax, n_mou)
    ax.set_xlim(0.5, len(c) + 0.5)
    ax.set_ylim(0, max(f.max(), a.max()) * 1.25)
    ax.set_xlabel("Matchweek")
    ax.set_title(f"Non-penalty xG, {window}-match rolling average")
    ax.legend(loc="lower right", ncol=2)
    return _png(fig)


def era_bars(table: pd.DataFrame) -> bytes:
    """table: index = metric label, columns = ['Mourinho', 'Hiddink', 'League avg']"""
    fig, axes = plt.subplots(1, len(table), figsize=(9, 2.8))
    colors = {"Mourinho": MOU, "Hiddink": BLUE, "League avg": GRID}
    for ax, (label, row) in zip(axes, table.iterrows()):
        vals = row.values.astype(float)
        bars = ax.bar(range(len(vals)), vals, color=[colors[c] for c in row.index], width=0.72)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(label, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, vals.max() * 1.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    fig.legend(handles, list(colors), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _png(fig)


def shotmap_pair(left: pd.DataFrame, right: pd.DataFrame, titles: tuple[str, str], suptitle: str) -> bytes:
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="#B9C1D1", pad_bottom=-8)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(9, 5))
    for ax, df, title in zip(axes, (left, right), titles):
        miss = df[~df.goal]
        goal = df[df.goal]
        pitch.scatter(miss.x, miss.y, s=miss.xg * 700 + 12, ax=ax, color=MOU, alpha=0.45, edgecolors="none")
        pitch.scatter(goal.x, goal.y, s=goal.xg * 700 + 12, ax=ax, color=RED, alpha=0.9,
                      edgecolors=INK, linewidth=0.6)
        ax.set_title(f"{title}\n{len(df)} shots · {df.xg.sum():.1f} xG · {int(df.goal.sum())} goals",
                     fontsize=10, color=INK, loc="center")
    fig.suptitle(suptitle, x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    return _png(fig)


def player_finishing(df: pd.DataFrame) -> bytes:
    """df: index player, columns ['Mourinho', 'Hiddink'] = goals - npxG"""
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.2))
    y = np.arange(len(df))
    h = 0.38
    ax.barh(y + h / 2, df["Mourinho"], height=h, color=MOU, label="Under Mourinho")
    ax.barh(y - h / 2, df["Hiddink"], height=h, color=BLUE, label="Under Hiddink / Holland")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, df.index)
    ax.invert_yaxis()
    ax.set_xlabel("Goals minus non-penalty xG (positive = finished above expectation)")
    ax.set_title("Finishing: who went cold, who recovered")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def percentile_profile(pcts: pd.Series, title: str, style: set[str] = frozenset()) -> bytes:
    """pcts: index = label, values 0..1. Labels in `style` are neutral style traits (gold),
    the rest are strengths (blue, high) or weaknesses (red, low)."""
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pcts) + 1.0))
    y = np.arange(len(pcts))
    colors = [GOLD if k in style else BLUE if v >= 0.66 else (RED if v <= 0.34 else MOU)
              for k, v in pcts.items()]
    ax.barh(y, pcts.values * 100, color=colors, height=0.62)
    for yi, v in zip(y, pcts.values):
        ax.text(v * 100 + 1.5, yi, f"{v * 100:.0f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(50, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(y, pcts.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("League percentile (20 teams)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def late_fade(team_curve: pd.Series, league_curve: pd.Series, team: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(team_curve))
    ax.bar(x - 0.2, league_curve.values, width=0.4, color=GRID, label="League average")
    ax.bar(x + 0.2, team_curve.values, width=0.4, color=RED, label=team)
    ax.set_xticks(x, team_curve.index)
    ax.set_ylabel("Share of xG conceded")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"When {team} concede chances (level-score situations only)")
    ax.set_ylim(0, max(team_curve.max(), league_curve.max()) * 1.3)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def match_xg_race(poss: pd.DataFrame, home: str, away: str, title: str) -> bytes:
    """poss: output of metrics.possession_xg for one match (rebounds already combined)."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for team, color in ((home, BLUE if home == "Chelsea" else RED), (away, BLUE if away == "Chelsea" else RED)):
        s = poss[poss.team == team].sort_values(["period", "minute"])
        mins = np.concatenate([[0], s.minute.values, [95]])
        cum = np.concatenate([[0], s.pxg.cumsum().values, [s.pxg.sum()]])
        ax.step(mins, cum, where="post", color=color, lw=2.2, label=f"{team} ({s.pxg.sum():.2f} xG)")
        g = s[s.goal]
        ax.scatter(g.minute, s.pxg.cumsum()[s.goal].values, color=color, edgecolors=INK, zorder=3, s=46)
    ax.set_xlim(0, 95)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xG")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _png(fig)


def match_bars(c: pd.DataFrame, team: str) -> bytes:
    """c: team's matches (date, opponent, venue, goals, goals_against, xg, xga)."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(c))
    ax.bar(x - 0.2, c.xg, width=0.4, color=BLUE, label="xG for")
    ax.bar(x + 0.2, c.xga, width=0.4, color=RED, label="xG against")
    for xi, (_, r) in zip(x, c.iterrows()):
        res = "W" if r.goals > r.goals_against else "D" if r.goals == r.goals_against else "L"
        ax.text(xi, max(r.xg, r.xga) + 0.12, f"{res} {r.goals}-{r.goals_against}", ha="center", fontsize=9,
                color=INK, fontweight="bold")
    labels = [f"{'vs' if v == 'H' else '@'} {o}\n{d:%d %b}" for o, v, d in zip(c.opponent, c.venue, c.date)]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, max(c.xg.max(), c.xga.max()) + 0.7)
    ax.set_ylabel("xG")
    ax.set_title(f"{team} 2026/27 — match by match")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def points_vs_xpts(t: pd.DataFrame, highlight: tuple[str, ...]) -> bytes:
    """t: index team, columns points, xpts. Sorted by points."""
    t = t.sort_values("points")
    fig, ax = plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(t))
    for yi, (team, r) in zip(y, t.iterrows()):
        hl = team in highlight
        ax.plot([r.xpts, r.points], [yi, yi], color=INK if hl else "#B9C1D1", lw=2 if hl else 1.2, zorder=1)
        ax.scatter(r.xpts, yi, color=GOLD, s=46 if hl else 26, zorder=2, edgecolors=INK if hl else "none")
        ax.scatter(r.points, yi, color=BLUE, s=46 if hl else 26, zorder=3, edgecolors=INK if hl else "none")
    ax.set_yticks(y, t.index)
    for lab in ax.get_yticklabels():
        if lab.get_text() in highlight:
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.scatter([], [], color=BLUE, label="Actual points")
    ax.scatter([], [], color=GOLD, label="Expected points (Understat xPts)")
    ax.legend(loc="lower right")
    ax.set_xlabel("Points after 5 matches")
    ax.set_title("Premier League 2026/27 — points vs expected points")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def translation_factors(tbl: pd.DataFrame, title: str) -> bytes:
    """tbl: index league name, columns per metric: '<m>' factor, '<m>_lo', '<m>_hi' (90% CI)."""
    metrics = [c for c in tbl.columns if not c.endswith(("_lo", "_hi"))]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(tbl) + 1.4))
    y = np.arange(len(tbl))
    h = 0.8 / len(metrics)
    colors = [BLUE, GOLD, MOU]
    for k, m in enumerate(metrics):
        yy = y - 0.4 + h * (k + 0.5)
        ax.barh(yy, tbl[m], height=h * 0.9, color=colors[k], label=m)
        ax.errorbar(tbl[m], yy, xerr=[tbl[m] - tbl[f"{m}_lo"], tbl[f"{m}_hi"] - tbl[m]],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
        for yi, v in zip(yy, tbl[m]):
            ax.text(0.505, yi, f"×{v:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
    ax.axvline(1, color=INK, lw=0.8, ls=":")
    ax.set_yticks(y, tbl.index)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel("Per-90 multiplier after moving to the Premier League (1.0 = no change)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def fingerprint_dumbbell(now: pd.Series, before: pd.Series, labels: dict, now_label: str, before_label: str,
                         title: str) -> bytes:
    """League-percentile fingerprint: this season (now) vs the manager's previous team (before)."""
    keys = list(labels)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(keys) + 1.3))
    y = np.arange(len(keys))
    for yi, k in zip(y, keys):
        ax.plot([before[k] * 100, now[k] * 100], [yi, yi], color="#B9C1D1", lw=2.2, zorder=1)
    ax.scatter(before[keys] * 100, y, color=GOLD, s=70, zorder=2, label=before_label)
    ax.scatter(now[keys] * 100, y, color=BLUE, s=70, zorder=3, label=now_label)
    ax.set_yticks(y, [labels[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.axvline(50, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("League percentile within its own league-season")
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.grid(axis="y", visible=False)
    return _png(fig)
