"""Plotly figures for the dashboard.

Colors come from one token set per theme. Triage uses the fixed status palette (the same in light
and dark). The two runs being compared take categorical slots 1 and 2: blue is always the policy
you picked and orange the one it is compared with, and every line is labelled with its policy.
Resource states use one blue ramp, darker for "in use". Each pair was checked with the dataviz
validator against the Streamlit background of its theme.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from .. import params as P

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
TRIAGE_COLOR = {"RED": "#d03b3b", "YELLOW": "#fab219", "GREEN": "#0ca30c"}
INCIDENT_LEGEND = {"RED": "RED patient waiting", "YELLOW": "YELLOW, no RED, waiting", "GREEN": "Only GREEN waiting",
                   "cleared": "Everyone has left"}
STATES = ("in use", "free", "unavailable")

TOKENS = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "zone_outer": "#f6f8fa", "zone_inner": "#edf1f5", "zone_core": "#e4e9ef", "zone_line": "#d5dce4", "link": "#9aa6b2",
        "hospital": "#0f2233", "unit": "#52606d", "selected": "#2a78d6", "compare": "#eb6834",
        "in use": "#1c5cab", "free": "#86b6ef", "unavailable": "#c3c2b7", "track": "#e1e0d9",
        "meter": "#2a78d6", "warning": "#fab219", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#0e1117", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "zone_outer": "#141922", "zone_inner": "#1a202a", "zone_core": "#212834", "zone_line": "#2c3440", "link": "#5c6876",
        "hospital": "#e6eaef", "unit": "#aab4c0", "selected": "#3987e5", "compare": "#d95926",
        "in use": "#5598e7", "free": "#184f95", "unavailable": "#52514e", "track": "#2c2c2a",
        "meter": "#3987e5", "warning": "#fab219", "critical": "#d03b3b",
    },
}


def on_fill(hex_color: str) -> str:
    """White or near-black text for a label set inside a fill, whichever reads better."""
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lum = sum(w * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for w, c in zip((0.2126, 0.7152, 0.0722), rgb))
    return "#0b0b0b" if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "#ffffff"


def _layout(tok: dict, height: int, **extra) -> dict:
    return dict(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT, size=12, color=tok["ink2"]),
                hoverlabel=dict(bgcolor=tok["surface"], bordercolor=tok["axis"], font=dict(family=FONT, size=12, color=tok["ink"])),
                **extra)


# ---------------------------------------------------------------------- map
def _placed(units: pd.DataFrame, incidents: pd.DataFrame, hospitals: pd.DataFrame) -> pd.DataFrame:
    """Where to draw each unit. Units on the same spot sit side by side; a unit on top of an incident
    moves just below its circle and one at a hospital just above its square, so neither hides the other."""
    spots = [(r.x, r.y, -1.15) for r in incidents.itertuples()] + [(r.x, r.y, 1.15) for r in hospitals.itertuples()]
    out = units.copy()
    out["px"], out["py"] = out.x.astype(float), out.y.astype(float)
    out["label_at"] = "bottom center"
    for _, group in out.groupby([out.x.round(2), out.y.round(2)]):
        x, y = group.x.iloc[0], group.y.iloc[0]
        dy = next((d for sx, sy, d in spots if abs(sx - x) < 0.05 and abs(sy - y) < 0.05), 0.0)
        for i, idx in enumerate(group.index):
            out.loc[idx, "px"] = x + (i - (len(group) - 1) / 2) * 1.0
            out.loc[idx, "py"] = y + dy
            if dy > 0:                     # above a hospital: label on top, clear of the square
                out.loc[idx, "label_at"] = "top center"
    return out


def map_figure(layers: dict[str, pd.DataFrame], city_km: float, tok: dict, height: int = 640) -> go.Figure:
    incidents, units, hospitals = layers["incidents"], layers["units"], layers["hospitals"]
    cx, cy = P.CITY_CENTER
    fig = go.Figure()
    shapes = [dict(type="rect", x0=0, y0=0, x1=city_km, y1=city_km, fillcolor=tok["zone_outer"], line=dict(color=tok["zone_line"], width=1), layer="below")]
    for radius, fill in ((P.ZONE_RADII_KM[1], tok["zone_inner"]), (P.ZONE_RADII_KM[0], tok["zone_core"])):
        shapes.append(dict(type="circle", x0=cx - radius, y0=cy - radius, x1=cx + radius, y1=cy + radius, fillcolor=fill,
                           line=dict(color=tok["zone_line"], width=1), layer="below"))
    notes = [dict(x=cx, y=cy + r - 0.5, text=name, showarrow=False, font=dict(size=11, color=tok["muted"]))
             for name, r in (("core", P.ZONE_RADII_KM[0]), ("inner zone", P.ZONE_RADII_KM[1]), ("outer zone", city_km / 2))]
    shapes.append(dict(type="line", x0=city_km - 5, x1=city_km, y0=-1.0, y1=-1.0, line=dict(color=tok["ink2"], width=2)))
    notes.append(dict(x=city_km - 5.3, y=-1.0, xanchor="right", text="5 km", showarrow=False, font=dict(size=11, color=tok["ink2"])))

    placed = _placed(units, incidents, hospitals) if not units.empty else units
    heading = placed.dropna(subset=["tx"]) if not placed.empty else placed
    xs, ys = [], []
    for r in heading.itertuples():
        xs += [r.px, r.tx, None]
        ys += [r.py, r.ty, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=tok["link"], width=1.5), hoverinfo="skip", showlegend=False)

    for r in hospitals.itertuples():           # ER-bay meter under each hospital: blue, then amber from 80%, red when full or ramped
        share = r.bays_used / r.bays_total if r.bays_total else 1.0
        color = tok["critical"] if share >= 1 or r.door else tok["warning"] if share >= 0.8 else tok["meter"]
        x0, y0 = r.x - 1.0, r.y - 1.15
        shapes.append(dict(type="rect", x0=x0, x1=x0 + 2.0, y0=y0, y1=y0 + 0.32, fillcolor=tok["track"], line=dict(width=0)))
        shapes.append(dict(type="rect", x0=x0, x1=x0 + 2.0 * min(share, 1.0), y0=y0, y1=y0 + 0.32, fillcolor=color, line=dict(width=0)))
        door = f", {r.door} at door" if r.door else ""
        notes.append(dict(x=r.x, y=y0 - 0.15, yanchor="top", text=f"{r.bays_used}/{r.bays_total}{door}",
                          showarrow=False, font=dict(size=11, color=tok["ink2"])))
    fig.add_scatter(
        x=hospitals.x, y=hospitals.y, mode="markers+text", name="Hospital (ER bays used / open)", text=hospitals.id,
        textfont=dict(size=10, color=on_fill(tok["hospital"])),
        marker=dict(symbol="square", size=24, color=tok["hospital"], line=dict(color=tok["surface"], width=2)),
        hovertext=[f"<b>{r.id} {r.name}</b><br>ER bays {r.bays_used} of {r.bays_total} in use, surge {r.surge}"
                   f"<br>{r.door} ambulances at the door, {r.corridor} patients in the corridor<br>ICU beds free {r.icu_free} of {r.icu_total}"
                   f"<br>doctor slots free {r.doctors_free}, nurse slots free {r.nurses_free}, staff lent in {r.lent_in}"
                   for r in hospitals.itertuples()], hoverinfo="text")

    for state in ("RED", "YELLOW", "GREEN", "cleared"):
        part = incidents[incidents.state == state] if not incidents.empty else incidents
        filled = state != "cleared"
        fig.add_scatter(
            x=part.x if len(part) else [None], y=part.y if len(part) else [None], mode="markers+text",
            name=INCIDENT_LEGEND[state], text=part.number if len(part) else None,
            textfont=dict(size=10, color=on_fill(TRIAGE_COLOR[state]) if filled else tok["muted"]),
            marker=dict(size=[12 + 4.5 * math.sqrt(n) for n in part.waiting] if len(part) else 12,
                        color=TRIAGE_COLOR[state] if filled else tok["surface"],
                        line=dict(color=tok["surface"] if filled else tok["muted"], width=2)),
            hovertext=[f"<b>{r.id}</b> {r.type}<br>{r.waiting} of {r.victims} victims waiting: RED {r.RED}, YELLOW {r.YELLOW}, GREEN {r.GREEN}"
                       f"<br>{'unit on scene' if r.reached else f'no unit yet; caller reported {r.reported} hurt'}"
                       f"{'' if r.access else ', scene not safe to enter yet'}<br>work left: {r.needs}<br>teams: {r.teams}"
                       for r in part.itertuples()] if len(part) else None, hoverinfo="text")

    for name, mask, symbol in (("Ambulance, ALS", lambda f: (f.kind == "ambulance") & (f.type == "ALS"), "triangle-up"),
                               ("Ambulance, BLS", lambda f: (f.kind == "ambulance") & (f.type == "BLS"), "triangle-up-open"),
                               ("Responder team", lambda f: f.kind == "team", "diamond")):
        part = placed[mask(placed)] if not placed.empty else placed
        fig.add_scatter(
            x=part.px if len(part) else [None], y=part.py if len(part) else [None], mode="markers+text", name=name,
            text=part.id if len(part) else None, textposition=list(part.label_at) if len(part) else "bottom center",
            textfont=dict(size=10, color=tok["ink2"]),
            marker=dict(symbol=symbol, size=12, color=tok["unit"],
                        opacity=[0.45 if s == "unavailable" else 1.0 for s in part.state] if len(part) else 1.0,
                        line=dict(color=tok["unit"] if symbol.endswith("open") else tok["surface"], width=2 if symbol.endswith("open") else 1.5)),
            hovertext=[f"<b>{r.id}</b> {r.type}<br>{r.doing}" + (f"<br>{r.onboard} patients on board" if r.onboard else "")
                       + (f"<br>task: {r.task}" if r.task else "") for r in part.itertuples()] if len(part) else None,
            hoverinfo="text")

    pad = 0.6
    fig.update_layout(**_layout(tok, height, margin=dict(l=4, r=4, t=4, b=4), shapes=shapes, annotations=notes,
                                showlegend=True, legend=dict(orientation="h", x=0, y=0, yanchor="top", itemsizing="constant",
                                                             font=dict(size=11, color=tok["ink2"]))))
    fig.update_xaxes(range=[-pad, city_km + pad], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-1.7, city_km + pad], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------- resources
def resource_figure(status: pd.DataFrame, tok: dict) -> go.Figure:
    """One bar per resource, split into in use, free and unavailable, as a share of the total."""
    order = list(dict.fromkeys(status.resource))
    totals = status.groupby("resource")["count"].sum().reindex(order)
    fig = go.Figure()
    for state in STATES:
        part = status[status.state == state].set_index("resource")["count"].reindex(order).fillna(0)
        fig.add_bar(
            y=order, x=part / totals.where(totals > 0, 1), orientation="h", name=state, text=part.astype(int),
            texttemplate="%{text}", textposition="inside", insidetextanchor="middle",
            textfont=dict(color=on_fill(tok[state]), size=12), customdata=totals.astype(int),
            marker=dict(color=tok[state], line=dict(color=tok["surface"], width=2)),
            hovertemplate=f"%{{y}}<br><b>%{{text}}</b> {state} of %{{customdata}}<extra></extra>")
    fig.update_layout(**_layout(tok, 90 + 40 * len(order), margin=dict(l=8, r=8, t=8, b=8), barmode="stack", bargap=0.42,
                                barcornerradius=4, uniformtext=dict(minsize=10, mode="hide"),
                                legend=dict(orientation="h", x=0, y=0, yanchor="top", traceorder="normal")))
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(autorange="reversed", fixedrange=True, ticks="", showline=False, automargin=True,
                     tickfont=dict(color=tok["ink"]), ticklabelstandoff=8)
    return fig


# ---------------------------------------------------------------------- performance over time
def series_figure(lines: list[tuple[str, pd.Series, str]], t: int, tok: dict, height: int = 240) -> go.Figure:
    """The same running total for two runs, with the current minute marked. Each line is labelled at its end
    unless the ends are too close to tell apart, in which case the legend and the hover carry the names."""
    fig = go.Figure()
    horizon = max(int(s.index.max()) for _, s, _ in lines)
    top = max(float(s.max()) for _, s, _ in lines) or 1.0
    ends = [float(s.iloc[-1]) for _, s, _ in lines]
    label_ends = len(lines) < 2 or abs(ends[0] - ends[1]) > 0.08 * top
    for (name, s, color), end in zip(lines, ends):
        fig.add_scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(color=color, width=2, shape="hv"),
                        hovertemplate=f"<b>%{{y:.0f}}</b> {name}<extra></extra>")
        now = float(s.loc[min(t, horizon)])
        fig.add_scatter(x=[t], y=[now], mode="markers", marker=dict(color=color, size=9, line=dict(color=tok["surface"], width=2)),
                        showlegend=False, hoverinfo="skip")
        if label_ends:
            fig.add_annotation(x=horizon, y=end, text=f"{name} {end:.0f}", xanchor="left", xshift=6, showarrow=False,
                               font=dict(size=11, color=tok["ink2"]))
    fig.add_vline(x=t, line=dict(color=tok["axis"], width=1))
    fig.update_layout(**_layout(tok, height, margin=dict(l=8, r=60, t=34, b=8), hovermode="x unified", showlegend=True,
                                legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11))))
    fig.update_xaxes(range=[0, horizon], showgrid=False, showline=True, linecolor=tok["axis"], ticks="outside", tickcolor=tok["axis"],
                     title=dict(text="minute", font=dict(size=11, color=tok["muted"])), fixedrange=True, automargin=True,
                     showspikes=True, spikemode="across", spikethickness=1, spikecolor=tok["axis"], spikedash="solid")
    fig.update_yaxes(range=[0, top * 1.12], gridcolor=tok["grid"], zeroline=False, showline=False, fixedrange=True, automargin=True)
    return fig
