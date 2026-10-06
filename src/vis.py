import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data"

x_format = alt.X("time:Q", title="Zeit (s)").scale(domain=(0, 20))
y_format = (
    alt.Y("position:Q", title="Position (m)")
    .scale(
        domain=(-3.0, 3.0),
    )
    .axis(
        values=np.arange(-3.0, 3.5, 0.5).tolist(),
    )
)

lf_data = pd.read_parquet(data_path / "leapfrog.parquet")
lf_data["model"] = "Leapfrog"

rk4_data = pd.read_parquet(data_path / "classical_runge_kutta.parquet")
rk4_data["model"] = "Runge Kutta (4. Ordnung)"

sim_data = pd.concat([lf_data, rk4_data])
sim_chart = (
    alt.Chart(sim_data)
    .mark_point(
        size=10,
        filled=True,
    )
    .encode(
        x=x_format,
        y=y_format,
        color=alt.Color("model:N").legend(title="Model"),
    )
)

num_data = alt.sequence(0, 20, 0.1, as_="time")
num_graph = (
    alt.Chart(num_data)
    .transform_calculate(position="cos(datum.time)")
    .mark_line(
        color="black",
        strokeWidth=1.5,
        strokeDash=[8, 4],
        opacity=0.7,
    )
    .encode(
        x=x_format,
        y=y_format,
    )
)

(sim_chart + num_graph).properties(
    title="Vergleich Runge-Kutta und Leapfrog",
    width=700,
    height=300,
).save(data_path / "chart.svg")
