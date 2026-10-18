import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data"

x_format = alt.X("time:Q", title="Zeit (s)").scale(domain=(0, 20))
y_format = (
    alt.Y("position:Q", title="Position (m)")
    .scale(
        domain=(-1.5, 1.5),
    )
    .axis(
        values=np.arange(-1.5, 2.5, 0.5).tolist(),
    )
)

model_domain = ["Exakte Lösung", "Leapfrog", "Runge-Kutta (4. Ordnung)", "Heun"]
model_range = ["black", "red", "orange", "green"]
model_scale = alt.Scale(domain=model_domain, range=model_range)

lf_data = pd.read_parquet(data_path / "leapfrog.parquet")
lf_data["model"] = "Leapfrog"
lf_elided_data = lf_data.iloc[::2]

rk4_data = pd.read_parquet(data_path / "classical_runge_kutta.parquet")
rk4_data["model"] = "Runge-Kutta (4. Ordnung)"

heun_data = pd.read_parquet(data_path / "heun.parquet")
heun_data["model"] = "Heun"
heun_elided_data = heun_data.iloc[1::2]

sim_data = pd.concat([lf_elided_data, rk4_data, heun_elided_data])
sim_chart = (
    alt.Chart(sim_data)
    .mark_line(
        size=14,
        opacity=1,
        strokeWidth=1.5,
    )
    .encode(
        x=x_format,
        y=y_format,
        color=alt.Color("model:N", scale=model_scale).legend(
            title="Verfahren", orient="bottom"
        ),
    )
)

num_data = pd.DataFrame({"time": np.arange(0.0, 20.1, 0.1)})
num_data["model"] = "Exakte Lösung"

num_graph = (
    alt.Chart(num_data)
    .transform_calculate(position="cos(datum.time)")
    .mark_line(
        color="black",
        strokeWidth=1,
        strokeDash=[4, 4],
    )
    .encode(
        x=x_format,
        y=y_format,
        color=alt.Color("model:N", scale=model_scale).legend(
            title="Verfahren", orient="bottom"
        ),
    )
)

(sim_chart + num_graph).properties(
    width=400,
    height=125,
).save(data_path / "chart.svg")
