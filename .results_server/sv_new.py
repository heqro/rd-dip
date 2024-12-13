import numpy as np
import os
import json
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import itertools
from flask import Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
from dash.dash_table import DataTable
from table_utilities import *
import pandas as pd

# Path to the directory containing your CSV files
EXPERIMENTS_DIR = "../deep-image-prior/results/Brain1"
JSONS_DIR = "def_jsons"
DENOISED_DIR = "def_denoised"
SSIM_DIR = "def_ssim"

color_palette = px.colors.qualitative.Set1  # Other options: Set2, Set3, etc.


# Flask app
server = Flask(__name__)
auth = HTTPBasicAuth()

# Predefined users and passwords (in production, this should be stored securely, not in plaintext)
USERS = {
    "idc": generate_password_hash("123"),  # Replace with your username and password
}


# Verify the username and password
@auth.verify_password
def verify_password(username, password):
    if username in USERS and check_password_hash(USERS[username], password):
        return username
    return None


# Dash app
app = dash.Dash(__name__, server=server)


@server.before_request
@auth.login_required
def before_request():
    pass


df = pd.read_csv(f"{EXPERIMENTS_DIR}/summary.csv", sep=";")


# Function to style cells
def style_max(row):
    styles = []
    for col in df.columns[1:]:
        max_value = df[col].max()
        if row[col] == max_value:
            styles.append(
                {
                    "if": {"column_id": col, "row_index": row.name},
                    "backgroundColor": "#FFD700",
                    "color": "black",
                }
            )
    return styles


# Apply styles to all rows
cell_styles = []
for index, row in df.iterrows():
    cell_styles.extend(style_max(row))


# Layout
app.layout = html.Div(
    [
        html.H1("Experiment Data Comparison"),
        html.Label("Select Experiment Directory:"),
        dcc.Dropdown(
            id="dir-selector",
            options=[
                {"label": dir, "value": dir}
                for dir in os.listdir("../deep-image-prior/results")
                if os.path.isdir(os.path.join("../deep-image-prior/results", dir))
            ],
            value="Brain1",  # Default directory
            clearable=False,
        ),
        html.Label("Select JSON files:"),
        dcc.Dropdown(
            id="json-selector",
            options=[
                {"label": f, "value": f}
                for f in os.listdir(f"{EXPERIMENTS_DIR}/{JSONS_DIR}")
            ],
            value=[],  # Default no file selected
            multi=True,
        ),
        dcc.Graph(id="comparison-plot"),
        html.Div(
            id="image-display",
            style={"display": "flex", "flex-wrap": "wrap", "gap": "10px"},
        ),
        html.Div(
            [
                html.H3("Summary Table (ALL DATA)"),
                DataTable(
                    id="summary-table",
                    columns=[
                        {
                            "name": col,
                            "id": col,
                            "type": "numeric",
                            "format": {"specifier": ".2f"},
                        }
                        for col in df.keys()
                    ],
                    data=df.to_dict("records"),  # Fixed data
                    style_table={"overflowX": "auto"},
                    sort_action="native",
                    css=[
                        {
                            "selector": ".dash-cell div.dash-cell-value",
                            "rule": "user-select: text;",
                        }
                    ],
                ),
            ]
        ),
    ]
)


# Symlog transformation function


def add_data_point(
    fig,
    max_data_idx: int,
    max_data: float,
    color,
    iterations: list[int],
    row: int,
    col: int,
    hovertext: str = "",
):
    fig.add_trace(
        go.Scatter(
            x=[iterations[max_data_idx]],
            y=[max_data],
            mode="markers",
            marker=dict(color=color, size=10, line=dict(color="black", width=2)),
            showlegend=False,
            hovertext=hovertext,
        ),  # Do not show in legend
        row=row,
        col=col,
    )


@app.callback(
    [
        Output("json-selector", "options"),
        Output("json-selector", "value"),
        Output("summary-table", "data"),
    ],
    [Input("dir-selector", "value")],
)
def update_json_selector_and_table(selected_dir):
    global EXPERIMENTS_DIR, df
    EXPERIMENTS_DIR = f"../deep-image-prior/results/{selected_dir}"
    df = pd.read_csv(f"{EXPERIMENTS_DIR}/summary.csv", sep=";")
    json_files = [
        {"label": f, "value": f}
        for f in os.listdir(f"{EXPERIMENTS_DIR}/{JSONS_DIR}")
        if f.endswith(".json")
    ]
    return json_files, [], df.to_dict("records")  # Update JSON options and table data


# Callback for interactive updates
@app.callback(
    [Output("comparison-plot", "figure"), Output("image-display", "children")],
    [Input("json-selector", "value")],
)
def update_plot(selected_files):
    if not selected_files:
        return go.Figure(), []

    # Create a subplot with three columns and shared legend
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Loss",
            "Addends",
            "PSNR",
            "SSIM",
        ),
        shared_yaxes=False,  # Each plot can have independent y-axes
        horizontal_spacing=0.1,  # Adjust spacing between plots
    )

    # Loop through selected files and add traces
    color_cycle = itertools.cycle(color_palette)
    for file in selected_files:
        with open(f"{EXPERIMENTS_DIR}/{JSONS_DIR}/{file}", "r") as jsonfile:
            data = json.load(jsonfile)

        iterations = list(
            range(len(data["loss_log"]["overall_loss"]))
        )  # for instance, grab it from here
        color = next(color_cycle)

        stop_mask_idx = data["stopping_criteria_indices"]["mask_idx"]
        stop_idx = data["stopping_criteria_indices"]["entire_image_idx"]
        # Create a line plot for Loss, PSNR, and SSIM using Plotly Express

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=data["loss_log"]["overall_loss"],
                mode="lines",
                name=file,
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )
        if stop_idx is not None and stop_idx.is_integer():
            fig.add_hline(
                y=data["loss_log"]["overall_loss"][stop_idx],
                line=dict(color=color, width=2, dash="dash"),  # Line color and width
                row=1,  # For subplots
                col=1,
            )
        if stop_mask_idx is not None and stop_mask_idx.is_integer():
            fig.add_hline(
                y=data["loss_log"]["overall_loss"][stop_mask_idx],
                line=dict(color=color, width=2, dash="dot"),  # Line color and width
                row=1,  # For subplots
                col=1,
            )

        for addend in data["loss_log"]["addends"].keys():
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=data["loss_log"]["addends"][addend]["values"],
                    mode="lines",
                    name=f'{data["loss_log"]["addends"][addend]["coefficient"]}·{addend}',
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=data["psnr_mask_log"],
                mode="lines",
                name=file,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        max_data = max(data["psnr_mask_log"])
        max_data_idx = data["psnr_mask_log"].index(max_data)
        add_data_point(fig, max_data_idx, max_data, color, iterations, 2, 1, hovertext="Maximum")  # type: ignore
        if stop_mask_idx is not None and stop_mask_idx.is_integer():
            add_data_point(
                fig,
                stop_mask_idx,
                data["psnr_mask_log"][stop_mask_idx],
                color,
                iterations,
                row=2,
                col=1,
                hovertext="Stopping criterion (mask)",
            )
        if stop_idx is not None and stop_idx.is_integer():
            add_data_point(
                fig,
                stop_idx,
                data["psnr_mask_log"][stop_idx],
                color,
                iterations,
                row=2,
                col=1,
                hovertext="Stopping criterion (entire img)",
            )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=data["ssim_mask_log"],
                mode="lines",
                name=file,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        max_data = max(data["ssim_mask_log"])
        max_data_idx = data["ssim_mask_log"].index(max_data)
        add_data_point(fig, max_data_idx, max_data, color, iterations, 2, 2, hovertext="Maximum")  # type: ignore
        if stop_mask_idx is not None and stop_mask_idx.is_integer():
            add_data_point(
                fig,
                stop_mask_idx,
                data["ssim_mask_log"][stop_mask_idx],
                color,
                iterations,
                row=2,
                col=2,
                hovertext="Stopping criterion (mask)",
            )
        if stop_idx is not None and stop_idx.is_integer():
            add_data_point(
                fig,
                stop_idx,
                data["ssim_mask_log"][stop_idx],
                color,
                iterations,
                2,
                2,
                hovertext="Stopping criterion (entire img)",
            )

    # Update layout with a single legend
    fig.update_layout(
        title="Comparison of Metrics Across Selected Files",
        showlegend=True,  # Enable legend globally
        legend=dict(
            title="Files", orientation="h", x=0.5, xanchor="center", y=-0.2
        ),  # Legend at the bottom
        height=1000,  # Adjust height
        width=1200,  # Adjust width
    )

    # Add axis labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="PSNR", row=2, col=1)
    fig.update_yaxes(title_text="SSIM", row=2, col=2)

    images = []
    images.append(
        html.Div(
            [
                html.Img(
                    src=f"assets/{EXPERIMENTS_DIR.split('/')[-1]}/{EXPERIMENTS_DIR.split('/')[-1]}_gt.png",
                    style={"height": "256px", "border": "1px solid black"},
                ),
                html.P("Ground truth", style={"text-align": "center"}),
            ]
        )
    )
    images.append(
        html.Div(
            [
                html.Img(
                    src=f"assets/{EXPERIMENTS_DIR.split('/')[-1]}/{EXPERIMENTS_DIR.split('/')[-1]}_Std0.15.png",
                    style={"height": "256px", "border": "1px solid black"},
                ),
                html.P("Noisy", style={"text-align": "center"}),
            ]
        )
    )
    for file in selected_files:
        image_name = file.replace(".json", ".png")
        images.append(
            html.Div(
                [
                    html.Img(
                        src=f"assets/{EXPERIMENTS_DIR.split('/')[-1]}/{image_name}",
                        style={"height": "256px", "border": "1px solid black"},
                    ),
                    html.P(
                        f"{file[:18]}...",
                        style={"text-align": "center"},
                        title=file,
                    ),
                ]
            )
        )

    return fig, images


# Run server
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8052)
