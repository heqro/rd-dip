import numpy as np
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import itertools
from flask import Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash

# Path to the directory containing your CSV files
CSV_DIRECTORY = "../deep-image-prior/results/Brain1/csvs"
IMAGE_DIRECTORY = "../deep-image-prior/results/Brain1/denoised_images"

# Get the list of CSV files
csv_files = [f for f in os.listdir(CSV_DIRECTORY) if f.endswith(".csv")]
# Use Plotly's built-in color palette (e.g., 'Set1' from the qualitative palettes)
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


# Layout
app.layout = html.Div(
    [
        html.H1("Experiment Data Comparison"),
        html.Label("Select CSV files:"),
        dcc.Dropdown(
            id="csv-selector",
            options=[{"label": f, "value": f} for f in csv_files],
            value=[],  # Default no file selected
            multi=True,
        ),
        dcc.Graph(id="comparison-plot"),
        html.Div(
            id="image-display",
            style={"display": "flex", "flex-wrap": "wrap", "gap": "10px"},
        ),
    ]
)


# Symlog transformation function
def symlog_transform(data, linthresh=1.0, linscale=1.0):
    return np.sign(data) * np.log1p(np.abs(data) / linthresh) * linscale


# Callback for interactive updates
@app.callback(
    [Output("comparison-plot", "figure"), Output("image-display", "children")],
    [Input("csv-selector", "value")],
)
def update_plot(selected_files):
    if not selected_files:
        return go.Figure(), []

    # Create a subplot with three columns and shared legend
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Loss",
            "PSNR",
            "SSIM",
        ),
        shared_yaxes=False,  # Each plot can have independent y-axes
        horizontal_spacing=0.1,  # Adjust spacing between plots
    )

    # Loop through selected files and add traces
    color_cycle = itertools.cycle(color_palette)
    for file in selected_files:
        data = pd.read_csv(os.path.join(CSV_DIRECTORY, file))
        iterations = list(range(len(data)))
        color = next(color_cycle)

        # Create a line plot for Loss, PSNR, and SSIM using Plotly Express

        if "Loss" in data:
            transformed_loss = symlog_transform(data["Loss"])
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=transformed_loss,
                    mode="lines",
                    name=file,
                    line=dict(color=color),
                ),
                row=1,
                col=1,
            )
        if "PSNR" in data:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=data["PSNR"],
                    mode="lines",
                    name=file,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            max_psnr_idx = data["PSNR"].idxmax()  # Get index of maximum PSNR
            max_psnr = data["PSNR"].max()  # Get maximum PSNR value
            fig.add_trace(
                go.Scatter(
                    x=[iterations[max_psnr_idx]],
                    y=[max_psnr],
                    mode="markers",
                    marker=dict(
                        color=color, size=10, line=dict(color="black", width=2)
                    ),
                    showlegend=False,
                ),  # Do not show in legend
                row=1,
                col=2,
            )
        if "SSIM" in data:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=data["SSIM"],
                    mode="lines",
                    name=file,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
            # Add a marker at the maximum SSIM value
            max_ssim_idx = data["SSIM"].idxmax()  # Get index of maximum SSIM
            max_ssim = data["SSIM"].max()  # Get maximum SSIM value
            fig.add_trace(
                go.Scatter(
                    x=[iterations[max_ssim_idx]],
                    y=[max_ssim],
                    mode="markers",
                    marker=dict(
                        color=color, size=10, line=dict(color="black", width=2)
                    ),
                    showlegend=False,
                ),  # Do not show in legend
                row=1,
                col=3,
            )
        # fig.add_trace(ssim_trace.data[0], row=1, col=3)

    # Update layout with a single legend
    fig.update_layout(
        title="Comparison of Metrics Across Selected Files",
        showlegend=True,  # Enable legend globally
        legend=dict(
            title="Files", orientation="h", x=0.5, xanchor="center", y=-0.2
        ),  # Legend at the bottom
        height=500,  # Adjust height
        width=1200,  # Adjust width
    )

    # Add axis labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=1, col=3)
    fig.update_yaxes(title_text="Loss", row=1, col=1)  # Symlog scale for Loss
    fig.update_yaxes(title_text="PSNR", row=1, col=2)
    fig.update_yaxes(title_text="SSIM", row=1, col=3)

    images = []
    images.append(
        html.Div(
            [
                html.Img(
                    src=f"assets/Brain1/Brain1.png",
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
                    src=f"assets/Brain1/Contaminated_Brain1_0.15.png",
                    style={"height": "256px", "border": "1px solid black"},
                ),
                html.P("Noisy", style={"text-align": "center"}),
            ]
        )
    )
    for file in selected_files:
        image_name = file.replace(".csv", ".png")
        image_path = os.path.join(IMAGE_DIRECTORY, image_name)
        if os.path.exists(image_path):
            images.append(
                html.Div(
                    [
                        html.Img(
                            src=f"assets/Brain1/{image_name}",
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
    app.run_server(debug=True, host="0.0.0.0", port=8050)
