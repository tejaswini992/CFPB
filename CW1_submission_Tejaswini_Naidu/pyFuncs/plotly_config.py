"""
Plotly configuration file settings
"""

import plotly.io as pio

def configure_plotly():
    # Create a Plotly template dictionary
    template = {
        "layout":
            {
            # Adjust the default plot size
            "width": 900,
            "height": 500,
            "title_font": {"size": 20},
            "legend": {
                "font": {"size": 8},
                "bordercolor": "gray",
                "bgcolor": "white"
            },
            # move legend to the right but keep it inside the plot
            "legend_x": 1.0,
            "legend_y": 1.0,
            "legend_xanchor": "right",
            "legend_yanchor": "top",
            "legend_traceorder": "normal",
            "legend_bordercolor": "black",
            "legend_borderwidth": 0.6,
            "xaxis": {
                "title_font": {"size": 16},
                "tickfont": {"size": 10},
                "showgrid": True,
                "gridcolor": "lightgray",
                "gridwidth": 0.5
            },
            "yaxis": {
                "title_font": {"size": 16},
                "tickfont": {"size": 12},
                "showgrid": False,
                "gridcolor": "lightgray",
                "gridwidth": 0.5
            },
        }
    }

    # Set the Plotly default template
    pio.templates["my_template"] = template
    pio.templates.default = "my_template"

# Call the configure_plotly function to set up the Plotly configuration
# configure_plotly()
