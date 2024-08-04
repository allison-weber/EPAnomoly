from datetime import date
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, ctx, callback, State
import dash_bootstrap_components as dbc
from glob import glob
from pathlib import Path
from viz_utils import (
    _get_site_data,
    _get_data_for_variable
)
from viz_models import (
    detect_anomaly_dbscan,
    create_clusters_dbscan
)
from bspline_hourly_outliers import (
    find_site_outliers_hourly_spline_mse,
    detect_anomalies_bsplines_hourly
)
from bspline_daily_outliers import (
    find_site_outliers_daily_spline_error,
    detect_anomalies_bsplines_daily
)

# Get all sites for daily data
site_names = glob("../data/daily/sites/*")
variable_names = glob("../data/daily/*")

# Remove "sites" from the list of variable names
variable_names = sorted([Path(x).name for x in variable_names if "sites" not in x])
VARIABLE_NAMES_TO_ENGLISH = {
    # Clean English translations of the variable names for display
    "AQI": "Air Quality Index",
    "CO": "Carbon Monoxide",
    "NO2": "Nitrogen Dioxide",
    "Ozone": "Ozone",
    "PM10": "Particulate Matter <10 microns",
    "PM2.5 FRM": "Particulate Matter <2.5 microns (FRM)",
    "PM2.5 non-FRM": "Particulate Matter <2.5 microns (non-FRM)",
    "PMc": "Particulate Matter Coarse Fraction",
    "SO2": "Sulfur Dioxide",
    "HAPs": "Hazardous Air Pollutants",
    "VOCs": "Volatile Organic Compounds",
    "NONOxNOy": "Nitrous oxides concentration (ppb)",
    "Temperature": "Temperature",
    "Pressure": "Barometric Pressure",
    "RH_DP": "Relative Humidity / Dew Point",
    "Wind": "Wind",
}

HOURLY_SPLINE_CRITICAL_VALUE = 15
DAILY_SPLINE_CRITICAL_VALUE = 6

##############################
###   CHARTING FUNCTIONS   ###
##############################


# Callback to update the store with the selected site
@callback(
    Output("clicked-site", "data"),
    [Input("map", "clickData")]
)
def update_store(clickData: dict):
    '''
    Updates the store with the selected site.

    Args:
    - clickData (dict): The click data from the map.
    '''
    if clickData is None:
        return None
    site_id = clickData["points"][0]["hovertext"]
    return site_id


@callback(
    [Output("map", "figure"),
     Output('current_variable', 'data'),
     Output("stored-data", "data")],
    [Input("variable-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("clicked-site", "data"),
     Input("model-dropdown", "value")],
    State("stored-data", "data")
)
def chart_map_sites(variable: str, start_date: str = None, end_date: str = None,
                    clicked_site: dict = None, model: str = None, old_data: dict=None ) -> go.Figure:
    '''
    Maps all of the sites reporting the given variable.

    Args:
    - variable (str): The variable to map.
    - start_date (str): The start date for the data, in format "YYYY-MM-DD".
    - end_date (str): The end date for the data, in format "YYYY-MM-DD".
    - model (str): Anomaly detection option (currently only DBSCAN)
    '''
    # deselect site if change variable
    if ctx.triggered_id == 'variable-dropdown':
        clicked_site = None
    # Only get new data if input other than site selected changes
    if ctx.triggered_id != 'clicked-site' or not old_data:
        print(f"Map data update: {ctx.triggered_id=} {variable=}")
        data = _get_data_for_variable(variable)

        if variable == 'AQI' or not variable:
            data = data.drop_nulls(subset=['AQI'])
        else:
            data = data.drop_nulls(subset=['Arithmetic Mean'])

        if start_date is None:
            start_date = data["Date Local"].min()
        if end_date is None:
            end_date = data["Date Local"].max()



        assert start_date <= end_date, "Start date must be before end date."

        # Filter for the date range
        data = data.filter(pl.col("Date Local") >= start_date).filter(pl.col("Date Local") <= end_date)

        # DBSCAN takes too long, for now this allows us to immediately draw the map,
        #     + interact with points while only updating map if DBSCAN selected from dropdown
        if model == 'DBSCAN': # only run if new variable selected (not new point)
            outliers = detect_anomaly_dbscan(data, variable)
        elif model == "B-Spline MSE (hourly)":
            outliers = detect_anomalies_bsplines_hourly(
                data, variable, critical_value=HOURLY_SPLINE_CRITICAL_VALUE,
                start_date=start_date, end_date=end_date
            )
        elif model == "B-Spline MSE (daily)":
            outliers = detect_anomalies_bsplines_daily(
                data, variable, critical_value=DAILY_SPLINE_CRITICAL_VALUE,
                start_date=start_date, end_date=end_date
            )

        # Now, for each site, get the first date and last date in the resulting data
        site_dates = data.group_by("site_id").agg(pl.min("Date Local").alias("first_date"), pl.max("Date Local").alias("last_date"))

        # Now, filter the data to only include the first and last dates for each site
        # Will be useful for later -- for showing how the data has changed over time.
        data = data.join(site_dates, on="site_id", how="inner").filter(pl.col("Date Local") == pl.col("last_date"))
        site_locations = pl.read_parquet("../data/aqs_sites.parquet").drop("__index_level_0__")\
            .with_columns((pl.col("State Code").cast(pl.Utf8) + pl.col("County Code").cast(pl.Utf8) + pl.col("Site Number").cast(pl.Utf8)).alias("site_id"))\
            .select("site_id", "Latitude", "Longitude", "State Name", "County Name", "City Name")

        merged_data = data.join(site_locations, on="site_id", how="inner")
        if model in ["DBSCAN", "B-Spline MSE (hourly)", "B-Spline MSE (daily)"]:
            merged_data = merged_data.join(outliers, on="site_id", how="left")
        old_data = merged_data.to_dict(as_series=False)
    else:
        print(f"Map redrawn with same data: {ctx.triggered_id=}")
        merged_data = pl.from_dict(old_data)

    # print(merged_data)

    if model == 'DBSCAN':
        chart = px.scatter_mapbox(
            merged_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="site_id",
            hover_data=["State Name", "County Name", "City Name"],
            color="DBSCAN anomaly detected?",
            color_discrete_map={
                'Yes': "red",
                'No': "blue"},
            zoom=3.45,
            title= "<b>Sensors Across The United States - Click one to view details</b>"
        )
    elif model == "B-Spline MSE (hourly)":
        chart = px.scatter_mapbox(
            merged_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="site_id",
            hover_data=["State Name", "County Name", "City Name"],
            color="Hourly spline anomaly detected?",
            color_discrete_map={
                'Yes': "red",
                'No': "blue",
                "Insufficient data": "gray"},
            zoom=3.45,
            title= "<b>Sensors Across The United States - Click one to view details</b>"
        )
    elif model == "B-Spline MSE (daily)":
        chart = px.scatter_mapbox(
            merged_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="site_id",
            hover_data=["State Name", "County Name", "City Name"],
            color="Daily spline anomaly detected?",
            color_discrete_map={
                'Yes': "red",
                'No': "blue",
                "Insufficient data": "gray"},
            zoom=3.45,
            title= "<b>Sensors Across The United States - Click one to view details</b>"
        )
    else:
        chart = px.scatter_mapbox(
            merged_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="site_id",
            hover_data=["State Name", "County Name", "City Name"],
            zoom=3.45,
            title= "<b>Sensors Across The United States - Click one to view details</b>"
        )

    # Change size and color of selected site
    if clicked_site is not None:
        clicked_site_id = clicked_site
        # Update the color of the clicked point
        if "outlier" not in merged_data.columns:
            chart.data[0].marker.color = ['green' if site_id == clicked_site_id else '#636efa' for site_id in merged_data['site_id']]
            chart.data[0].marker.size = [20 if site_id == clicked_site_id else 8 for site_id in merged_data['site_id']]
        else:
            for i in range(len(chart.data)):
                chart.data[i].marker.size = [20 if site_id == clicked_site_id else 8 for site_id in chart.data[i].hovertext]
    # Set to 50 if we want to exclude Alaska?
    # There aren't many sensors there, so maybe it maeks sense to not use such a limited number of sensors to predict over such a large landmass
    chart.update_layout(mapbox_style="open-street-map",
                        mapbox_bounds={"west": -145, "east": -45, "south": -20, "north": 90},
                        margin = dict(l = 15, r = 15, t = 30, b = 5)
    )
    chart.update_mapboxes(center = {"lat": 37.0902, "lon": -95.7129},)

    return chart, {"current_variable": variable}, old_data


@callback(
    Output("selected-variables-container", component_property="children"),
    [Input("clicked-site", "data"),
     Input("variable-dropdown", "value")]
)
def create_variable_checklist(clicked_site: dict, selected_variable: str):
    '''
    Creates a checklist of all available variables for the selected site.

    Args:
    - clicked_site (dict): The click data from the map.
    - selected_variable (str): The currently selected variable.
    '''
    # When dashboard first visualizes, nothing has been selected yet from the variable list, so no charts below the map should be shown
    if clicked_site is None:
        return html.Div([
        dcc.Checklist(
            options = [],
            value=[],
            id="selected-variables-list",
            style={"display": "none"}
        )
        ])
    
    site_id = clicked_site
    site_data = _get_site_data(site_id)

    variables = site_data.keys()
    # Move the selected variable to the front of the list
    variables = [selected_variable] + [var for var in variables if var != selected_variable]

    checklist = dcc.Checklist(
        options = [{"label": VARIABLE_NAMES_TO_ENGLISH[var], "value": var} for var in variables],
        value=[selected_variable],
        id="selected-variables-list",
        className = "Variables_Chart",
        inputStyle = {'margin-right': '10px'}
    )

    return html.Div([
        html.Label("Available variables at the site - Choose to display the trend in charts below:", className = "Decisions"
                   ),
        checklist,
        dbc.Row(dbc.Col(html.Hr(className = "Divider"), width={'size':10, 'offset':1}))
    ],)

@callback(
    Output("time-series-container", component_property="children"),
    [Input("clicked-site", "data"),
     Input("selected-variables-list", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("model-dropdown", "value"),
     Input('current_variable', 'data')]
)
def chart_site_data(clicked_site: dict, selected_variables: list,
                    start_date: str = None, end_date: str = None,
                    model: str = None, selected_variable: dict = None):
    '''
    Creates a dash component containing 1 line chart for each of the variables in the data_dict.

    If there is no data for that component, chart returns blank with words NO DATA AVAILABLE.

    Date Picker Range is also integrated to narrow down the available data range.

    Args:
    - site_id (str): The site ID to chart.
    - selected_variables (list): The variables to chart. Chosen by the user, defined by the component in the create_variable_checklist callback.
    - start_date (str): The start date for the data, in format "YYYY-MM-DD".
    - end_date (str): The end date for the data, in format "YYYY-MM-DD".
    - model (str): name of anomaly detection model
    '''
    if clicked_site is None:
        return []
    site_id = clicked_site
    site_data = _get_site_data(site_id)
    charts = []
    if selected_variable is None:
        current_variable = 'AQI'
    else:
        current_variable = selected_variable['current_variable']

    for variable in selected_variables:

        # if change variable and selected point not have that variable-skip
        # better to deselect point but this at least removes the error
        if variable not in site_data:
            continue

        data = site_data[variable]

        if variable not in selected_variables:  # Only show charts for the variables that are selected
            continue

        # Filter charts for the selected date range
        if start_date is None:
            start_date = data["Date Local"].min()
        if end_date is None:
            end_date = data["Date Local"].max()

        data = data.filter(pl.col("Date Local") >= start_date).filter(pl.col("Date Local") <= end_date)
        
        # Set render mode to svg to avoid Chrome deleting the map -- see https://community.plotly.com/t/too-many-active-webgl-contexts/16379/3
        if data.shape[0] == 0:  # Check if the data is empty
                    # Create a blank figure with a text annotation (happy to change this if there's better phrasing)
                    chart = go.Figure()
                    chart.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="NO DATA AVAILABLE",
                        showarrow=False,
                        font=dict(size=20)
                    )
                    chart.update_layout(xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0,1]),
                                        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0,1]))
        else:
            chart = px.line(data, x="Date Local", y=variable,
                            title=f"<b>{variable}</b> at site <b>{site_id}</b>",
                            render_mode="svg", template="plotly_white")
            chart.update_layout(title=f"<b>{variable}</b> at site <b>{site_id}</b>",
                                font={'size': 13.5},
                                )
            chart.update_yaxes(gridcolor='rgba(0,0,0,0.125)')
            chart.update_xaxes(gridcolor='rgba(0,0,0,0.125)')
            chart.update_traces(connectgaps=True)
            if model == 'DBSCAN' and variable == current_variable:
                df = data.drop_nulls(subset=variable)
                clustering = create_clusters_dbscan(df, variable)
                df = df.with_columns(pl.Series(name='outliers', values=clustering.labels_))
                df = df.with_columns(pl.lit(10).alias('marker_size'))
                df = df.filter(pl.col("outliers") == -1)
                chart.add_traces(px.scatter(df, x="Date Local", y=variable,
                                            color_discrete_sequence=["red"],
                                            size='marker_size').data)
            elif model == "B-Spline MSE (hourly)" and variable != "AQI":
                _, df = find_site_outliers_hourly_spline_mse(data, variable, critical_value=HOURLY_SPLINE_CRITICAL_VALUE)
                df = df.filter(pl.col("outlier") == 1)
                df = df.with_columns(pl.lit(10).alias('marker_size'))
                chart.add_traces(px.scatter(df, x="Date Local", y=variable,
                                            color_discrete_sequence=["red"],
                                            size='marker_size').data)
            elif model == "B-Spline MSE (daily)":
                _, df = find_site_outliers_daily_spline_error(data, variable, critical_value=DAILY_SPLINE_CRITICAL_VALUE)
                df = df.filter(pl.col("outlier") == 1)
                df = df.with_columns(pl.lit(10).alias('marker_size'))
                chart.add_traces(px.scatter(df, x="Date Local", y=variable,
                                            color_discrete_sequence=["red"],
                                            size='marker_size').data)
                
        chart_dcc = dcc.Graph(figure=chart)
        charts.append(chart_dcc)
        row_separator = dbc.Row(dbc.Col(html.Hr(className = "Divider"), width={'size':10, 'offset':1 }))
        charts.append(row_separator)
    
    # Create a div containing each of the charts
    return charts


###############################
###   DASH APP DEFINITION   ###
###############################

#Create the app
#app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.SANDSTONE])
app.title = "EPAnomaly - Team 103"

app.layout = html.Div([
    dcc.Store(id='clicked-site', storage_type='memory'), # Stores the clicked site in memory
    dcc.Store(id='current_variable', storage_type='memory'),
    dcc.Store(id='stored-data', storage_type='memory'),
    html.H1("EPAnomaly: Spatiotemporal Visualization of Anomalies in Environmental Factors", className = "Project_Name"
            ),
    html.H5("Team 103 (HDDA Survivors): Marco Segura Camacho, Raaed Mahmud Khan, \
            Calista Randazzo, Abhinuv Uppal, Ben Yu, Allison Weber", className = "Team_Names"
            ),
        
    html.Div([
        dbc.Row(dbc.Col(html.Hr(className = "Divider"), width={'size':10, 'offset':1 }))
            ]),

    # Instructional Text
    html.Div([
        html.P(
            "Welcome to the EPAnomaly Dashboard for exploring patterns and anomalies in environmental data across the United States from the EPA's network of sensors in the Air Quality System.",
            style={'padding': '10px', 'font-size': '18px', 'font-weight': 'bold'},
            className="Instructions"
        ),
        html.P(
            "This dashboard provides the tools to explore environmental anomalies and trends in air quality data. \
            These data can be used to inform policy decisions, public health initiatives, and further research into the causes and effects of air pollution.",
            className = "Instructions-Text"
        ),
        html.P(
            "An environmental anomaly (outlier) is defined as a measurement that is significantly different from normal measurements. These can correspond \
            to events such as heat waves and cold fronts for temperature anomalies, wildfires for air quality and particulate matter anomalies, or even measurement system malfunctions.\
            We provide several methods for automatic anomaly detection (see below). This tool allows you to see where and when anomalies occurred over custom time ranges. \
            Data currently run from 2018 through late 2023. You can click on a site to see the details of the measurements at that location as well as when detected anomalies occurred.",
            className = "Instructions-Text"
        ),
        
        html.P(
            "In order to get started:",
            className = "Instructions-Text"
        ),
        html.Ul([
            html.Li("Select an environmental measurement and (optional) an anomaly detection method from the dropdown menus. Anomaly detection may take anywhere from a couple seconds (splines) to up to 2 minutes (DBSCAN)."),
            html.Li("(Optional) Adjust the date range to focus on specific periods."),
            html.Li("Click any site on the map to view details regarding that location. Sites with detected anomalies are highlighted on the map and anomalies per measurement at that location are highlighted before for your chosen timeframe."),
            html.Li("Below the map, selected site details will appear, allowing you to analyze variable trends over time and identify anomalies."),
        ],
        style={'padding': '40px'}, 
        className="Instructions-list"),
        html.P(
            "Pollutant Measurement Definitions:",
            style={'padding': '10px', 'font-size': '18px', 'font-weight': 'bold'},
            className="Measurements-Header"
        ),
        html.Ul([
            html.Li(["AQI: Air Quality Index – Indicates pollution levels in the air. Exposure to higher AQI values are associated with a greater risk of respiratory and cardiovascular diseases.",
                html.Ul([
                    html.Li("Good: 0-50"),
                    html.Li("Moderate: 51-100"),
                    html.Li("Unhealthy for Sensitive Groups: 101-150"),
                    html.Li("Unhealthy: 151-200"),
                    html.Li("Very Unhealthy: 201-300"),
                    html.Li("Hazardous: 301+")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["CO: Carbon Monoxide – Colorless, odorless gas that interferes with the blood's ability to carry oxygen, leading to potentially fatal health effects if inhaled in large amounts.",
                html.Ul([
                    html.Li("Safe levels: Under 9 ppm over an 8-hour period")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["NO2: Nitrogen Dioxide – This gas contributes to smog and air pollution, irritating the lungs and increasing susceptibility to respiratory infections.",
                html.Ul([
                    html.Li("Safe levels: Below 100 ppb over a 1-hour period")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["Ozone – A reactive gas that can cause coughing, throat irritation, and exacerbate asthma as well as other respiratory conditions.",
                html.Ul([
                    html.Li("Safe levels: Below 0.060 ppm during an 8-hour period")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["PM10 and PM2.5: Particulate Matter – These fine particles can penetrate deep into the lungs and into the bloodstream, causing cardiovascular, cerebrovascular, and respiratory impacts.",
                html.Ul([
                    html.Li("PM2.5: Safe under 12 µg/m³ annually"),
                    html.Li("PM10: Safe under 50 µg/m³ daily")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["SO2: Sulfur Dioxide – This gas can lead to throat and eye irritation and exacerbate asthma and other respiratory conditions.",
                html.Ul([
                    html.Li("Safe levels: Under 75 ppb over a 1-hour period")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["HAPs: Hazardous Air Pollutants – These pollutants include various chemicals that are known to cause cancer, birth defects, and other serious health problems.",
                html.Ul([
                    html.Li("Safe levels: Exposure should be minimized as there are no safe levels")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["VOCs: Volatile Organic Compounds – These chemicals can release organic compounds while being used, and some can cause respiratory irritation and damage to the body's tissues.",
                html.Ul([
                    html.Li("Safe levels: Exposure should be minimized, with indoor levels typically higher than outdoor")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["NONOxNOy: Nitrogen Oxides – These compounds are involved in the formation of ozone and particulate matter, both of which can impair lung function and aggravate respiratory diseases.",
                html.Ul([
                    html.Li("Safe levels: Below 100 ppb")
                ], style={'marginLeft': '50px', 'fontSize': '12px'})
            ]),
            html.Li(["Temperature, Pressure, Relative Humidity / Dew Point, Wind: While generally not directly harmful, these meteorological measurements can influence the concentration and dispersal of air pollutants.",
            ]),
        ], 
        style={'padding': '40px'},
        className="Environmental-Measurements"),
        html.P(
            "Anomaly Detection Model Definitions:",
            style={'padding': '10px', 'font-size': '18px', 'font-weight': 'bold'},
            className="Models-Header"
        ),
        html.Ul([
            html.Li("DBSCAN: Density-Based Spatial Clustering of Applications with Noise – This model groups sites with similar pollutant levels and identifies points that don't group well with others as anomalies. This model excels at identifying days with unusual values compared to the global trend in a non-parametric fashion."),
            html.Li("B-Spline MSE (hourly): Uses a cubic spline to approximate the hourly frequency data for each site and identifies days with large average errors as anomalies. This model excels at identifying days with unusual hourly pollutant levels, but can fail at identifying relative to the global trend."),
            html.Li("B-Spline MSE (daily): Uses a cubic spline to approximate the daily frequency data and identifies sites with large errors as anomalies. This model excels at identifying days with unusual daily pollutant levels when compared to the global trend, but may present false positives attributable to their parametric nature."),
        ], 
        style={'padding': '40px'},
        className="Models-List")
    ]),
        
        html.Div([
            dbc.Row(
                dbc.Col(
                    html.Label("Available Measurements:", className = "Decisions"
                            )
                        )
                       ),
            dbc.Row(
                dbc.Col(
                dcc.Dropdown(
                id="variable-dropdown",
                options=[
                            {'label': v, 'value': k}
                            for k, v in VARIABLE_NAMES_TO_ENGLISH.items()
                        ],
                value=list(VARIABLE_NAMES_TO_ENGLISH.keys())[0],
                placeholder = "Select a variable",
                className = "Dropdown_Menu"
                            )
                )
            )
        ]),
    
        # Set date range to ensure that all available data is represented by default
        html.Div([
            html.Label("Date Range:", className = "Decisions"
                       ),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=date(2018, 1, 1),
                end_date=date(2024, 1, 1),
                display_format='YYYY-MM-DD',
                className = "Datepicker_Menu"
            )
        ]),

        html.Div([
            html.Label("Model Selection: ", className = "Decisions"
                       ),
            dcc.Dropdown(
                id="model-dropdown",
                options=['DBSCAN', "B-Spline MSE (hourly)", "B-Spline MSE (daily)", 'None'],
                placeholder="Select an anomaly detection method",
                className = "Dropdown_Menu"
            )
        ]),
        
       html.Div([
           dbc.Row(dbc.Col(html.Hr(className = "Divider"), width={'size':10, 'offset':1}))
               ]),
        html.Div([
            dcc.Graph(
                id="map",
                className = "Graph_Map",
            )], id="map-container"),
    
        html.Div([
            dbc.Row(dbc.Col(html.Hr(className = "Divider"), width={'size':10, 'offset':1}))
                ]),
            
        # Once a value from the dropdown is selected, a chart is shown. Select-boxes also shown.
        html.Div([
            html.Label("Select a site on the map to view its data."
                       ),
            dcc.Checklist(
                options=[{"label": "Placeholder", "value": "Placeholder"}],
                value=["Placeholder"],
                id="selected-variables-list",
                style={"display": "block"
                       }
            ),
        ], id="selected-variables-container"),
        html.Div([
        ], id="time-series-container"),
    ])


if __name__ == "__main__":
    app.run_server(debug=False)
