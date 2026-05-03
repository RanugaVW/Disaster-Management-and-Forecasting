import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Get absolute paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Create output directory
output_dir = os.path.join(project_root, 'Visualization')
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
# Read the dataset
data_path = os.path.join(project_root, 'data', 'master_feature_matrix.csv')
df = pd.read_csv(data_path)

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# Extract Year-Month
df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

print("Aggregating data...")
# Aggregate by division and year_month
agg_df = df.groupby(['division', 'year_month']).agg({
    'rain_sum': 'mean',
    'temperature_2m_mean': 'mean',
    'soil_moisture_7_to_28cm': 'mean',
    'soil_moisture_28_to_100cm': 'mean',
    'soil_moisture_100_to_255cm': 'mean'
}).reset_index()

# Calculate an average soil moisture across the layers for a simpler plot
agg_df['soil_moisture_avg'] = agg_df[['soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm']].mean(axis=1)

# Get unique divisions
divisions = sorted(agg_df['division'].unique())

print("Generating interactive plot...")
# Create a figure with subplots
fig = make_subplots(rows=3, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Monthly Average Rain Sum (mm)", 
                                    "Monthly Mean Temperature (°C)", 
                                    "Monthly Mean Soil Moisture"))

# Add traces for every division, but only make the first one visible
for i, division in enumerate(divisions):
    div_data = agg_df[agg_df['division'] == division]
    visible = (i == 0)
    
    # Trace 0: Rain Sum
    fig.add_trace(go.Bar(
        x=div_data['year_month'],
        y=div_data['rain_sum'],
        name='Rain Sum',
        marker_color='blue',
        visible=visible
    ), row=1, col=1)
    
    # Trace 1: Temperature
    fig.add_trace(go.Scatter(
        x=div_data['year_month'],
        y=div_data['temperature_2m_mean'],
        mode='lines',
        name='Temperature',
        line=dict(color='red'),
        visible=visible
    ), row=2, col=1)

    # Trace 2: Soil Moisture
    fig.add_trace(go.Scatter(
        x=div_data['year_month'],
        y=div_data['soil_moisture_avg'],
        mode='lines',
        name='Soil Moisture',
        line=dict(color='green'),
        visible=visible
    ), row=3, col=1)

# Create dropdown buttons
buttons = []
for i, division in enumerate(divisions):
    # For each division, we need 3 traces visible, others invisible
    visibility = [False] * (len(divisions) * 3)
    visibility[i*3] = True      # Rain
    visibility[i*3 + 1] = True  # Temperature
    visibility[i*3 + 2] = True  # Soil Moisture
    
    button = dict(
        label=division,
        method="update",
        args=[{"visible": visibility},
              {"title": f"Monthly Climate Metrics for {division}"}]
    )
    buttons.append(button)

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=buttons,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top",
            showactive=True,
        )
    ],
    title=dict(text=f"Monthly Climate Metrics for {divisions[0]}", x=0.5, xanchor='center', y=0.95),
    height=800,
    showlegend=False
)

fig.update_xaxes(title_text="Date", row=3, col=1)

# Save the plot
output_path = os.path.join(output_dir, 'monthly_climate_metrics.html')
fig.write_html(output_path)

print(f"Visualization saved to {output_path}")
