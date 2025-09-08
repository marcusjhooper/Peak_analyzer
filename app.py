import dash
from dash import html, dcc, callback, Input, Output, State
import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import logging
import re
import pandas as pd
from pathlib import Path
from dash.dash_table import DataTable
from pyfaidx import Fasta
import crested
from tensorflow import keras
import matplotlib
import zipfile
import tempfile
from itertools import repeat
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive plotting


###for use on molgen shiny (rsconnect):



# Import our modules
from utils import parse_coordinates, get_sequence, format_sequence_with_line_numbers, highlight_motif, calculate_gc_content, reverse_complement
from data_loading import (
    load_cell_type_colors, 
    scan_for_bigwigs, 
    load_differential_peaks, 
    load_model_settings,
    scan_for_peak_tables,
    load_gene_annotations,
    find_genes_in_region
)
from plotting import create_contribution_scores_plot, create_plot, contribution_scores_df, plot_heatmap, create_contribution_scores_dataframe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app

# Set the data directory

#DATA_DIR = '/home/mh/app/Denali/data'  #local
DATA_DIR = '/allen/programs/celltypes/workgroups/rnaseqanalysis/mouse_multiome/app/data'







# isilon
app = dash.Dash(__name__, title='DeNAli',
                suppress_callback_exceptions=True,
                assets_folder='assets')  # Change back to 'assets' folder for CSS



# Register CRESTED custom objects
custom_objects = {
    'CosineMSELogLoss': crested.tl.losses.CosineMSELogLoss,
    'CosineMSELoss': crested.tl.losses.CosineMSELoss,
    'PoissonLoss': crested.tl.losses.PoissonLoss,
    'PoissonMultinomialLoss': crested.tl.losses.PoissonMultinomialLoss
}

# Add global CSS for font settings
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load data
cell_type_colors = load_cell_type_colors(os.path.join(DATA_DIR, 'other/cell_type_colors.csv'))
diff_peaks_df, diff_peaks_columns = load_differential_peaks()
bigwig_df = scan_for_bigwigs()
peak_table_options = scan_for_peak_tables(DATA_DIR)
gene_annotations = load_gene_annotations(DATA_DIR)

# Load hierarchy information
hierarchy_df = pd.read_csv(os.path.join(DATA_DIR, 'other/AIT21_cldf.csv'))
# Drop duplicates while preserving the first occurrence of each unique combination
hierarchy_df = hierarchy_df.drop_duplicates(subset=['class_id_label', 'subclass_id_label', 'supertype_id_label'], keep='first')

# Create hierarchy mappings
class_to_subclass = hierarchy_df.groupby('class_id_label')['subclass_id_label'].apply(list).to_dict()
subclass_to_supertype = hierarchy_df.groupby('subclass_id_label')['supertype_id_label'].apply(list).to_dict()


# Create BigWig file mappings
class_bigwigs = {
    row['class_id_label']: {
        'file': os.path.join(DATA_DIR, f"class/{row['class_id_label']}.bw"),
        'url': f"/assets/class/{row['class_id_label']}.bw"
    } for _, row in hierarchy_df.drop_duplicates('class_id_label').iterrows()
}

subclass_bigwigs = {
    row['subclass_id_label']: {
        'file': os.path.join(DATA_DIR, f"subclass/{row['subclass_id_label']}.bw"),
        'url': f"/assets/subclass/{row['subclass_id_label']}.bw"
    } for _, row in hierarchy_df.drop_duplicates('subclass_id_label').iterrows()
}

supertype_bigwigs = {
    row['supertype_id_label']: {
        'file': os.path.join(DATA_DIR, f"supertype/{row['supertype_id_label']}.bw"),
        'url': f"/assets/supertype/{row['supertype_id_label']}.bw"
    } for _, row in hierarchy_df.drop_duplicates('supertype_id_label').iterrows()
}

# Create dropdown options
class_options = [
    {'label': class_name, 'value': class_bigwigs[class_name]['url']}
    for class_name in sorted(class_bigwigs.keys())
]

subclass_options = [
    {'label': subclass, 'value': subclass_bigwigs[subclass]['url']}
    for subclass in sorted(subclass_bigwigs.keys())
]

supertype_options = [
    {'label': supertype, 'value': supertype_bigwigs[supertype]['url']}
    for supertype in sorted(supertype_bigwigs.keys())
]

# Load model settings for dropdown
model_settings_df = pd.read_csv(os.path.join(DATA_DIR, 'model/model_settings.csv'))
logger.info(f"Loaded model settings from: {os.path.join(DATA_DIR, 'model/model_settings.csv')}")
logger.info(f"Found {len(model_settings_df)} models in settings file")

model_options = []
for _, row in model_settings_df.iterrows():
    # Convert relative path to absolute path
    model_path = os.path.join(DATA_DIR, row['model_file'].replace('./data/', ''))
    logger.info(f"Adding model option: {row['model_name']} -> {model_path}")
    model_options.append({
        'label': row['model_name'],
        'value': model_path
    })


tab_selected_style = {
    'backgroundColor': '#156082',
    'color': 'white',
}

# Add CRESTED model state
crested_model = None

# layout
# title
app.layout = html.Div([
    #links
    html.Div([
        html.A("|UCSC genome browser|", href="https://genome.ucsc.edu/cgi-bin/hgTracks?db=mm10", style={'margin-right': '10px'}, target="_blank"),
        html.A("|HOF enhancers|", href="https://enhancer-cheatsheet.replit.app/", style={'margin-right': '10px'}, target="_blank"),
        html.A("|AIT21_cl.df_and_prioritization|", href="https://docs.google.com/spreadsheets/d/17XN915ZXh2pXy_KUMhVf1z4sdroHxqvttznmiIXEQ70/edit?usp=sharing", style={'margin-right': '10px'}, target="_blank")

        ], style={'float': 'right'}),
    html.H1(
        [
            html.Span("D", style={"color": "blue", "fontSize": "48px"}),
            html.Span("e", style={"color": "black", "fontSize": "32px"}),
            html.Span("N", style={"color": "red", "fontSize": "48px"}),
            html.Span("A", style={"color": "green", "fontSize": "48px"}),
            html.Span("l", style={"color": "black", "fontSize": "32px"}),
            html.Span("i", style={"color": "black", "fontSize": "32px"})
        ],
        style={"textAlign": "left", "fontFamily": "Arial"}
    ),


    
    # Tabs
    dcc.Tabs([
        # Genomic Viewer Tab (Differential Peaks and BigWig Viewer)
        dcc.Tab(label='Bigwig Viewer',selected_style=tab_selected_style,
            children=[
            # Differential Peaks Table
            html.Div([
                html.H2('Differential Peaks'),
                html.Div([
                    html.Label('Select Peak Table:'),
                    dcc.Dropdown(
                        id='peak-table-dropdown',
                        options=peak_table_options,
                        value=None,
                        style={'width': '300px', 'marginBottom': '20px'}
                        ),
                    
                    #hmm doesn't seem to work
                    html.Div([dcc.Loading(
                        id="loading-table",
                        type="dot",  # options: "default", "circle", "dot", "cube"
                        children=html.Div(
                            DataTable(
                                id='peaks-table',
                                columns=diff_peaks_columns,
                                data=diff_peaks_df.to_dict('records'),
                                page_size=10,
                                filter_action="native",
                                sort_action="native",
                                sort_mode="single",
                                style_table={'width': '100%', 'minWidth': '100%', 'overflowX': 'auto'},
                                style_cell={
                                'textAlign': 'left',
                                'padding': '5px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'cursor': 'pointer',
                                'minWidth': '100px',
                                'maxWidth': '200px'
                                },
                                style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold',
                                'textAlign': 'left'
                                },
                                cell_selectable=True,
                                selected_cells=[],
                                row_selectable='single',
                                selected_rows=[],
                                style_data_conditional=[
                                {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                                },
                                {
                                'if': {'state': 'selected'},
                                'backgroundColor': 'rgb(200, 230, 255)',
                                'border': '1px solid rgb(0, 116, 217)'
                                }
                                ],
                                filter_options={
                                'case': 'insensitive',
                                'placeholder': 'Filter...'
                                },
                                page_action="native",
                                page_current=0
                                )
                            ),
                        style={"transform": "scale(2)", "transformOrigin": "left top"}
                        )]),


                ], style={'marginBottom': '30px', 'width': '100%'}),
                
                # BigWig Viewer Controls
                html.Div([
                    html.Label('mm10 Genomic Coordinates (e.g., chr18:58788458-58788958):'),
                    dcc.Input(
                        id='coordinates-input',
                        value='chr18:58788458-58788958',
                        type='text',
                        style={'width': '300px', 'marginRight': '10px'}
                    ),

                    html.Button('Update Plot', id='update-button', n_clicks=0, style={'marginRight': '10px', 'backGroundColor': "Springgreen"}),

                    html.Button('Zoom Out 3x', id='zoom-out-3x', n_clicks=0, 
                               style={'marginRight': '10px'}),
                    html.Button('Zoom Out 10x', id='zoom-out-10x', n_clicks=0, 
                               style={'marginRight': '10px'}),
                    html.Button('Zoom In 3x', id='zoom-in-3x', n_clicks=0, 
                               style={'marginRight': '10px'}),
                    html.Button('Zoom In 10x', id='zoom-in-10x', n_clicks=0, 
                               style={'marginRight': '20px'}),
                    
                    # Pan controls
                    html.Label('Pan:', style={'marginRight': '10px', 'fontWeight': 'bold'}),
                    html.Button('<<<\n100kb', id='pan-left-100k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '5px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'}),
                    html.Button('<<\n10kb', id='pan-left-10k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '5px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'}),
                    html.Button('<\n1kb', id='pan-left-1k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '10px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'}),
                    html.Button('>\n1kb', id='pan-right-1k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '5px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'}),
                    html.Button('>>\n10kb', id='pan-right-10k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '5px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'}),
                    html.Button('>>>\n100kb', id='pan-right-100k', n_clicks=0, 
                               style={'width': '50px', 'height': '50px', 'marginRight': '10px', 
                                     'whiteSpace': 'pre-line', 'fontSize': '12px', 'lineHeight': '1.2'})
                ], style={'marginBottom': '20px'}),
                
                # Hidden div to store original coordinates for highlighting
                html.Div(id='original-coordinates-store', style={'display': 'none'}),
                
                # BigWig Plot
                html.Div([
                    html.H3("BigWig Viewer", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    # Input boxes row
                    html.Div([
                        # Class column
                        html.Div([
                            html.H3('Class', style={'marginBottom': '20px'}),
                            dcc.Dropdown(
                                id='class-dropdown',
                                options=class_options,
                                value=[class_bigwigs[class_name]['url'] for class_name in sorted(class_bigwigs.keys())],
                                multi=True,
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select...',
                                optionHeight=35,
                                maxHeight=200
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Subclass column
                        html.Div([
                            html.H3('Subclass', style={'marginBottom': '20px'}),
                            dcc.Dropdown(
                                id='subclass-class-filter',
                                options=[
                                    {'label': class_name, 'value': class_name}
                                    for class_name in sorted(class_bigwigs.keys())
                                ],
                                value=None,
                                placeholder='Filter by class...',
                                clearable=True
                            ),
                            dcc.Dropdown(
                                id='subclass-dropdown',
                                options=subclass_options,
                                multi=True,
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select...',
                                optionHeight=35,
                                maxHeight=200,
                                style={'width': '100%', 'marginBottom': '20px'}
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Supertype column
                        html.Div([
                            html.H3('Supertype', style={'marginBottom': '20px'}),
                            dcc.Dropdown(
                                id='supertype-subclass-filter',
                                options=supertype_options,
                                value=None,
                                placeholder='Filter by subclass...',
                                clearable=True
                            ),
                            dcc.Dropdown(
                                id='supertype-dropdown',
                                options=supertype_options,
                                multi=True,
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select...',
                                optionHeight=35,
                                maxHeight=200,
                                style={'width': '100%', 'marginBottom': '20px'}
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'})
                    ], style={'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'space-between'}),
                    
                    # Plots row
                    html.Div([
                        # Class plot
                        html.Div([dcc.Loading(
                        id="loading-class",
                        type="dot",  # options: "default", "circle", "dot", "cube"
                        children=html.Img(id="class-plot", className='plot-container'),
                        style={"transform": "scale(2)", "transformOrigin": "left top"}
                        )], className='plot-column'),

                        # Subclass plot
                        html.Div([dcc.Loading(
                        id="loading-subclass",
                        type="dot",  # options: "default", "circle", "dot", "cube"
                        children=html.Img(id="subclass-plot", className='plot-container'),
                        style={"transform": "scale(2)", "transformOrigin": "left top"}
                        )], className='plot-column'),

                        # Supertype plot
                        html.Div([dcc.Loading(
                        id="loading-supertype",
                        type="dot",  # options: "default", "circle", "dot", "cube"
                        children=html.Img(id="supertype-plot", className='plot-container'),
                        style={"transform": "scale(2)", "transformOrigin": "left top"}
                        )], className='plot-column'),
                    ], style={'display': 'flex', 'justifyContent': 'space-between'})
                ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ], style={'padding': '20px', 'overflowY': 'auto'})
        ]),
        
        # Sequence Analysis Tab
        dcc.Tab(label='Sequence Analysis',selected_style=tab_selected_style, children=[
            html.Div([
                html.H2('Sequence Viewer and Analysis'),
                
                # Sequence Input Section
                html.Div([
                    html.H3('Sequence Input'),
                    html.Div([
                        # Genomic coordinates input
                        html.Div([
                            html.Label('Genomic Coordinates (e.g., chr18:58788458-58788958):'),
                            dcc.Input(
                                id='sequence-coordinates-input',
                                value='chr19:23980545-23981046',
                                type='text',
                                style={'width': '300px', 'marginRight': '10px'}
                            ),
                            html.Button('Get Sequence', id='get-sequence-button', n_clicks=0, 
                                       style={'marginRight': '10px'}),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Custom sequence input
                        html.Div([
                            html.Label('Or Enter Custom Sequence:'),
                            dcc.Textarea(
                                id='custom-sequence-input',
                                placeholder='Enter DNA sequence...',
                                style={'width': '300px', 'height': '30px', 'marginRight': '10px'}
                            ),
                            html.Button('Use Custom Sequence', id='use-custom-sequence-button', n_clicks=0),
                            html.Button('Remove sequence', id='clear-sequence-button', n_clicks=0),
                        ], style={'display': 'inline-block'}),
                        
                        # Sequence analysis options
                        dcc.Checklist(
                            id='sequence-options',
                            options=[
                                {'label': 'Show reverse complement', 'value': 'rev_comp'},
                                {'label': 'Show GC content', 'value': 'gc_content'},
                            ],
                            value=[],
                            inline=True,
                            style={'marginLeft': '10px'}
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # Motif search input
                    html.Div([
                        html.Label('Search for motif:', style={'marginRight': '5px'}),
                        dcc.Input(id='motif-input', type='text', placeholder='e.g., TATAAA', 
                                 style={'width': '120px', 'marginRight': '10px'}),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div(id='sequence-output', className='sequence-output')
                ], style={'marginBottom': '30px'}),
                
                # CRESTED Model Section
                html.Div([
                    html.H3('CRESTED Model'),
                    html.Div([
                        # Model selection dropdown
                        html.Div([
                            html.Label('Select Model:'),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=model_options,
                                value=None,
                                style={'width': '300px', 'marginRight': '10px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Custom model input
                        html.Div([
                            html.Label('Or Load Custom Model:'),
                            dcc.Input(
                                id='custom-model-path',
                                type='text',
                                placeholder='Enter path to custom model',
                                style={'width': '300px', 'marginRight': '10px'}
                            ),
                        ], style={'display': 'inline-block'}),
                        html.Div([dcc.Loading(

                            id="loading-model",
                            type="dot",  # options: "default", "circle", "dot", "cube"
                            children= html.Button('Load Model', id='load-model-button', n_clicks=0,
                                style={'marginRight': '10px'}),
                            style={"transform": "scale(2)", "transformOrigin": "left top"}
                            )]),
                        
                    ], style={'marginBottom': '20px'}),
                    
                    # Class selection dropdown
                    html.Div([
                        html.Div([
                            html.Label('Select Classes to Analyze:'),
                            dcc.Checklist(
                                id='select-all-classes',
                                options=[{'label': 'Select All', 'value': 'all'}],
                                value=[],
                                style={'marginLeft': '10px', 'display': 'inline-block'}
                            )
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='class-selection-dropdown',
                            multi=True,
                            style={'width': '100%', 'marginBottom': '20px'}
                        ),
                    ]),
                    html.Button('Run Scores', id='run-scores-button', n_clicks=0),
                    html.Label('  Zoom to: '),
                    dcc.Textarea(
                        id='zoom-input',
                        placeholder='n bases...',
                        style={'width': '300px', 'height': '20px', 'marginRight': '10px'}
                        ),
                    
                    # Plot type selector
                    html.Div([
                        html.Label('Plot Type: '),
                        dcc.RadioItems(
                            id='plot-type-selector',
                            options=[
                                {'label': 'Sequence Logo', 'value': 'line'},
                                {'label': 'Heatmap', 'value': 'heatmap'}
                            ],
                            value='heatmap',
                            inline=True,
                            style={'marginLeft': '10px'}
                        ),
                    ], style={'marginTop': '10px', 'marginBottom': '10px'}),
                    
                    # Heatmap options (shown by default since heatmap is default)
                    html.Div(id='heatmap-options', style={'display': 'block', 'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}, children=[
                        html.Div([
                            html.Label('Color Map: '),
                            dcc.Dropdown(
                                id='heatmap-cmap',
                                options=[
                                    {'label': 'Cool Warm', 'value': 'coolwarm'},
                                    {'label': 'RdBu', 'value': 'RdBu'},
                                    {'label': 'Viridis', 'value': 'viridis'},
                                    {'label': 'Plasma', 'value': 'plasma'},
                                    {'label': 'Inferno', 'value': 'inferno'}
                                ],
                                value='coolwarm',
                                style={'width': '150px', 'display': 'inline-block', 'marginLeft': '10px'}
                            ),
                        ], style={'marginTop': '5px', 'marginBottom': '5px'}),
                    ]),
                    
                    html.Div([dcc.Loading(
                        id="loading",
                        type="dot",  # options: "default", "circle", "dot", "cube"
                        children=html.Div(id="contribution-scores-plot"),
                        style={"transform": "scale(2)", "transformOrigin": "left top"}
                        )
                    ]),
                    
                    html.Div(id='model-status', style={'marginBottom': '20px'}),
                    #html.Div(id='contribution-scores-plot', style={'marginBottom': '20px'}),
                    
                    # Download button and component
                    html.Div([
                        html.Button('Download Plot', id='download-plot-button', n_clicks=0,
                                  style={'marginTop': '10px'}),
                        dcc.Download(id='plot-download')
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),
                    
                    # Batch processing section
                    html.Div([
                        html.H3('Batch Processing', style={'marginTop': '30px'}),
                        html.P('Upload a CSV file containing coordinates to process in batch. The CSV should have a column named "coordinates" with values in the format "chr:start-end".'),
                        dcc.Upload(
                            id='batch-coordinates-upload',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select CSV File')
                            ]),
                            className='upload-area',
                            multiple=False
                        ),
                        html.Div(id='batch-upload-status', style={'marginBottom': '10px'}),
                        html.Button('Run Scores Batch', id='run-batch-button', n_clicks=0),
                        html.Div(id='batch-processing-status', style={'marginTop': '10px'}),
                        dcc.Download(id='batch-download')
                    ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                ], style={'marginBottom': '30px'}),
            ], style={'padding': '20px'})
        ]),
    ])
], style={'margin': '0 auto', 'padding': '20px', 'width': '100%'})

# Add global variable to store the current plot bytes
current_plot_bytes = None

# Add global variable to store the current plot type
current_plot_type = 'heatmap'

# Add global variable to store batch results
batch_results = None

@callback(
    [Output('model-status', 'children'),
     Output('run-scores-button', 'disabled')],
    [Input('load-model-button', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('custom-model-path', 'value')]
)
def load_crested_model(n_clicks, selected_model, custom_model_path):
    if n_clicks == 0:
        return "No model loaded", True
    
    global crested_model
    try:
        # Determine which model path to use
        model_path = custom_model_path if custom_model_path else selected_model
        if not model_path:
            return "Please select a model or provide a custom model path", True
        
        logger.info(f"Attempting to load model from path: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at path: {model_path}")
            return html.Div([
                html.P("Error loading model:"),
                html.P(f"Model file not found at: {model_path}")
            ]), True
        
        # Load model settings
        logger.info(f"Loading model settings for: {model_path}")
        model_settings = load_model_settings(model_path, DATA_DIR)
        
        # Load the model using keras with custom objects
        logger.info("Loading model with keras...")
        crested_model = keras.models.load_model(model_path, custom_objects=custom_objects)
        
        # Store model settings in the model object
        crested_model.settings = model_settings
        
        return html.Div([
            html.P("Model loaded successfully!"),
        ]), False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return html.Div([
            html.P("Error loading model:"),
            html.P(str(e))
        ]), True

@callback(
    [Output('class-selection-dropdown', 'options'),
     Output('select-all-classes', 'value')],
    [Input('model-status', 'children')]
)
def update_class_selection(model_status):
    if not crested_model:
        return [], []
    
    try:
        model_settings = getattr(crested_model, 'settings', {})
        class_labels = model_settings.get('indexed_class_labels', [])
        
        if not class_labels:
            logger.warning("No class labels found in model settings")
            return [], []
        
        # Create options for the dropdown
        options = [{'label': label, 'value': i} for i, label in enumerate(class_labels)]
        logger.info(f"Populated class selection dropdown with {len(options)} options")
        return options, []
    except Exception as e:
        logger.error(f"Error updating class selection: {str(e)}")
        return [], []

@callback(
    Output('heatmap-options', 'style'),
    [Input('plot-type-selector', 'value')]
)
def toggle_heatmap_options(plot_type):
    """Show/hide heatmap options based on plot type selection."""
    if plot_type == 'heatmap':
        return {'display': 'block', 'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}
    else:
        return {'display': 'none'}

@callback(
    Output('contribution-scores-plot', 'children'),
    [Input('run-scores-button', 'n_clicks')],
    [State('sequence-coordinates-input', 'value'),
     State('class-selection-dropdown', 'value'),
     State('custom-sequence-input', 'value'),
     State('zoom-input', 'value'),
     State('plot-type-selector', 'value'),
     State('heatmap-cmap', 'value')
     ]
)

def run_contribution_scores(n_clicks, coordinates, selected_classes, custom_sequence=None, zoom_to=500, plot_type='heatmap', heatmap_cmap='coolwarm'):
    if (zoom_to is None) | (zoom_to == ''):
        zoom_to = 500
    zoom_to = int(zoom_to)
    if n_clicks == 0 or not crested_model:
        return ""

    try:
        logger.info("Starting contribution scores calculation...")
        
        model_settings = crested_model.settings
        model_settings = getattr(crested_model, 'settings', {
            'sequence_length': int(model_settings['sequence_length']),
            'batch_size': int(model_settings['batch_size']),
            'zoom_n_bases': zoom_to
        })
        
        # Get the sequence
        chrom, start, end = parse_coordinates(coordinates)
        
        # Get the sequence length from model settings
        required_sequence_length = model_settings['sequence_length']
        logger.info(f"Using sequence length from model settings: {required_sequence_length}")
        
        # Calculate center position
        center = (start + end) // 2
        
        # Calculate new start and end positions to get the required sequence length
        half_length = required_sequence_length // 2
        new_start = center - half_length
        new_end = center + half_length
        
        # Define genome path and check if it exists
        genome_path = os.path.join(DATA_DIR, 'genome/mm10.fa')
        if not os.path.exists(genome_path):
            raise FileNotFoundError(f"Genome file not found at {genome_path}")
        
        # Get the sequence
        try:
            # Load the genome file
            genome = Fasta(genome_path)
            
            # Check if chromosome exists
            if chrom not in genome:
                raise ValueError(f"Chromosome {chrom} not found in genome file")
            
            # Get chromosome length
            chrom_length = len(genome[chrom])
            logger.info(f"Chromosome {chrom} length: {chrom_length}")
            
            # Validate coordinates
            if new_start < 0 or new_end > chrom_length:
                raise ValueError(f"Coordinates {new_start}-{new_end} are out of range for chromosome {chrom} (length: {chrom_length})")
            
            # Get the sequence
            logger.info(f"Getting sequence for {chrom}:{new_start}-{new_end}")
            if (custom_sequence is None) | (custom_sequence==''):
                sequence = genome[chrom][new_start:new_end]
                seq_str = str(sequence.seq).upper()
                title = f"Contribution Scores for {chrom}:{start}-{end}"
            elif custom_sequence is not None:
                if len(custom_sequence) ==model_settings['sequence_length']:
                    seq_str = custom_sequence
                    title = 'Contribution Scores for custom sequence'
                elif len(custom_sequence) < model_settings['sequence_length']:
                    length = len(custom_sequence)
                    pad_left = ''.join(list(repeat('N',(model_settings['sequence_length']-length)//2) ))# add n on each side
                    pad_right = ''.join(list(repeat('N',(model_settings['sequence_length']-len(pad_left)-length))))
                    seq_str = str(pad_left)+str(custom_sequence)+str(pad_right)
                    title = 'Contribution Scores for custom sequence'
                    print(custom_sequence)
                    print(seq_str)


            
            # Validate sequence
            if not seq_str:
                raise ValueError("Empty sequence returned")
            
            
        except Exception as e:
            logger.error(f"Error getting sequence: {str(e)}")
            raise ValueError(f"Failed to get valid sequence: {str(e)}")
        
        # Run contribution scores
        logger.info("Calculating contribution scores...")
        try:
            # Get the number of classes from the model's output shape
            n_classes = crested_model.output_shape[1]
            logger.info(f"Model has {n_classes} classes")
            
            # Use selected classes if provided, otherwise use all classes
            if selected_classes is not None and len(selected_classes) > 0:
                target_idx = selected_classes
            else:
                target_idx = list(range(n_classes))
            
            scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
                method='integrated_grad',
                input=seq_str,
                target_idx=target_idx,
                genome=genome_path,
                model=crested_model,
                batch_size=model_settings['batch_size']
            )
        except Exception as e:
            logger.error(f"Error in contribution_scores calculation: {str(e)}")
            raise
        
        # Get class labels from model settings
        class_labels = model_settings.get('indexed_class_labels', [f'Class {i+1}' for i in range(n_classes)])
        # Filter class labels to only show selected ones
        if selected_classes is not None and len(selected_classes) > 0:
            class_labels = [class_labels[i] for i in selected_classes]
        logger.info(f"Using class labels: {class_labels}")
        
        # Create the plot based on selected type
        logger.info(f"Plot type selected: {plot_type}")
        if plot_type == 'line':
            base64_data, plot_bytes = create_contribution_scores_plot(
                scores, 
                one_hot_encoded_sequences, 
                chrom, 
                start, 
                end, 
                new_start, 
                new_end,
                n_classes=len(class_labels),
                class_labels=class_labels,
                zoom_n_bases=zoom_to,
                title=title
            )
        elif plot_type == 'heatmap':
            print(f"DEBUG: Heatmap branch entered")
            # Create contribution scores dataframe using the already calculated scores
            coordinates_str = f"{chrom}:{new_start}-{new_end}"
            
            try:
                score_df = create_contribution_scores_dataframe(
                    scores=scores,
                    one_hot_encoded_sequences=one_hot_encoded_sequences,
                    class_labels=class_labels,
                    coordinates=coordinates_str,
                    sequence=seq_str
                )
                logger.info(f"Successfully created DataFrame with shape: {score_df.shape}")
            except Exception as e:
                logger.error(f"Error in create_contribution_scores_dataframe: {str(e)}")
                raise
            
            # Create heatmap
            
            base64_data, plot_bytes = plot_heatmap(
                df=score_df,
                coordinates=coordinates_str,
                zoom_to=zoom_to,
                figsize=(30, 5),
                cmap=heatmap_cmap,
                z_score=0,  # No z-score by default
                col_cluster=False
            )
        
        # Store the plot bytes and type globally
        global current_plot_bytes, current_plot_type
        current_plot_bytes = plot_bytes
        current_plot_type = plot_type
        
        # Format model settings for display
        display_settings = {
            'sequence_length': required_sequence_length,
            'batch_size': model_settings['batch_size'],
            'zoom_n_bases': model_settings['zoom_n_bases'],
            'class_labels': ', '.join(class_labels)
        }
        
        return html.Div([
            html.P(f"Original coordinates: {chrom}:{start}-{end}"),
            html.P(f"Adjusted coordinates: {chrom}:{new_start}-{new_end}"),
            html.Img(src=base64_data, style={'width': '100%'})
        ])
    except Exception as e:
        error_msg = f"Error running contribution scores: {str(e)}"
        logger.error(error_msg)
        return html.Div([
            html.P("Error running contribution scores:"),
            html.P(str(e))
        ])

@callback(
    Output('plot-download', 'data'),
    [Input('download-plot-button', 'n_clicks')],
    [State('sequence-coordinates-input', 'value'),
     State('custom-sequence-input', 'value'),
     State('plot-type-selector', 'value')]
)
def download_plot(n_clicks, coordinates, custom_sequence, plot_type):
    global current_plot_type
    # Use the stored plot type if not provided
    if plot_type is None:
        plot_type = current_plot_type
    if n_clicks == 0 or not current_plot_bytes:
        return dash.no_update
    
    try:
        # Create filename based on plot type and whether custom sequence is used
        plot_type_suffix = f"_{plot_type}" if plot_type else ""
        if custom_sequence and custom_sequence.strip():
            filename = f"contribution_scores{plot_type_suffix}.png"
        else:
            # Clean up coordinates for filename
            clean_coords = coordinates.replace(':', '_').replace('-', '_')
            filename = f"contribution_scores{plot_type_suffix}_{clean_coords}.png"
        
        return dcc.send_bytes(current_plot_bytes, filename)
    except Exception as e:
        logger.error(f"Error downloading plot: {str(e)}")
        return dash.no_update

@callback(
    [Output('coordinates-input', 'value'),
     Output('sequence-coordinates-input', 'value'),
     Output('class-plot', 'src'),
     Output('subclass-plot', 'src'),
     Output('supertype-plot', 'src'),
     Output('peaks-table', 'selected_rows'),
     Output('original-coordinates-store', 'children')],
    [Input('peaks-table', 'selected_cells'),
     Input('update-button', 'n_clicks'),
     Input('zoom-out-3x', 'n_clicks'),
     Input('zoom-out-10x', 'n_clicks'),
     Input('zoom-in-3x', 'n_clicks'),
     Input('zoom-in-10x', 'n_clicks'),
     Input('pan-left-100k', 'n_clicks'),
     Input('pan-left-10k', 'n_clicks'),
     Input('pan-left-1k', 'n_clicks'),
     Input('pan-right-1k', 'n_clicks'),
     Input('pan-right-10k', 'n_clicks'),
     Input('pan-right-100k', 'n_clicks'),
     Input('class-dropdown', 'value'),
     Input('subclass-dropdown', 'value'),
     Input('supertype-dropdown', 'value')],
    [State('peaks-table', 'data'),
     State('peaks-table', 'derived_virtual_data'),
     State('peaks-table', 'derived_virtual_selected_rows'),
     State('peaks-table', 'page_current'),
     State('peaks-table', 'page_size'),
     State('coordinates-input', 'value'),
     State('original-coordinates-store', 'children')],
    prevent_initial_call=False
)
def update_coordinates_and_plots(selected_cells, update_clicks, zoom_out_3x, zoom_out_10x, zoom_in_3x, zoom_in_10x, 
                               pan_left_100k, pan_left_10k, pan_left_1k, pan_right_1k, pan_right_10k, pan_right_100k,
                               class_files, subclass_files, supertype_files, table_data, derived_virtual_data, 
                               derived_virtual_selected_rows, page_current, page_size, current_coords, stored_original_coords):
    # Get the context to determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        # On initial load, use the default coordinates with default zoom
        default_coords = 'chr18:58788458-58788958'
        zoom_level = -1000  # Default zoom out 1kb
        
        # Convert URLs to file paths
        class_file_paths = [class_bigwigs[class_name]['file'] for class_name in sorted(class_bigwigs.keys())]
        subclass_file_paths = [subclass_bigwigs[subclass]['file'] for subclass in sorted(subclass_bigwigs.keys())]
        supertype_file_paths = [supertype_bigwigs[supertype]['file'] for supertype in sorted(supertype_bigwigs.keys())]
        
        class_plot = create_plot(class_file_paths, default_coords, zoom_level, default_coords, cell_type_colors, 1.0, gene_annotations)
        subclass_plot = create_plot(subclass_file_paths, default_coords, zoom_level, default_coords, cell_type_colors, 1.0, gene_annotations)
        supertype_plot = create_plot(supertype_file_paths, default_coords, zoom_level, default_coords, cell_type_colors, 1.0, gene_annotations)
        return default_coords, default_coords, class_plot, subclass_plot, supertype_plot, [], default_coords
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize coordinates
    new_coords = current_coords
    original_coords = stored_original_coords if stored_original_coords is not None else current_coords
    selected_row = None
    zoom_factor = 1.0
    
    
    zoom_level = -1000  # Default 1kb zoom-out
    
    # Handle zoom controls
    if trigger_id in ['zoom-out-3x', 'zoom-out-10x', 'zoom-in-3x', 'zoom-in-10x']:
        if trigger_id == 'zoom-out-3x':
            zoom_factor = 3.0
        elif trigger_id == 'zoom-out-10x':
            zoom_factor = 10.0
        elif trigger_id == 'zoom-in-3x':
            zoom_factor = 1.0/3.0
        elif trigger_id == 'zoom-in-10x':
            zoom_factor = 1.0/10.0
        
        # Apply zoom to current coordinates and update new_coords
        # Keep original_coords unchanged for highlighting
        try:
            chrom, start, end = parse_coordinates(current_coords)
            center = (start + end) // 2
            half_width = (end - start) // 2
            
            # Apply the zoom factor
            new_half_width = int(half_width * zoom_factor)
            new_half_width = max(50, new_half_width)  # Minimum 100bp window
            
            new_start = center - new_half_width
            new_end = center + new_half_width
            new_coords = f"{chrom}:{new_start}-{new_end}"
            
        except Exception as e:
            logger.error(f"Error applying zoom: {e}")
            # Fall back to original coordinates if parsing fails
            new_coords = current_coords
    
    # Handle pan controls
    if trigger_id in ['pan-left-100k', 'pan-left-10k', 'pan-left-1k', 'pan-right-1k', 'pan-right-10k', 'pan-right-100k']:
        try:
            chrom, start, end = parse_coordinates(current_coords)
            window_size = end - start
            
            # Determine pan distance
            if trigger_id == 'pan-left-100k':
                pan_distance = -100000
            elif trigger_id == 'pan-left-10k':
                pan_distance = -10000
            elif trigger_id == 'pan-left-1k':
                pan_distance = -1000
            elif trigger_id == 'pan-right-1k':
                pan_distance = 1000
            elif trigger_id == 'pan-right-10k':
                pan_distance = 10000
            elif trigger_id == 'pan-right-100k':
                pan_distance = 100000
            
            # Apply pan
            new_start = start + pan_distance
            new_end = end + pan_distance
            new_coords = f"{chrom}:{new_start}-{new_end}"
            
        except Exception as e:
            logger.error(f"Error applying pan: {e}")
            # Fall back to original coordinates if parsing fails
            new_coords = current_coords
    
    # Handle user input updates
    if trigger_id == 'update-button':
        new_coords = current_coords
        original_coords = current_coords
    
    # Handle table selection
    if trigger_id == 'peaks-table' and selected_cells:
        # Get the first selected cell
        cell = selected_cells[0]
        row_idx = cell['row']
        
        # Determine which data to use based on whether the table is filtered
        if derived_virtual_data:
            data_to_use = derived_virtual_data
            actual_row_idx = row_idx + (page_current * page_size)
        else:
            data_to_use = table_data
            actual_row_idx = row_idx + (page_current * page_size)
        
        # Get the coordinates from the 'coordinates' column of the selected row
        if actual_row_idx < len(data_to_use):
            row_data = data_to_use[actual_row_idx]
            if 'coordinates' in row_data:
                coord_value = row_data['coordinates']
                if isinstance(coord_value, str) and re.match(r'[^:]+:\d+-\d+', coord_value):
                    new_coords = coord_value
                    original_coords = coord_value
                    selected_row = [row_idx]
                else:
                    logger.warning(f"Invalid coordinate format: {coord_value}")
            else:
                logger.warning("No coordinates found in row data")
        else:
            logger.warning(f"Row index {actual_row_idx} out of range for data length {len(data_to_use)}")
    
    # Convert URLs to file paths
    class_file_paths = []
    if class_files:
        for url in class_files:
            # Find the class name from the URL
            class_name = next((name for name, data in class_bigwigs.items() if data['url'] == url), None)
            if class_name:
                class_file_paths.append(class_bigwigs[class_name]['file'])
    
    subclass_file_paths = []
    if subclass_files:
        for url in subclass_files:
            # Find the subclass name from the URL
            subclass_name = next((name for name, data in subclass_bigwigs.items() if data['url'] == url), None)
            if subclass_name:
                subclass_file_paths.append(subclass_bigwigs[subclass_name]['file'])
    
    supertype_file_paths = []
    if supertype_files:
        for url in supertype_files:
            # Find the supertype name from the URL
            supertype_name = next((name for name, data in supertype_bigwigs.items() if data['url'] == url), None)
            if supertype_name:
                supertype_file_paths.append(supertype_bigwigs[supertype_name]['file'])
    
    # Create plots for each category
    # For zoom controls, we've already applied the zoom to new_coords, so use zoom_factor=1.0
    # For other triggers (like default zoom), use the original zoom_factor
    plot_zoom_factor = 1.0 if trigger_id in ['zoom-out-3x', 'zoom-out-10x', 'zoom-in-3x', 'zoom-in-10x'] else zoom_factor
    
    class_plot = create_plot(class_file_paths, new_coords, zoom_level, original_coords, cell_type_colors, plot_zoom_factor, gene_annotations)
    subclass_plot = create_plot(subclass_file_paths, new_coords, zoom_level, original_coords, cell_type_colors, plot_zoom_factor, gene_annotations)
    supertype_plot = create_plot(supertype_file_paths, new_coords, zoom_level, original_coords, cell_type_colors, plot_zoom_factor, gene_annotations)
    
    return new_coords, original_coords, class_plot, subclass_plot, supertype_plot, selected_row if selected_row else dash.no_update, original_coords

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                  'N': 'N', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def calculate_gc_content(seq):
    """Calculate the GC content of a DNA sequence."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    total = len(seq) - seq.count('N')
    if total == 0:
        return 0
    return (gc_count / total) * 100

def highlight_motif(sequence, motif):
    """Split sequence into parts based on motif occurrences."""
    if not motif:
        return [(sequence, False)]
    
    # Case insensitive search
    motif_upper = motif.upper()
    sequence_upper = sequence.upper()
    
    parts = []
    last_end = 0
    
    for match in re.finditer(motif_upper, sequence_upper):
        start, end = match.span()
        # Add the part before the match
        if start > last_end:
            parts.append((sequence[last_end:start], False))
        # Add the matched part
        parts.append((sequence[start:end], True))
        last_end = end
    
    # Add the remaining part after the last match
    if last_end < len(sequence):
        parts.append((sequence[last_end:], False))
    
    return parts

def format_sequence_with_line_numbers(seq, width=50):
    """Format a sequence with line numbers and fixed width."""
    lines = []
    for i in range(0, len(seq), width):
        line_num = i + 1
        chunk = seq[i:i+width]
        lines.append(f"{line_num:8d} {chunk}")
    return '\n'.join(lines)



@app.callback(
    Output('custom-sequence-input', 'value'),  # Update the value of 'my-input'
    Input('clear-sequence-button', 'n_clicks') # Triggered by button clicks
    )
def clear_sequence_on_button_click(n_clicks):
    if n_clicks: # Check if the button has been clicked at least once
        return '' # restart
        return dash.no_update # Do not update if the button hasn't been clicked



@callback(
    Output('sequence-output', 'children'),
    [Input('get-sequence-button', 'n_clicks'),
     Input('use-custom-sequence-button', 'n_clicks'),
     Input('clear-sequence-button', 'n_clicks'),
     Input('motif-input', 'value')],
    [State('sequence-coordinates-input', 'value'),
     State('custom-sequence-input', 'value'),
     State('sequence-options', 'value')]
)
def get_sequence(get_clicks, custom_clicks,clear_sequence, motif, coordinates, custom_sequence, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Click a button to fetch sequence"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if trigger_id == 'clear-sequence-button':
            seq_str = ''
        elif trigger_id == 'get-sequence-button':
            # Parse coordinates
            chrom, start, end = parse_coordinates(coordinates)
            
            # Load mm10 genome from gzipped file
            genome = Fasta(os.path.join(DATA_DIR, 'genome/mm10.fa'))
            
            # Get sequence
            sequence = genome[chrom][start:end]
            seq_str = sequence.seq
        elif trigger_id == 'use-custom-sequence-button':
            if not custom_sequence:
                return "Please enter a custom sequence"
            seq_str = custom_sequence.upper()
        else:
            # If triggered by motif input, use the current sequence
            if not hasattr(get_sequence, 'last_sequence'):
                return "Please get a sequence first"
            seq_str = get_sequence.last_sequence
        
        # Store the current sequence for motif highlighting
        get_sequence.last_sequence = seq_str
        
        # Create output components
        output_components = [html.H4(f'Sequence:')]
        
        # Add sequence length info
        seq_length = len(seq_str)
        output_components.append(html.P(f"Length: {seq_length} bp"))
        
        # Calculate and display GC content if requested
        if 'gc_content' in options:
            gc_percent = calculate_gc_content(seq_str)
            output_components.append(html.P(f"GC Content: {gc_percent:.2f}%"))
        
        # Format sequence with line numbers and highlight motif if provided
        formatted_seq = format_sequence_with_line_numbers(seq_str)
        if motif:
            # Count occurrences
            motif_count = seq_str.upper().count(motif.upper())
            output_components.append(html.P(f"Motif '{motif}' found: {motif_count} occurrences"))
            
            # Split sequence into parts and highlight motif
            parts = highlight_motif(formatted_seq, motif)
            highlighted_sequence = []
            for text, is_motif in parts:
                if is_motif:
                    highlighted_sequence.append(html.Span(text, style={
                        'backgroundColor': 'yellow',
                        'fontWeight': 'bold'
                    }))
                else:
                    highlighted_sequence.append(text)
            
            output_components.append(html.Div([
                html.P("Forward Strand:"),
                html.Pre(highlighted_sequence, style={'fontFamily': 'monospace'})
            ]))
        else:
            # Display the sequence without highlighting
            output_components.append(html.Div([
                html.P("Forward Strand:"),
                html.Pre(formatted_seq, style={'fontFamily': 'monospace'})
            ]))
        
        # Show reverse complement if requested
        if 'rev_comp' in options:
            rev_comp_seq = reverse_complement(seq_str)
            formatted_rev_comp = format_sequence_with_line_numbers(rev_comp_seq)
            
            if motif:
                # Count occurrences in reverse complement
                rev_motif_count = rev_comp_seq.upper().count(motif.upper())
                output_components.append(html.P(f"Motif '{motif}' found in reverse complement: {rev_motif_count} occurrences"))
                
                # Split reverse complement into parts and highlight motif
                parts = highlight_motif(formatted_rev_comp, motif)
                highlighted_rev_comp = []
                for text, is_motif in parts:
                    if is_motif:
                        highlighted_rev_comp.append(html.Span(text, style={
                            'backgroundColor': 'yellow',
                            'fontWeight': 'bold'
                        }))
                    else:
                        highlighted_rev_comp.append(text)
                
                output_components.append(html.Div([
                    html.P("Reverse Complement:"),
                    html.Pre(highlighted_rev_comp, style={'fontFamily': 'monospace'})
                ]))
            else:
                output_components.append(html.Div([
                    html.P("Reverse Complement:"),
                    html.Pre(formatted_rev_comp, style={'fontFamily': 'monospace'})
                ]))
        
        return html.Div(output_components)
    except Exception as e:
        logger.error(f"Error in sequence analysis: {str(e)}")
        return f"Error analyzing sequence: {str(e)}"

# Add callback to handle select all checkbox
@callback(
    Output('class-selection-dropdown', 'value'),
    [Input('select-all-classes', 'value')],
    [State('class-selection-dropdown', 'options')]
)
def select_all_classes(select_all, options):
    if select_all and 'all' in select_all:
        return [option['value'] for option in options]
    return []

# Add a new callback to handle peak table selection
@callback(
    [Output('peaks-table', 'columns'),
     Output('peaks-table', 'data')],
    [Input('peak-table-dropdown', 'value')]
)
def update_peak_table(selected_table):
    if not selected_table:
        return dash.no_update, dash.no_update
    
    try:
        # Load the selected table
        df, columns = load_differential_peaks(selected_table)
        return columns, df.to_dict('records')
    except Exception as e:
        logger.error(f"Error updating peak table: {str(e)}")
        return dash.no_update, dash.no_update

@callback(
    Output('batch-upload-status', 'children'),
    [Input('batch-coordinates-upload', 'contents')],
    [State('batch-coordinates-upload', 'filename')]
)
def update_batch_upload_status(contents, filename):
    if contents is None:
        return ""
    
    try:
        # Parse the uploaded CSV
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Validate the CSV format
        if 'coordinates' not in df.columns:
            return html.Div("Error: CSV must contain a 'coordinates' column", style={'color': 'red'})
        
        # Validate coordinate format
        invalid_coords = []
        for coord in df['coordinates']:
            if not isinstance(coord, str) or not re.match(r'[^:]+:\d+-\d+', coord):
                invalid_coords.append(coord)
        
        if invalid_coords:
            return html.Div([
                html.P(f"Error: Found {len(invalid_coords)} invalid coordinate(s) in the format 'chr:start-end'", style={'color': 'red'}),
                html.P("Invalid coordinates:", style={'color': 'red'}),
                html.Pre(str(invalid_coords), style={'color': 'red'})
            ])
        
        return html.Div([
            html.P(f"Successfully loaded {len(df)} coordinates from {filename}"),
            html.P("Click 'Run Scores Batch' to process all coordinates")
        ])
    except Exception as e:
        return html.Div(f"Error processing file: {str(e)}", style={'color': 'red'})

@callback(
    [Output('batch-processing-status', 'children'),
     Output('batch-download', 'data')],
    [Input('run-batch-button', 'n_clicks')],
    [State('batch-coordinates-upload', 'contents'),
     State('class-selection-dropdown', 'value')]
)
def process_batch(n_clicks, contents, selected_classes):
    if n_clicks == 0 or not contents or not crested_model:
        return dash.no_update, dash.no_update
    
    try:
        # Parse the uploaded CSV
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Create a temporary directory for the plots
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get model settings
            model_settings = getattr(crested_model, 'settings', {
                'sequence_length': 1500,
                'batch_size': 128,
                'zoom_n_bases': 500
            })
            
            # Get class labels from model settings
            class_labels = model_settings.get('indexed_class_labels', [f'Class {i+1}' for i in range(crested_model.output_shape[1])])
            
            # Filter class labels based on selected classes
            if selected_classes is not None and len(selected_classes) > 0:
                class_labels = [class_labels[i] for i in selected_classes]
            
            # Process each coordinate
            for i, row in df.iterrows():
                coordinates = row['coordinates']
                try:
                    # Get the sequence
                    chrom, start, end = parse_coordinates(coordinates)
                    
                    # Calculate center position
                    center = (start + end) // 2
                    half_length = model_settings['sequence_length'] // 2
                    new_start = center - half_length
                    new_end = center + half_length
                    
                    # Get the sequence
                    genome = Fasta(os.path.join(DATA_DIR, 'genome/mm10.fa'))
                    sequence = genome[chrom][new_start:new_end]
                    seq_str = str(sequence.seq).upper()
                    
                    # Run contribution scores
                    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
                        input=seq_str,
                        target_idx=selected_classes if selected_classes else list(range(crested_model.output_shape[1])),
                        genome=os.path.join(DATA_DIR, 'genome/mm10.fa'),
                        model=crested_model,
                        batch_size=model_settings['batch_size']
                    )
                    
                    # Create the plot
                    base64_data, plot_bytes = create_contribution_scores_plot(
                        scores,
                        one_hot_encoded_sequences,
                        chrom,
                        start,
                        end,
                        new_start,
                        new_end,
                        n_classes=len(class_labels),
                        class_labels=class_labels,
                        zoom_n_bases=zoom_to
                    )
                    
                    # Save the plot
                    clean_coords = coordinates.replace(':', '_').replace('-', '_')
                    plot_path = os.path.join(temp_dir, f'contribution_scores_{clean_coords}.png')
                    
                    # Ensure the plot bytes are valid before saving
                    if plot_bytes and len(plot_bytes) > 0:
                        with open(plot_path, 'wb') as f:
                            f.write(plot_bytes)
                    else:
                        logger.error(f"Invalid plot bytes for coordinates: {coordinates}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing coordinates {coordinates}: {str(e)}")
                    continue
            
            # Create a zip file of all plots
            zip_path = os.path.join(temp_dir, 'contribution_scores_batch.zip')
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.png'):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.basename(file_path))
            
            # Read the zip file
            with open(zip_path, 'rb') as f:
                zip_bytes = f.read()
            
            return html.Div([
                html.P("Batch processing completed successfully!"),
                html.P(f"Processed {len(df)} coordinates"),
                html.P("Download will start automatically...")
            ]), dcc.send_bytes(zip_bytes, "contribution_scores_batch.zip")
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return html.Div(f"Error in batch processing: {str(e)}", style={'color': 'red'}), dash.no_update

@callback(
    Output('class-dropdown', 'style'),
    [Input('class-dropdown', 'value')]
)
def update_class_dropdown_style(selected_values):
    if not selected_values:
        return {'width': '100%', 'marginBottom': '20px'}
    count = len(selected_values)
    return {
        'width': '100%',
        'marginBottom': '20px',
        '--select-count': f'"{count} selected"'
    }

@callback(
    Output('subclass-dropdown', 'style'),
    [Input('subclass-dropdown', 'value')]
)
def update_subclass_dropdown_style(selected_values):
    if not selected_values:
        return {'width': '100%', 'marginBottom': '20px'}
    count = len(selected_values)
    return {
        'width': '100%',
        'marginBottom': '20px',
        '--select-count': f'"{count} selected"'
    }

@callback(
    Output('supertype-dropdown', 'style'),
    [Input('supertype-dropdown', 'value')]
)
def update_supertype_dropdown_style(selected_values):
    if not selected_values:
        return {'width': '100%', 'marginBottom': '20px'}
    count = len(selected_values)
    return {
        'width': '100%',
        'marginBottom': '20px',
        '--select-count': f'"{count} selected"'
    }

@callback(
    [Output('subclass-dropdown', 'options'),
     Output('subclass-dropdown', 'value')],
    [Input('subclass-class-filter', 'value')]
)
def update_subclass_options(selected_class):
    logger.info(f"Selected class for subclass filter: {selected_class}")
    
    # Always show all available subclasses in the options
    options = [
        {'label': subclass, 'value': subclass_bigwigs[subclass]['url']}
        for subclass in sorted(subclass_bigwigs.keys())
    ]
    
    if not selected_class:
        # If no class is selected, don't pre-select any subclasses
        logger.info(f"No class selected, showing all subclasses without pre-selection")
        return options, []
    
    # Pre-select subclasses for the selected class
    available_subclasses = class_to_subclass.get(selected_class, [])
    logger.info(f"Available subclasses for {selected_class}: {available_subclasses}")
    
    if not available_subclasses:
        logger.warning(f"No subclasses found for class: {selected_class}")
        return options, []
    
    # Pre-select the subclasses for the selected class, ensuring no duplicates
    selected_values = []
    seen_values = set()
    for subclass in sorted(available_subclasses):
        if subclass in subclass_bigwigs:
            value = subclass_bigwigs[subclass]['url']
            if value not in seen_values:
                selected_values.append(value)
                seen_values.add(value)
    
    logger.info(f"Pre-selected {len(selected_values)} unique subclasses")
    logger.info(f"Selected values: {selected_values}")
    return options, selected_values

@callback(
    [Output('supertype-subclass-filter', 'options'),
     Output('supertype-dropdown', 'options'),
     Output('supertype-dropdown', 'value')],
    [Input('subclass-class-filter', 'value'),
     Input('supertype-subclass-filter', 'value')]
)
def update_supertype_options(selected_class, selected_subclass):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"Supertype callback triggered by: {trigger_id}")
    
    # Always show all available supertypes in the options
    supertype_options = [
        {'label': supertype, 'value': supertype_bigwigs[supertype]['url']}
        for supertype in sorted(supertype_bigwigs.keys())
    ]
    
    if trigger_id == 'subclass-class-filter':
        logger.info(f"Updating supertype options based on class: {selected_class}")
        if not selected_class:
            # If no class is selected, show all classes in the filter but don't select any
            class_options = [
                {'label': class_name, 'value': class_name}
                for class_name in sorted(class_bigwigs.keys())
            ]
            return class_options, supertype_options, []
        else:
            # Show only subclasses for the selected class in the filter
            available_subclasses = class_to_subclass.get(selected_class, [])
            subclass_options = [
                {'label': subclass, 'value': subclass}
                for subclass in sorted(available_subclasses)
            ]
            # Don't pre-select any supertypes when class is selected
            return subclass_options, supertype_options, []
    
    else:  # trigger_id == 'supertype-subclass-filter'
        logger.info(f"Updating supertype options based on subclass: {selected_subclass}")
        if not selected_subclass:
            # If no subclass is selected, don't pre-select any supertypes
            return dash.no_update, supertype_options, []
        else:
            # Pre-select supertypes for the selected subclass
            available_supertypes = subclass_to_supertype.get(selected_subclass, [])
            
            # Ensure no duplicate selections
            selected_values = []
            seen_values = set()
            for supertype in sorted(available_supertypes):
                if supertype in supertype_bigwigs:
                    value = supertype_bigwigs[supertype]['url']
                    if value not in seen_values:
                        selected_values.append(value)
                        seen_values.add(value)
            
            logger.info(f"Pre-selected {len(selected_values)} unique supertypes")
            return dash.no_update, supertype_options, selected_values

@callback(
    Output('class-dropdown', 'options'),
    [Input('class-dropdown', 'value')]
)
def update_class_options(selected_values):
    # Create options for all classes
    options = [
        {'label': class_name, 'value': class_bigwigs[class_name]['url']}
        for class_name in sorted(class_bigwigs.keys())
    ]
    return options

if __name__ == '__main__':
    app.run(debug=True)
