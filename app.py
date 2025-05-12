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
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive plotting

# Import our modules
from utils import parse_coordinates, get_sequence, format_sequence_with_line_numbers, highlight_motif, calculate_gc_content, reverse_complement
from data_loading import (
    load_cell_type_colors, 
    scan_for_bigwigs, 
    load_differential_peaks, 
    load_model_settings,
    scan_for_peak_tables
)
from plotting import create_contribution_scores_plot, create_plot

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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
        <style>
            * {
                font-family: Arial, sans-serif !important;
            }
            pre, code {
                font-family: monospace !important;
            }
            .dropdown-compact .Select-control {
                height: 30px !important;
                min-height: 30px !important;
                border-radius: 4px !important;
                overflow: hidden !important;
            }
            .dropdown-compact .Select-value {
                line-height: 30px !important;
                padding: 0 8px !important;
            }
            .dropdown-compact .Select-placeholder {
                line-height: 30px !important;
                padding: 0 8px !important;
            }
            .dropdown-compact .Select-input {
                height: 28px !important;
                padding: 0 8px !important;
            }
            .dropdown-compact .Select-menu-outer {
                max-height: 200px !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            .dropdown-compact.is-focused .Select-control {
                height: auto !important;
                min-height: auto !important;
                border-color: #80bdff !important;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
            }
            .dropdown-compact .Select--multi .Select-value {
                background-color: #e9ecef !important;
                border-radius: 3px !important;
                margin: 2px !important;
                padding: 0 6px !important;
                max-width: 150px !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                white-space: nowrap !important;
            }
            .dropdown-compact .Select--multi .Select-value-icon {
                border-right: 1px solid #ced4da !important;
                padding: 0 4px !important;
            }
            .dropdown-compact .Select--multi .Select-value-label {
                padding: 0 4px !important;
            }
            .dropdown-compact .Select-arrow {
                top: 8px !important;
            }
            .dropdown-compact .Select-clear {
                top: 8px !important;
            }
            /* Add a count indicator */
            .dropdown-compact .Select-multi-value-wrapper {
                position: relative !important;
                padding-right: 60px !important;
            }
            .dropdown-compact .Select-multi-value-wrapper::after {
                content: attr(data-count) !important;
                position: absolute !important;
                right: 8px !important;
                top: 50% !important;
                transform: translateY(-50%) !important;
                background-color: #e9ecef !important;
                padding: 2px 6px !important;
                border-radius: 3px !important;
                font-size: 12px !important;
                color: #495057 !important;
            }
            /* Hide all but the first selected value when not focused */
            .dropdown-compact:not(.is-focused) .Select--multi .Select-value:not(:first-child) {
                display: none !important;
            }
            /* Show count when not focused */
            .dropdown-compact:not(.is-focused) .Select-multi-value-wrapper::after {
                content: attr(data-count) !important;
            }
            /* Hide count when focused */
            .dropdown-compact.is-focused .Select-multi-value-wrapper::after {
                display: none !important;
            }
        </style>
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
cell_type_colors = load_cell_type_colors()
diff_peaks_df, diff_peaks_columns = load_differential_peaks()
bigwig_df = scan_for_bigwigs()
peak_table_options = scan_for_peak_tables()

# Load model settings for dropdown
model_settings_df = pd.read_csv('data/model/model_settings.csv')
model_options = [{'label': row['model_name'], 'value': row['model_file']} 
                for _, row in model_settings_df.iterrows()]

# Add CRESTED model state
crested_model = None

# Set default font to Arial
app.layout = html.Div([
    html.H1('Genomic Data Viewer', style={'fontFamily': 'Arial, sans-serif'}),
    
    # Tabs
    dcc.Tabs([
        # Genomic Viewer Tab (Differential Peaks and BigWig Viewer)
        dcc.Tab(label='Genomic Viewer', children=[
            # Differential Peaks Table
            html.Div([
                html.H2('Differential Peaks', style={'fontFamily': 'Arial, sans-serif'}),
                html.Div([
                    html.Label('Select Peak Table:', style={'fontFamily': 'Arial, sans-serif'}),
                    dcc.Dropdown(
                        id='peak-table-dropdown',
                        options=peak_table_options,
                        value=None,
                        style={'width': '300px', 'marginBottom': '20px', 'fontFamily': 'Arial, sans-serif'}
                    ),
                    DataTable(
                        id='peaks-table',
                        columns=diff_peaks_columns,
                        data=diff_peaks_df.to_dict('records'),
                        page_size=10,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="single",
                        style_table={
                            'width': '100%',
                            'minWidth': '100%',
                            'overflowX': 'auto'
                        },
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
                ], style={'marginBottom': '30px', 'width': '100%'}),
                
                # BigWig Viewer Controls
                html.Div([
                    html.Label('Genomic Coordinates (e.g., chr13:113379626-113380127):', 
                              style={'fontFamily': 'Arial, sans-serif'}),
                    dcc.Input(
                        id='coordinates-input',
                        value='chr13:113379626-113380127',
                        type='text',
                        style={'width': '300px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}
                    ),
                    html.Button('Update Plot', id='update-button', n_clicks=0, 
                               style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    html.Button('Zoom Out 1kb', id='zoom-out-1kb', n_clicks=0, 
                               style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    html.Button('Zoom Out 10kb', id='zoom-out-10kb', n_clicks=0, 
                               style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    html.Button('Zoom In 1kb', id='zoom-in-1kb', n_clicks=0, 
                               style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    html.Button('Zoom In 10kb', id='zoom-in-10kb', n_clicks=0, 
                               style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    dcc.Checklist(
                        id='default-zoom',
                        options=[{'label': 'Default zoom out 1kb', 'value': 'zoom_out'}],
                        value=['zoom_out'],
                        style={'display': 'inline-block', 'marginLeft': '10px', 'fontFamily': 'Arial, sans-serif'}
                    )
                ], style={'marginBottom': '20px'}),
                
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
                                options=[
                                    {'label': f"{row['folder']}/{row['name']}", 'value': row['path']}
                                    for _, row in bigwig_df[bigwig_df['category'] == 'class'].iterrows()
                                ],
                                value=[row['path'] for _, row in bigwig_df[bigwig_df['category'] == 'class'].iterrows()],
                                multi=True,
                                style={'width': '100%', 'marginBottom': '20px'},
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select files...',
                                optionHeight=20,
                                maxHeight=60
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Subclass column
                        html.Div([
                            html.H3('Subclass', style={'marginBottom': '20px'}),
                            dcc.Dropdown(
                                id='subclass-dropdown',
                                options=[
                                    {'label': f"{row['folder']}/{row['name']}", 'value': row['path']}
                                    for _, row in bigwig_df[bigwig_df['category'] == 'subclass'].iterrows()
                                ],
                                value=[row['path'] for _, row in bigwig_df[bigwig_df['category'] == 'subclass'].iterrows()],
                                multi=True,
                                style={'width': '100%', 'marginBottom': '20px'},
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select files...',
                                optionHeight=20,
                                maxHeight=60
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Supertype column
                        html.Div([
                            html.H3('Supertype', style={'marginBottom': '20px'}),
                            dcc.Dropdown(
                                id='supertype-dropdown',
                                options=[
                                    {'label': f"{row['folder']}/{row['name']}", 'value': row['path']}
                                    for _, row in bigwig_df[bigwig_df['category'] == 'supertype'].iterrows()
                                ],
                                value=[row['path'] for _, row in bigwig_df[bigwig_df['category'] == 'supertype'].iterrows()],
                                multi=True,
                                style={'width': '100%', 'marginBottom': '20px'},
                                className='dropdown-compact',
                                searchable=True,
                                clearable=True,
                                placeholder='Select files...',
                                optionHeight=20,
                                maxHeight=60
                            ),
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'})
                    ], style={'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'space-between'}),
                    
                    # Plots row
                    html.Div([
                        # Class plot
                        html.Div([
                            html.Img(id='class-plot', style={'width': '100%', 'objectFit': 'contain'})
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box', 'minHeight': '800px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'}),
                        
                        # Subclass plot
                        html.Div([
                            html.Img(id='subclass-plot', style={'width': '100%', 'objectFit': 'contain'})
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box', 'minHeight': '800px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'}),
                        
                        # Supertype plot
                        html.Div([
                            html.Img(id='supertype-plot', style={'width': '100%', 'objectFit': 'contain'})
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box', 'minHeight': '800px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between'})
                ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ], style={'padding': '20px', 'overflowY': 'auto'})
        ]),
        
        # Sequence Analysis Tab
        dcc.Tab(label='Sequence Analysis', children=[
            html.Div([
                html.H2('Sequence Viewer and Analysis', style={'fontFamily': 'Arial, sans-serif'}),
                
                # Sequence Input Section
                html.Div([
                    html.H3('Sequence Input', style={'fontFamily': 'Arial, sans-serif'}),
                    html.Div([
                        # Genomic coordinates input
                        html.Div([
                            html.Label('Genomic Coordinates (e.g., chr13:113379626-113380127):', 
                                      style={'fontFamily': 'Arial, sans-serif'}),
                            dcc.Input(
                                id='sequence-coordinates-input',
                                value='chr13:113379626-113380127',
                                type='text',
                                style={'width': '300px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}
                            ),
                            html.Button('Get Sequence', id='get-sequence-button', n_clicks=0, 
                                       style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Custom sequence input
                        html.Div([
                            html.Label('Or Enter Custom Sequence:', 
                                      style={'fontFamily': 'Arial, sans-serif'}),
                            dcc.Textarea(
                                id='custom-sequence-input',
                                placeholder='Enter DNA sequence...',
                                style={'width': '300px', 'height': '30px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}
                            ),
                            html.Button('Use Custom Sequence', id='use-custom-sequence-button', n_clicks=0,
                                      style={'fontFamily': 'Arial, sans-serif'}),
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
                            style={'marginLeft': '10px', 'fontFamily': 'Arial, sans-serif'}
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # Motif search input
                    html.Div([
                        html.Label('Search for motif:', 
                                  style={'marginRight': '5px', 'fontFamily': 'Arial, sans-serif'}),
                        dcc.Input(id='motif-input', type='text', placeholder='e.g., TATAAA', 
                                 style={'width': '120px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div(id='sequence-output', style={
                        'marginTop': '10px',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'whiteSpace': 'pre-wrap',
                        'fontFamily': 'Arial, sans-serif',
                        'overflowX': 'auto'
                    })
                ], style={'marginBottom': '30px'}),
                
                # CRESTED Model Section
                html.Div([
                    html.H3('CRESTED Model', style={'fontFamily': 'Arial, sans-serif'}),
                    html.Div([
                        # Model selection dropdown
                        html.Div([
                            html.Label('Select Model:', style={'fontFamily': 'Arial, sans-serif'}),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=model_options,
                                value=None,
                                style={'width': '300px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Custom model input
                        html.Div([
                            html.Label('Or Load Custom Model:', style={'fontFamily': 'Arial, sans-serif'}),
                            dcc.Input(
                                id='custom-model-path',
                                type='text',
                                placeholder='Enter path to custom model',
                                style={'width': '300px', 'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}
                            ),
                        ], style={'display': 'inline-block'}),
                        
                        html.Button('Load Model', id='load-model-button', n_clicks=0,
                                  style={'marginRight': '10px', 'fontFamily': 'Arial, sans-serif'}),
                    ], style={'marginBottom': '20px'}),
                    
                    # Class selection dropdown
                    html.Div([
                        html.Div([
                            html.Label('Select Classes to Analyze:', style={'fontFamily': 'Arial, sans-serif'}),
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
                            style={'width': '100%', 'marginBottom': '20px', 'fontFamily': 'Arial, sans-serif'}
                        ),
                    ]),
                    
                    html.Button('Run Scores', id='run-scores-button', n_clicks=0,
                              style={'fontFamily': 'Arial, sans-serif'}),
                    
                    html.Div(id='model-status', style={'marginBottom': '20px'}),
                    html.Div(id='contribution-scores-plot', style={'marginBottom': '20px'}),
                    
                    # Download button and component
                    html.Div([
                        html.Button('Download Plot', id='download-plot-button', n_clicks=0,
                                  style={'marginTop': '10px', 'fontFamily': 'Arial, sans-serif'}),
                        dcc.Download(id='plot-download')
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),
                    
                    # Batch processing section
                    html.Div([
                        html.H3('Batch Processing', style={'marginTop': '30px', 'fontFamily': 'Arial, sans-serif'}),
                        html.P('Upload a CSV file containing coordinates to process in batch. The CSV should have a column named "coordinates" with values in the format "chr:start-end".'),
                        dcc.Upload(
                            id='batch-coordinates-upload',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select CSV File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'fontFamily': 'Arial, sans-serif'
                            },
                            multiple=False
                        ),
                        html.Div(id='batch-upload-status', style={'marginBottom': '10px'}),
                        html.Button('Run Scores Batch', id='run-batch-button', n_clicks=0,
                                  style={'fontFamily': 'Arial, sans-serif'}),
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
        
        # Load model settings
        model_settings = load_model_settings(model_path)
        
        # Load the model using keras with custom objects
        crested_model = keras.models.load_model(model_path, custom_objects=custom_objects)
        
        # Store model settings in the model object
        crested_model.settings = model_settings
        
        return html.Div([
            html.P("Model loaded successfully!", style={'color': 'green'}),
            html.P(f"Model path: {model_path}"),
            html.P(f"Model settings: {model_settings}")
        ]), False
    except Exception as e:
        return html.Div([
            html.P("Error loading model:", style={'color': 'red'}),
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
    Output('contribution-scores-plot', 'children'),
    [Input('run-scores-button', 'n_clicks')],
    [State('sequence-coordinates-input', 'value'),
     State('class-selection-dropdown', 'value')]
)
def run_contribution_scores(n_clicks, coordinates, selected_classes):
    if n_clicks == 0 or not crested_model:
        return "No model loaded or scores not run yet"
    
    try:
        logger.info("Starting contribution scores calculation...")
        
        # Get model settings
        model_settings = getattr(crested_model, 'settings', {
            'sequence_length': 1500,
            'batch_size': 128,
            'zoom_n_bases': 500
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
        genome_path = 'data/genome/mm10.fa'
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
            sequence = genome[chrom][new_start:new_end]
            seq_str = str(sequence.seq).upper()
            
            # Validate sequence
            if not seq_str:
                raise ValueError("Empty sequence returned")
            
            # Log sequence details
            logger.info(f"Retrieved sequence of length {len(seq_str)}")
            logger.debug(f"First 10 bases: {seq_str[:10]}")
            logger.debug(f"Last 10 bases: {seq_str[-10:]}")
            
        except Exception as e:
            logger.error(f"Error getting sequence: {str(e)}")
            raise ValueError(f"Failed to get valid sequence: {str(e)}")
        
        logger.info(f"Got sequence of length {len(seq_str)} (centered at {center})")
        logger.info(f"Original coordinates: {start}-{end}")
        logger.info(f"Adjusted coordinates: {new_start}-{new_end}")
        
        # Run contribution scores
        logger.info("Calculating contribution scores...")
        try:
            # Get the number of classes from the model's output shape
            n_classes = crested_model.output_shape[1]
            logger.info(f"Model has {n_classes} classes")
            
            # Use selected classes if provided, otherwise use all classes
            if selected_classes is not None and len(selected_classes) > 0:
                target_idx = selected_classes
                logger.info(f"Calculating contribution scores for selected classes: {target_idx}")
            else:
                target_idx = list(range(n_classes))
                logger.info(f"No classes selected, calculating for all classes: {target_idx}")
            
            scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
                input=seq_str,
                target_idx=target_idx,
                genome=genome_path,
                model=crested_model,
                batch_size=model_settings['batch_size']
            )
            logger.info("Successfully calculated contribution scores")
        except Exception as e:
            logger.error(f"Error in contribution_scores calculation: {str(e)}")
            raise
        
        # Get class labels from model settings
        class_labels = model_settings.get('indexed_class_labels', [f'Class {i+1}' for i in range(n_classes)])
        # Filter class labels to only show selected ones
        if selected_classes is not None and len(selected_classes) > 0:
            class_labels = [class_labels[i] for i in selected_classes]
        logger.info(f"Using class labels: {class_labels}")
        
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
            zoom_n_bases=model_settings['zoom_n_bases']
        )
        
        # Store the plot bytes globally
        global current_plot_bytes
        current_plot_bytes = plot_bytes
        
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
            html.P(f"Model settings: {display_settings}"),
            html.Img(src=base64_data, style={'width': '100%'})
        ])
    except Exception as e:
        error_msg = f"Error running contribution scores: {str(e)}"
        logger.error(error_msg)
        return html.Div([
            html.P("Error running contribution scores:", style={'color': 'red'}),
            html.P(str(e))
        ])

@callback(
    Output('plot-download', 'data'),
    [Input('download-plot-button', 'n_clicks')],
    [State('sequence-coordinates-input', 'value'),
     State('custom-sequence-input', 'value')]
)
def download_plot(n_clicks, coordinates, custom_sequence):
    if n_clicks == 0 or not current_plot_bytes:
        return dash.no_update
    
    try:
        # Create filename based on whether custom sequence is used
        if custom_sequence and custom_sequence.strip():
            filename = "contribution_scores_plot.png"
        else:
            # Clean up coordinates for filename
            clean_coords = coordinates.replace(':', '_').replace('-', '_')
            filename = f"contribution_scores_plot_{clean_coords}.png"
        
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
     Output('peaks-table', 'selected_rows')],
    [Input('peaks-table', 'selected_cells'),
     Input('update-button', 'n_clicks'),
     Input('zoom-out-1kb', 'n_clicks'),
     Input('zoom-out-10kb', 'n_clicks'),
     Input('zoom-in-1kb', 'n_clicks'),
     Input('zoom-in-10kb', 'n_clicks'),
     Input('default-zoom', 'value'),
     Input('class-dropdown', 'value'),
     Input('subclass-dropdown', 'value'),
     Input('supertype-dropdown', 'value')],
    [State('peaks-table', 'data'),
     State('peaks-table', 'derived_virtual_data'),
     State('peaks-table', 'derived_virtual_selected_rows'),
     State('peaks-table', 'page_current'),
     State('peaks-table', 'page_size'),
     State('coordinates-input', 'value')]
)
def update_coordinates_and_plots(selected_cells, update_clicks, zoom_out_1kb, zoom_out_10kb, zoom_in_1kb, zoom_in_10kb, 
                               default_zoom, class_files, subclass_files, supertype_files, table_data, derived_virtual_data, 
                               derived_virtual_selected_rows, page_current, page_size, current_coords):
    # Get the context to determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.debug(f"Triggered by: {trigger_id}")
    
    # Initialize coordinates and zoom level
    new_coords = current_coords
    selected_row = None
    zoom_level = 0
    
    # Apply default zoom if checkbox is checked
    if 'zoom_out' in default_zoom and trigger_id in ['peaks-table', 'update-button']:
        zoom_level = -1000
    
    # Handle zoom controls
    if trigger_id == 'zoom-out-1kb':
        zoom_level = -1000
    elif trigger_id == 'zoom-out-10kb':
        zoom_level = -10000
    elif trigger_id == 'zoom-in-1kb':
        zoom_level = 1000
    elif trigger_id == 'zoom-in-10kb':
        zoom_level = 10000
    
    # If triggered by cell selection, update coordinates and row selection
    if trigger_id == 'peaks-table' and selected_cells:
        # Get the first selected cell
        cell = selected_cells[0]
        row_idx = cell['row']
        
        # Determine which data to use based on whether the table is filtered
        if derived_virtual_data:
            # Table is filtered, use derived_virtual_data
            data_to_use = derived_virtual_data
            # Calculate the actual row index in the filtered view
            actual_row_idx = row_idx + (page_current * page_size)
            logger.debug("Using filtered data (derived_virtual_data)")
        else:
            # Table is not filtered, use table_data
            data_to_use = table_data
            # Calculate the actual row index in the full dataset
            actual_row_idx = row_idx + (page_current * page_size)
            logger.debug("Using unfiltered data (table_data)")
        
        # Get the coordinates from the 'coordinates' column of the selected row
        if actual_row_idx < len(data_to_use):
            row_data = data_to_use[actual_row_idx]
            if 'coordinates' in row_data:
                coord_value = row_data['coordinates']
                logger.debug(f"Coordinates from row: {coord_value}")
                logger.debug(f"Page current: {page_current}, Page size: {page_size}")
                logger.debug(f"Row index in current view: {row_idx}, Actual row index: {actual_row_idx}")
                logger.debug(f"Data length: {len(data_to_use)}")
                
                if isinstance(coord_value, str) and re.match(r'[^:]+:\d+-\d+', coord_value):
                    new_coords = coord_value
                    logger.debug(f"Found valid coordinates: {new_coords}")
                    # Set the selected row to the current row index in the current view
                    selected_row = [row_idx]
                else:
                    logger.warning(f"Invalid coordinate format: {coord_value}")
            else:
                logger.warning("No coordinates found in row data")
        else:
            logger.warning(f"Row index {actual_row_idx} out of range for data length {len(data_to_use)}")
    
    # Store the original coordinates before applying zoom
    original_coords = new_coords
    
    # Create plots for each category
    class_plot = create_plot(class_files, new_coords, zoom_level, original_coords, cell_type_colors)
    subclass_plot = create_plot(subclass_files, new_coords, zoom_level, original_coords, cell_type_colors)
    supertype_plot = create_plot(supertype_files, new_coords, zoom_level, original_coords, cell_type_colors)
    
    # Log the final coordinates being used
    logger.debug(f"Final coordinates: {new_coords}")
    logger.debug(f"Original coordinates: {original_coords}")
    logger.debug(f"Selected row index: {selected_row}")
    
    return new_coords, new_coords, class_plot, subclass_plot, supertype_plot, selected_row if selected_row else dash.no_update

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

@callback(
    Output('sequence-output', 'children'),
    [Input('get-sequence-button', 'n_clicks'),
     Input('use-custom-sequence-button', 'n_clicks'),
     Input('motif-input', 'value')],
    [State('sequence-coordinates-input', 'value'),
     State('custom-sequence-input', 'value'),
     State('sequence-options', 'value')]
)
def get_sequence(get_clicks, custom_clicks, motif, coordinates, custom_sequence, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Click a button to fetch sequence"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if trigger_id == 'get-sequence-button':
            # Parse coordinates
            chrom, start, end = parse_coordinates(coordinates)
            
            # Load mm10 genome from gzipped file
            genome = Fasta('data/genome/mm10.fa')
            
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
            logger.info(f"Created temporary directory: {temp_dir}")
            
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
            logger.info(f"Using class labels: {class_labels}")
            
            # Process each coordinate
            for i, row in df.iterrows():
                coordinates = row['coordinates']
                try:
                    logger.info(f"Processing coordinates: {coordinates}")
                    
                    # Get the sequence
                    chrom, start, end = parse_coordinates(coordinates)
                    
                    # Calculate center position
                    center = (start + end) // 2
                    half_length = model_settings['sequence_length'] // 2
                    new_start = center - half_length
                    new_end = center + half_length
                    
                    # Get the sequence
                    genome = Fasta('data/genome/mm10.fa')
                    sequence = genome[chrom][new_start:new_end]
                    seq_str = str(sequence.seq).upper()
                    
                    # Run contribution scores
                    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
                        input=seq_str,
                        target_idx=selected_classes if selected_classes else list(range(crested_model.output_shape[1])),
                        genome='data/genome/mm10.fa',
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
                        zoom_n_bases=model_settings['zoom_n_bases']
                    )
                    
                    # Save the plot
                    clean_coords = coordinates.replace(':', '_').replace('-', '_')
                    plot_path = os.path.join(temp_dir, f'contribution_scores_{clean_coords}.png')
                    logger.info(f"Saving plot to: {plot_path}")
                    
                    # Ensure the plot bytes are valid before saving
                    if plot_bytes and len(plot_bytes) > 0:
                        with open(plot_path, 'wb') as f:
                            f.write(plot_bytes)
                        logger.info(f"Successfully saved plot for coordinates: {coordinates}")
                        logger.info(f"Plot file size: {os.path.getsize(plot_path)} bytes")
                    else:
                        logger.error(f"Invalid plot bytes for coordinates: {coordinates}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing coordinates {coordinates}: {str(e)}")
                    continue
            
            # Create a zip file of all plots
            zip_path = os.path.join(temp_dir, 'contribution_scores_batch.zip')
            logger.info(f"Creating zip file at: {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.png'):
                            file_path = os.path.join(root, file)
                            logger.info(f"Adding file to zip: {file_path}")
                            zipf.write(file_path, os.path.basename(file_path))
            
            # Read the zip file
            with open(zip_path, 'rb') as f:
                zip_bytes = f.read()
            logger.info(f"Zip file size: {len(zip_bytes)} bytes")
            
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

if __name__ == '__main__':
    app.run(debug=True)
