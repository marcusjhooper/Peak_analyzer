import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_cell_type_colors(color_file='data/other/cell_type_colors.csv'):
    """Load cell type colors into a dictionary."""
    cell_type_colors_df = pd.read_csv(color_file, sep='\t')
    return dict(zip(cell_type_colors_df['cell_type'].tolist(), cell_type_colors_df['color']))

def scan_for_bigwigs(root_dir='data'):
    """Scan the directory for bigwig files and organize them by folder."""
    bigwig_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.bw'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                folder = os.path.dirname(rel_path)
                # Extract the category from the path
                category = folder.split('/')[0] if '/' in folder else folder
                # Get cell type name (filename without .bw)
                cell_type = file.replace('.bw', '')
                bigwig_files.append({
                    'path': full_path,
                    'name': file,
                    'folder': folder if folder else 'root',
                    'category': category,
                    'cell_type': cell_type
                })
    
    return pd.DataFrame(bigwig_files)

def scan_for_peak_tables():
    """Scan the data/differential_peaks directory for available peak tables."""
    try:
        peaks_dir = Path('data/differential_peaks')
        if not peaks_dir.exists():
            logger.warning("Differential peaks directory not found")
            return []
        
        # Find all feather files in the peaks directory
        peak_files = list(peaks_dir.glob('*.feather'))
        if not peak_files:
            logger.warning("No peak tables found in data/differential_peaks")
            return []
        
        # Create options for the dropdown
        options = [{'label': f.stem, 'value': str(f)} for f in peak_files]
        logger.info(f"Found {len(options)} peak tables")
        return options
    except Exception as e:
        logger.error(f"Error scanning for peak tables: {str(e)}")
        return []

def load_differential_peaks(table_path=None):
    """Load differential peaks from a feather file."""
    try:
        # If no specific table is provided, use the default
        if table_path is None:
            table_path = 'data/differential_peaks/differential_peaks.feather'
        
        # Check if the file exists
        if not os.path.exists(table_path):
            logger.warning(f"Peak table not found at {table_path}")
            return pd.DataFrame(), []
        
        # Load the feather file
        df = pd.read_feather(table_path)
        
        # Define all possible columns we might want to display
        possible_columns = {
            'coordinates': 'Genomic Coordinates',
            'neighborhood': 'Neighborhood',
            'class_label': 'Class Label',
            'cell_type': 'Cell Type',
            'supertype_label': 'Supertype Label',
            'adjusted_p_value': 'Adjusted P-value',
            'gini_supertype_label_in_neighborhood': 'Gini Supertype Label in Neighborhood',
            'gini_subclass_id_label_in_class':'Gini_subclass_label_in_class',
            'log2(fold_change)': 'Log2 Fold Change',
            # Additional columns that might be present
            'peak_name': 'Peak Name',
            'gene_name': 'Gene Name',
            'distance': 'Distance to Gene',
            'annotation': 'Annotation',
            'pvalue': 'P-value',
            'pval_adj': 'Adjusted P-value',
            'log2(FC)': 'Log2 Fold Change'
        }
        
        # Find which columns are actually present in the data
        available_columns = {}
        missing_columns = []
        
        for col, name in possible_columns.items():
            # Check for exact match
            if col in df.columns:
                available_columns[col] = name
            # Check for case-insensitive match
            elif any(col.lower() == c.lower() for c in df.columns):
                matching_col = next(c for c in df.columns if col.lower() == c.lower())
                available_columns[matching_col] = name
            # Check for variations in log2(fold_change)
            elif col == 'log2(fold_change)' and 'log2(FC)' in df.columns:
                available_columns['log2FoldChange'] = name
            # Check for variations in adjusted_p_value
            elif col == 'adjusted_p_value' and 'pval_adj' in df.columns:
                available_columns['padj'] = name
            else:
                missing_columns.append(col)
        
        # Log which columns were found and which were not
        if missing_columns:
            logger.warning(f"Could not find the following columns in {table_path}: {missing_columns}")
            logger.info(f"Available columns in the table: {list(df.columns)}")
        
        # Create column definitions for the DataTable
        columns = []
        for col, name in available_columns.items():
            column_def = {
                "name": name,
                "id": col,
                "type": "numeric" if df[col].dtype in ['float64', 'int64'] else "text",
            }
            if column_def["type"] == "numeric":
                column_def["format"] = {"specifier": ".2f"}
            columns.append(column_def)
        
        # If no columns were found, try to use all available columns with their original names
        if not columns:
            logger.warning("No matching columns found, using all available columns")
            for col in df.columns:
                column_def = {
                    "name": col,
                    "id": col,
                    "type": "numeric" if df[col].dtype in ['float64', 'int64'] else "text",
                }
                if column_def["type"] == "numeric":
                    column_def["format"] = {"specifier": ".2f"}
                columns.append(column_def)
        
        logger.info(f"Loaded peak table with {len(columns)} columns")
        return df, columns
    except Exception as e:
        logger.error(f"Error loading differential peaks: {str(e)}")
        return pd.DataFrame(), []

def load_model_settings(model_path):
    """Load model-specific settings from CSV file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Dictionary containing model settings
    """
    try:
        # Check if model settings file exists
        settings_path = 'data/model/model_settings.csv'
        if not os.path.exists(settings_path):
            logger.warning(f"Model settings file not found at {settings_path}, using default settings")
            return {
                'sequence_length': 1500,
                'batch_size': 128,
                'zoom_n_bases': 500
            }
        
        # Read the settings file
        settings_df = pd.read_csv(settings_path)
        
        # Extract model name from path (remove .keras extension if present)
        model_name = os.path.basename(model_path)
        if model_name.endswith('.keras'):
            model_name = model_name[:-6]  # Remove .keras extension
        
        logger.info(f"Looking for settings for model: {model_name}")
        
        # Find settings for this model
        model_settings = settings_df[settings_df['model_name'] == model_name]
        
        if model_settings.empty:
            logger.warning(f"No settings found for model {model_name}, using default settings")
            model_settings = settings_df[settings_df['model_name'] == 'default']
            if model_settings.empty:
                raise ValueError("No default settings found in model_settings.csv")
        
        # Convert to dictionary
        settings = model_settings.iloc[0].to_dict()
        
        # Convert numeric values to integers
        for key in ['sequence_length', 'batch_size', 'zoom_n_bases']:
            if key in settings:
                settings[key] = int(settings[key])
        
        # Parse indexed_class_labels if present
        if 'indexed_class_labels' in settings and isinstance(settings['indexed_class_labels'], str):
            try:
                # Remove any quotes and split by comma
                labels_str = settings['indexed_class_labels'].strip("'")
                # Split by comma and strip whitespace and quotes from each label
                class_labels = [label.strip().strip("'") for label in labels_str.split(',')]
                settings['indexed_class_labels'] = class_labels
                logger.info(f"Parsed class labels: {class_labels}")
            except Exception as e:
                logger.error(f"Error parsing indexed_class_labels: {str(e)}")
                raise
        
        logger.info(f"Loaded settings for model {model_name}: {settings}")
        return settings
        
    except Exception as e:
        logger.error(f"Error loading model settings: {str(e)}")
        raise 