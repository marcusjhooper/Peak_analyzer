import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_cell_type_colors(color_file):
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

def scan_for_peak_tables(data_dir=None):
    """Scan the data/differential_peaks directory for available peak tables."""
    try:
        if data_dir is not None:
            peaks_dir = Path(os.path.join(data_dir, 'differential_peaks'))
        else:
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

def load_model_settings(model_path, data_dir):
    """Load model settings from model_settings.csv."""
    try:
        # Extract model name from path
        model_name = os.path.basename(model_path).replace('.keras', '')
        logger.info(f"Looking for settings for model: {model_name}")
        
        # Load model settings
        settings_path = os.path.join(data_dir, 'model/model_settings.csv')
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Model settings file not found at {settings_path}")
        
        settings_df = pd.read_csv(settings_path)
        
        # First try to find settings by model name
        model_settings = settings_df[settings_df['model_name'] == model_name]
        
        # If not found, try to find settings by model file
        if model_settings.empty:
            model_settings = settings_df[settings_df['model_file'].str.contains(model_name)]
        
        #not implemented yet
        #if model_settings.empty:
            #logger.warning(f"No settings found for model {model_name}, using default settings")
            # Use default settings based on model name

        
        # Convert settings to dictionary
        settings = model_settings.iloc[0].to_dict()
        logger.info(settings.keys())
        logger.info(settings.values())
        # Convert string representation of list to actual list for indexed_class_labels
        if isinstance(settings['indexed_class_labels'], str):
            # Remove any leading/trailing quotes and split by comma
            labels_str = settings['indexed_class_labels'].strip("'")
            # Split by comma and clean up each label
            class_labels = [label.strip().strip("'") for label in labels_str.split(',')]
            settings['indexed_class_labels'] = class_labels
        
        return settings
    except Exception as e:
        logger.error(f"Error loading model settings: {str(e)}")
        raise

def load_gene_annotations(data_dir=None):
    """Load gene annotations from BED file."""
    try:
        if data_dir is not None:
            gene_file = os.path.join(data_dir, 'annotation/mm10_gene_regions.bed')
        else:
            gene_file = 'data/annotation/mm10_gene_regions.bed'
        
        if not os.path.exists(gene_file):
            logger.warning(f"Gene annotation file not found at {gene_file}")
            return pd.DataFrame()
        
        # Load BED file
        genes = pd.read_csv(gene_file, sep='\t', header=None,
                           names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
        
        logger.info(f"Loaded {len(genes)} gene annotations")
        return genes
        
    except Exception as e:
        logger.error(f"Error loading gene annotations: {str(e)}")
        return pd.DataFrame()


