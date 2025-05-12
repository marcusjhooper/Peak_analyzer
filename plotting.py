import io
import base64
import tempfile
import os
import matplotlib.pyplot as plt
import pyBigWig
import logging
from utils import parse_coordinates
import crested
import numpy as np

logger = logging.getLogger(__name__)

def create_plot(selected_files, coords, zoom_level=0, original_coords=None, cell_type_colors=None):
    """Create a plot from BigWig files."""
    try:
        if not selected_files:
            return None
            
        
        # Parse coordinates
        chrom, start, end = parse_coordinates(coords)
        
        # Parse original coordinates if provided
        original_start = None
        original_end = None
        if original_coords:
            _, original_start, original_end = parse_coordinates(original_coords)
        
        # Adjust coordinates based on zoom level
        if zoom_level != 0:
            center = (start + end) // 2
            half_width = (end - start) // 2
            if zoom_level > 0:
                # Zoom in
                half_width = max(100, half_width - abs(zoom_level))
            else:
                # Zoom out
                half_width = half_width + abs(zoom_level)
            start = center - half_width
            end = center + half_width
        
        # First pass: get the maximum value across all files using the original coordinates
        max_value = 0
        for file_path in selected_files:
            try:
                bw = pyBigWig.open(file_path)
                # Use original coordinates if available, otherwise use current coordinates
                if original_start is not None and original_end is not None:
                    values = bw.values(chrom, original_start, original_end)
                else:
                    values = bw.values(chrom, start, end)
                if values:
                    current_max = max(v for v in values if v is not None)
                    max_value = max(max_value, current_max)
                bw.close()
            except Exception as e:
                logger.error(f"Error getting max value from {file_path}: {str(e)}")
                continue
        
        # Add 10% padding to the max value
        max_value = max_value * 1.1
        
        # Create the figure with stacked subplots
        fig, axes = plt.subplots(
            len(selected_files), 1,
            figsize=(12, 0.5 * len(selected_files)),  # Reduced height from 2 to 1 unit per plot
            sharex=True,
            gridspec_kw={'height_ratios': [1] * len(selected_files), 'hspace': 0.2}  # Reduced spacing between plots
        )
        
        # If there's only one subplot, make it a list for consistency
        if len(selected_files) == 1:
            axes = [axes]
        
        # Plot each selected bigwig file
        for i, (file_path, ax) in enumerate(zip(selected_files, axes)):
            try:
                # Get the file name and cell type
                file_name = os.path.basename(file_path)
                cell_type = file_name.replace('.bw', '')
                logger.debug(f"Processing file: {file_name}, cell type: '{cell_type}'")
                
                # Get the color for this cell type
                color = cell_type_colors.get(cell_type) if cell_type_colors else None
                if color is None:
                    # If no color found, use a default color
                    color = plt.cm.tab10(i % 10)
                    logger.debug(f"Using default color for {cell_type}")
                
                # Open the bigwig file
                bw = pyBigWig.open(file_path)
                
                # Get the values for the specified region
                values = bw.values(chrom, start, end)
                positions = range(start, end)
                
                # Plot the filled area
                ax.fill_between(positions, values, alpha=0.3, color=color)
                
                # Plot the line
                ax.plot(positions, values, color=color)
                
                # Highlight original region if zoomed out
                if original_start is not None and original_end is not None and (start < original_start or end > original_end):
                    # Get values for the original region
                    original_values = bw.values(chrom, original_start, original_end)
                    original_positions = range(original_start, original_end)
                    
                    # Plot a light blue background for the original region
                    ax.axvspan(original_start, original_end, color='lightblue', alpha=0.2)
                    
                    # Plot the original line in a lighter color
                    ax.plot(original_positions, original_values, color=color, alpha=0.5)
                
                # Add label to top left of the plot
                ax.text(0.02, 0.95, cell_type, 
                       transform=ax.transAxes,
                       fontsize=8,  # Reduced font size to match smaller plot
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                fc='white', 
                                ec='gray', 
                                alpha=0.8))
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Remove x-axis ticks and labels
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                
                # Set consistent y-axis limits for all plots
                ax.set_ylim(0, max_value)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Remove y-axis label since we're using text annotation
                ax.set_ylabel('')
                
                # Close the file
                bw.close()
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        # Add a title with the coordinates
        #plt.suptitle(f'{chrom}:{start}-{end}', fontsize=10)  # Reduced title font size
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64 for display in Dash
        data = base64.b64encode(buf.getbuffer()).decode("utf8")
        return f"data:image/png;base64,{data}"
    
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        return None

def create_contribution_scores_plot(scores, one_hot_encoded_sequences, chrom, start, end, new_start, new_end, n_classes, class_labels=None, zoom_n_bases=500):
    """Create a plot of contribution scores."""
    try:
        # Log the shape of the scores array
        logger.info(f"Scores shape: {scores.shape}")
        
        # Use provided class labels or create default ones
        if class_labels is None or len(class_labels) != n_classes:
            logger.warning(f"Using default class labels for {n_classes} classes")
            class_labels = [f"Class {i+1}" for i in range(n_classes)]
        
        logger.info(f"Creating plot with {n_classes} classes and labels: {class_labels}")
        
        # Create a new figure with a specific size
        plt.figure(figsize=(12, 6))
        
        # Create the contribution scores plot
        crested.pl.patterns.contribution_scores(
            scores,
            one_hot_encoded_sequences,
            sequence_labels="",
            class_labels=class_labels,
            zoom_n_bases=zoom_n_bases,
            title=f"Contribution Scores for {chrom}:{start}-{end}"
        )
        
        # Get the current axis and add vertical lines
        ax = plt.gca()
        original_start_rel = start - new_start
        original_end_rel = end - new_start
        ax.axvline(x=original_start_rel, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=original_end_rel, color='red', linestyle='--', alpha=0.5)
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        
        # Get the raw bytes
        plot_bytes = buf.getvalue()
        
        # Convert to base64 for display in Dash
        data = base64.b64encode(plot_bytes).decode("utf8")
        
        # Close the plot to free memory
        plt.close()
        
        return f"data:image/png;base64,{data}", plot_bytes
    
    except Exception as e:
        logger.error(f"Error creating contribution scores plot: {str(e)}")
        raise 