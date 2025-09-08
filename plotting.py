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
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

logger = logging.getLogger(__name__)

def get_scores_all_classes(feature_scores, coordinates, classes):
    """Helper function to extract scores for all classes using one-hot encoded sequences."""
    try:
        onehot_seqs = [feature_scores[1][seq_ind] for seq_ind in list(range(len(feature_scores[1])))] #onehot_seqs
        sequences = [crested.utils.hot_encoding_to_sequence(seq) for seq in onehot_seqs]
        
        seq_scores = []
        for sequence_index in list(range(len(sequences))):
            feature_coordinates = coordinates[sequence_index]
            sequence = sequences[sequence_index]
            scores = feature_scores[0][sequence_index] #scores for the individual sequence
            seq = feature_scores[1][sequence_index] #scores for the individual sequence
            #get all scores per class
            class_indices = list(range(len(classes)))
            class_scores_list = []
            for class_ind in class_indices:
                clas=classes[class_ind]
                class_scores = pd.DataFrame({clas:[float(scores[class_ind,seq_ind,seq[seq_ind].astype(bool)][0]) for seq_ind in list(range(scores.shape[1]))]})
                class_scores_list.append(class_scores)
            class_scores = pd.concat(class_scores_list, axis=1)
            class_scores['sequence'] = sequence
            class_scores['seq_position'] = list(range(len(sequence)))
            class_scores['seq_nucleotide'] = [sequence[x] for x in list(range(len(sequence)))]
            class_scores['coordinates'] = feature_coordinates
            seq_scores.append(class_scores)
        return_data = pd.concat(seq_scores)
        return return_data
    except Exception as e:
        logger.error(f"Error in get_scores_all_classes: {str(e)}")
        raise

def mark_sequence_matches(df, seq_col, pattern):
    """Mark sequence matches for pattern highlighting."""
    if pattern is None:
        df['match_status'] = 'no_match'
        return df
    
    # Simple pattern matching - you might want to enhance this
    df['match_status'] = 'no_match'
    # Add pattern matching logic here
    return df

def create_contribution_scores_dataframe(scores, one_hot_encoded_sequences, class_labels, coordinates, sequence):
    """Create a dataframe from already calculated contribution scores."""
    try:
        
        # Prepare data for processing
        feature_scores = [scores, one_hot_encoded_sequences]
        coordinates_list = [coordinates]  # Single coordinate for single sequence
        
        score_df = get_scores_all_classes(feature_scores, coordinates_list, class_labels)
        
        # Debug: check if DataFrame is empty
        logger.info(f"Created DataFrame with {len(score_df)} rows and columns: {list(score_df.columns)}")
        
        if score_df.empty:
            logger.warning("DataFrame is empty, returning empty DataFrame")
            return score_df
        
        # Convert from wide to long format for heatmap
        # The DataFrame has class names as columns, we need to melt it
        class_columns = [col for col in score_df.columns if col not in ['sequence', 'seq_position', 'seq_nucleotide', 'coordinates']]
        logger.info(f"Class columns to melt: {class_columns}")
        
        # Melt the DataFrame to long format
        score_df_long = score_df.melt(
            id_vars=['sequence', 'seq_position', 'seq_nucleotide', 'coordinates'],
            value_vars=class_columns,
            var_name='class_label',
            value_name='contribution_score'
        )
        
        logger.info(f"Converted to long format: {len(score_df_long)} rows, columns: {list(score_df_long.columns)}")
        
        # Add scaled scores
        try:
            score_df_long['scaled_contribution_score'] = score_df_long.groupby('coordinates')['contribution_score'].transform(zscore)
        except Exception as e:
            logger.warning(f"Could not add scaled scores: {e}")
            score_df_long['scaled_contribution_score'] = score_df_long['contribution_score']
        
        return score_df_long
        
    except Exception as e:
        logger.error(f"Error in create_contribution_scores_dataframe: {str(e)}")
        raise

def contribution_scores_df(sequence,
                           target_idx,
                           model,
                           genome,
                           form='long',  # long/wide
                           scale=True,
                           method='integrated_grad',
                           all_class_names=None,
                           batch_size=128,
                           coordinates=None):
    """Generate a dataframe for contribution scores."""
    try:
        # Get class names from model settings if not provided
        if all_class_names is None:
            model_settings = getattr(model, 'settings', {})
            all_class_names = model_settings.get('indexed_class_labels', [f'Class_{i}' for i in range(len(target_idx))])
        
        # Calculate contribution scores
        feature_scores = crested.tl.contribution_scores(
            method=method,
            input=sequence,
            target_idx=target_idx,
            model=model,
            genome=genome,
            batch_size=batch_size
        )
        
        # Debug: log the structure of feature_scores
        logger.info(f"Feature scores type: {type(feature_scores)}")
        if hasattr(feature_scores, 'shape'):
            logger.info(f"Feature scores shape: {feature_scores.shape}")
        else:
            logger.info(f"Feature scores length: {len(feature_scores) if hasattr(feature_scores, '__len__') else 'no length'}")
        
        # Extract scores for all classes
        # Use provided coordinates or create a placeholder
        if coordinates is None:
            coordinates = "sequence"
        
        coordinates_list = [coordinates]  # Single coordinate for single sequence
        score_df = get_scores_all_classes(feature_scores=feature_scores, coordinates=coordinates_list, classes=all_class_names)
        
        # Add scaled scores
        try:
            score_df['scaled_contribution_score'] = score_df.groupby('coordinates')['contribution_score'].transform(zscore)
        except Exception as e:
            logger.warning(f"Could not add scaled scores: {e}")
            score_df['scaled_contribution_score'] = score_df['contribution_score']
        
        if form == 'long':
            return_df = score_df
        elif form == 'wide':
            if scale:
                score_df['chr_position'] = [f"{x}_{y}" for x, y in zip(score_df['coordinates'], score_df['seq_position'])]
                score_df = score_df.reset_index()
                wide_df = score_df.pivot(columns='class_label', index='index', values='scaled_contribution_score')
            else:
                wide_df = score_df.pivot(columns='class_label', index='index', values='contribution_score')
            return_df = wide_df

        return return_df
    except Exception as e:
        logger.error(f"Error in contribution_scores_df: {str(e)}")
        raise

def plot_heatmap(
    df, 
    coordinates, 
    zoom_to=None,
    seq_positions_to_plot=(0, 1500),
    figsize=(30, 5), 
    palette=None,
    cmap="coolwarm",
    z_score=1,
    col_cluster=False,
    pattern_mark=None,
    save_file=None,
    xlabelsize=4
):
    """Create a heatmap plot of contribution scores."""
    try:
        # Debug: check DataFrame structure
        logger.info(f"Plot heatmap: DataFrame shape {df.shape}, columns: {list(df.columns)}")
        logger.info(f"Looking for coordinates: {coordinates}")
        
        if palette is None:
            palette = {'A':'Green','C':'Blue','G':'Orange','T':'Red'}
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check if coordinates column exists
        if 'coordinates' not in df.columns:
            raise ValueError("DataFrame does not have 'coordinates' column")
        
        # Filter by coordinates
        df_sub = df[df['coordinates'] == coordinates]
        df_sub = df_sub.sort_values('seq_position', ascending=True)
        match_df = df_sub[['coordinates','seq_position','seq_nucleotide']].drop_duplicates()
        match_df = mark_sequence_matches(df=match_df, seq_col="seq_nucleotide", pattern=pattern_mark)
        df_sub = df_sub.merge(match_df[['coordinates','seq_position','match_status']], left_on=['coordinates','seq_position'], right_on=['coordinates','seq_position'])
        
        if df_sub.empty:
            raise ValueError(f"No data found for coordinates: {coordinates}")
        
        # Pivot to wide format
        heatmap_data = df_sub.pivot(index="class_label", columns="seq_position", values="contribution_score")
        
        # Sort columns to ensure seq_position order
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
        
        # Get ordered nucleotides aligned with heatmap columns
        nuc_df = df_sub.drop_duplicates(subset=['seq_position'])[['seq_position','seq_nucleotide']]
        nuc_df = nuc_df.set_index('seq_position').reindex(heatmap_data.columns)
        ordered_labels = nuc_df['seq_nucleotide'].tolist()
        
        # Map colors for nucleotides
        ordered_colors = [palette.get(nuc, 'black') for nuc in ordered_labels]
        pattern_match_palette = dict(zip(['no_match','match','reverse'],['#d3d3d3','#00FF7F','blue']))
        pattern_match_cols = [pattern_match_palette.get(x, '#d3d3d3') for x in match_df['match_status']]
        
        
        # Zoom handling
        if zoom_to is not None and zoom_to < heatmap_data.shape[1]:
            mid = heatmap_data.shape[1] // 2
            start = max(mid - zoom_to // 2, 0)
            end = start + zoom_to
            heatmap_data = heatmap_data.iloc[:, start:end]
            ordered_labels = ordered_labels[start:end]
            ordered_colors = ordered_colors[start:end]
            pattern_match_cols = pattern_match_cols[start:end]
        elif seq_positions_to_plot is not None:
            start = seq_positions_to_plot[0]
            end = seq_positions_to_plot[1]
            heatmap_data = heatmap_data.iloc[:, start:end]
            ordered_labels = ordered_labels[start:end]
            ordered_colors = ordered_colors[start:end]
            pattern_match_cols = pattern_match_cols[start:end]
        
        # Create clustermap without any clustering to avoid dendrograms
        g = sns.clustermap(
            heatmap_data,
            cmap=cmap,
            col_cluster=False,  # Disable column clustering
            row_cluster=False,  # Disable row clustering
            z_score=z_score,
            figsize=figsize
        )
        
        # Pattern match colors are all grey anyway, so we'll just use nucleotide colors
        # This avoids the grey bar issue entirely
        
        # Remove the automatic color bar
        if hasattr(g, 'cax') and g.cax is not None:
            g.cax.remove()
        elif hasattr(g, 'cbar_ax') and g.cbar_ax is not None:
            g.cbar_ax.remove()
        
        # Add simple color bar above the plot
        cbar_ax = g.fig.add_axes([0.15, 0.96, 0.7, 0.02])
        data_min = heatmap_data.min().min()
        data_max = heatmap_data.max().max()
        norm = plt.Normalize(data_min, data_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([data_min, data_max])
        cbar.set_ticklabels(['Min', 'Max'])
        
        # Remove the title to avoid overlapping
        # g.fig.suptitle('Contribution Scores', fontsize=14, fontweight='bold', y=0.98)
        
        # Adjust the heatmap position to reduce white space and align with color bar
        g.ax_heatmap.set_position([0.15, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
        
        # Set all xticks to show labels
        num_cols = heatmap_data.shape[1]
        g.ax_heatmap.set_xticks(np.arange(num_cols))
        g.ax_heatmap.set_xticklabels(ordered_labels, rotation=0)
        
        # Color the labels and increase font size
        for label, color in zip(g.ax_heatmap.get_xticklabels(), ordered_colors):
            label.set_color(color)
            label.set_fontsize(12)  # Increase sequence letter size
        g.ax_heatmap.tick_params(axis='both', labelsize=xlabelsize)
        plt.yticks(fontsize=20)
        g.tick_params(axis='y', labelsize=20)
        
        # Rotate y-axis labels to horizontal
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
        
        # Remove axis titles
        g.ax_heatmap.set_xlabel('')
        g.ax_heatmap.set_ylabel('')
        
        if save_file is not None:
            plt.savefig(save_file)
        
        # Remove any automatic titles
        plt.title('')
        
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
        logger.error(f"Error creating heatmap: {str(e)}")
        raise

def create_plot(selected_files, coords, zoom_level=0, original_coords=None, cell_type_colors=None, zoom_factor=1.0):
    """Create a plot from BigWig files."""
    try:
        if not selected_files:
            logger.warning("No files selected for plotting")
            return None
            
        # Sort files by filename
        selected_files = sorted(selected_files, key=lambda x: os.path.basename(x))
        
        # Parse coordinates
        chrom, start, end = parse_coordinates(coords)
        logger.info(f"Creating plot for coordinates: {chrom}:{start}-{end}")
        
        # Parse original coordinates if provided
        original_start = None
        original_end = None
        if original_coords:
            _, original_start, original_end = parse_coordinates(original_coords)
            logger.info(f"Original coordinates: {original_start}-{original_end}")
        
        # Adjust coordinates based on zoom level and zoom factor
        if zoom_level != 0 or zoom_factor != 1.0:
            center = (start + end) // 2
            half_width = (end - start) // 2
            
            # First apply the fixed zoom level (for default zoom behavior)
            if zoom_level != 0:
                if zoom_level > 0:
                    # Zoom in
                    half_width = max(100, half_width - abs(zoom_level))
                else:
                    # Zoom out
                    half_width = half_width + abs(zoom_level)
            
            # Then apply the multiplicative zoom factor
            if zoom_factor != 1.0:
                half_width = int(half_width * zoom_factor)
                # Ensure minimum width of 100bp
                half_width = max(50, half_width)
            
            start = center - half_width
            end = center + half_width
            logger.info(f"Adjusted coordinates after zoom: {start}-{end} (factor: {zoom_factor}, level: {zoom_level})")
        
        # First pass: get the maximum value across all files using the original coordinates
        max_value = 0
        valid_files = []
        for file_path in selected_files:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                continue
                
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
                    valid_files.append(file_path)
                bw.close()
            except Exception as e:
                logger.error(f"Error getting max value from {file_path}: {str(e)}")
                continue
        
        if not valid_files:
            logger.error("No valid files found to plot")
            return None
            
        # Add 10% padding to the max value
        max_value = max_value * 1.1
        
        # Create the figure with stacked subplots
        fig, axes = plt.subplots(
            len(valid_files), 1,
            figsize=(12, 0.5 * len(valid_files)),
            sharex=True,
            gridspec_kw={'height_ratios': [1] * len(valid_files), 'hspace': 0.2}
        )
        
        # If there's only one subplot, make it a list for consistency
        if len(valid_files) == 1:
            axes = [axes]
        
        # Plot each selected bigwig file
        for i, (file_path, ax) in enumerate(zip(valid_files, axes)):
            try:
                # Get the file name and cell type
                file_name = os.path.basename(file_path)
                cell_type = file_name.replace('.bw', '')
                logger.info(f"Processing file: {file_name}, cell type: '{cell_type}'")
                
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
                
                # Highlight original region if it's different from current view
                if original_start is not None and original_end is not None and (original_start != start or original_end != end):
                    # Check if original region overlaps with current view
                    overlap_start = max(start, original_start)
                    overlap_end = min(end, original_end)
                    
                    if overlap_start < overlap_end:
                        # Get values for the overlapping region
                        original_values = bw.values(chrom, overlap_start, overlap_end)
                        original_positions = range(overlap_start, overlap_end)
                        
                        # Plot a light blue background for the original region
                        ax.axvspan(overlap_start, overlap_end, color='lightblue', alpha=0.2)
                        
                        # Plot the original line in a lighter color
                        ax.plot(original_positions, original_values, color=color, alpha=0.5)
                
                # Add label to top left of the plot
                ax.text(0.02, 0.95, cell_type, 
                       transform=ax.transAxes,
                       fontsize=8,
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

def create_contribution_scores_plot(scores, one_hot_encoded_sequences, chrom, start, end, new_start, new_end, n_classes, class_labels=None, zoom_n_bases=500, title = ''):
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
            title=title
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