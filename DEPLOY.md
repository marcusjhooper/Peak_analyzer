# Deploying the ATAC-seq Visualization Tool with rsconnect-python

This guide provides instructions for deploying the ATAC-seq Visualization Tool using rsconnect-python.

## Prerequisites

1. Install rsconnect-python:
```bash
pip install rsconnect-python
```

2. Create a Connect account if you don't have one already at https://www.rstudio.com/products/connect/

3. Get your Connect API key:
   - Log into your Connect server
   - Go to your user profile
   - Click on "API Keys"
   - Create a new API key

## Deployment Steps

1. Create a requirements.txt file (if not already present):
```bash
pip freeze > requirements.txt
```

2. Configure the data directory in your app.py:
```python
import os
from dash import Dash

# Set the data directory
DATA_DIR = '/path/to/your/data'  # Change this to your data directory

# Initialize the Dash app with custom static folder
app = Dash(__name__, 
          static_folder=DATA_DIR,
          static_url_path='/data')

# Update your file paths to use the static URL
class_bigwigs = {row['class_id_label']: f"/data/class/{row['class_id_label']}.bw" for _, row in hierarchy_df.drop_duplicates('class_id_label').iterrows()}
subclass_bigwigs = {row['subclass_id_label']: f"/data/subclass/{row['subclass_id_label']}.bw" for _, row in hierarchy_df.drop_duplicates('subclass_id_label').iterrows()}
supertype_bigwigs = {row['supertype_label']: f"/data/supertype/{row['supertype_label']}.bw" for _, row in hierarchy_df.drop_duplicates('supertype_label').iterrows()}
```

3. Deploy the application:
```bash
rsconnect - deploy from github

## Important Notes

1. Data Files:
   - The app expects the following directory structure in your data directory:
     ```
     data/
     ├── class/
     ├── subclass/
     ├── supertype/
     ├── genome/
     ├── model/
     └── other/
     ```
   - Make sure the data directory is accessible to the Connect server
   - The data directory should be mounted at the same path on the Connect server
   - Update the DATA_DIR path in app.py to match your Connect server's data directory

2. Environment Variables:
   - If your app uses any environment variables, make sure to configure them in Connect
   - Go to your app settings in Connect and add any required environment variables
   - You can also set the data directory path as an environment variable:
     ```python
     DATA_DIR = os.getenv('DATA_DIR', '/path/to/your/data')
     ```

3. Memory Requirements:
   - The app may require significant memory due to the genome data and model loading
   - Configure appropriate memory limits in Connect based on your needs

4. Troubleshooting:
   - Check the Connect logs if the app fails to deploy
   - Verify all dependencies are listed in requirements.txt
   - Ensure the data directory is properly mounted and accessible
   - Check file permissions on the data directory

## Updating the Deployment

To update an existing deployment:

1. Make your changes to the code
2. Run the deploy command again:
```bash
rsconnect deploy dash --server https://your-connect-server.com --api-key your-api-key --name ATAC_vis --entrypoint app:app .
```

## Monitoring

- Monitor your app's performance through the Connect dashboard
- Check logs for any errors or issues
- Monitor memory usage and adjust resources if needed
- Verify data file access in the logs

## Support

If you encounter any issues:
1. Check the Connect logs
2. Verify all dependencies and data files are present
3. Ensure the data directory is properly mounted and accessible
4. Contact your Connect administrator if problems persist 
