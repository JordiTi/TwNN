import os
import pandas as pd

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_dir=script_dir
output_dir = os.path.join(parent_dir, 'text_normalized')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(output_path)
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Get the initial position (first row)
        initial_x = df['x'].iloc[0]
        initial_y = df['y'].iloc[0]
        
        # Normalize by subtracting initial position
        df['x'] = df['x'] - initial_x
        df['y'] = df['y'] - initial_y
        
        # Save to output directory
        df.to_csv(output_path, index=False)
        print(f"Processed {filename}")