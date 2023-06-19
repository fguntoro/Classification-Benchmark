import pandas as pd
import argparse
import os
import re

# Read the input files and output file path from Snakemake
input_files = snakemake.input
output_file = str(snakemake.output)


def main():
    # Initialize an empty DataFrame to store the summary
    summary = pd.DataFrame()

    # Iterate over each input file
    for file in input_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Concatenate the current DataFrame with the summary DataFrame
        summary = pd.concat([summary, df], axis=0)

    # Print the summary DataFrame
    print(summary)

    # Save the summary DataFrame to the output file
    print("Saving summary to", output_file)
    summary.to_csv(output_file, index=False)


# Call the main function
main()
