#!/usr/bin/env python3
"""Script to add SD values to GRU and LSTM job configuration files."""

import sys
import re

# SD patterns for GRU (15 nodes, repeats 8 times = 120 tasks)
GRU_SD_PATTERN = [84, 65, 71, 71, 57, 93, 93, 95, 60, 60, 60, 95, 60, 60, 60]

# SD patterns for LSTM (18 nodes, repeats 8 times = 144 tasks)
LSTM_SD_PATTERN = [114, 85, 94, 94, 75, 123, 123, 125, 79, 79, 79, 125, 79, 79, 125, 79, 79, 79]

def add_sd_to_file(input_file, output_file, sd_pattern):
    """Add SD values to job configuration file."""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    task_index = 0
    output_lines = []

    for line in lines:
        # Check if this is an earliest_start line
        if 'earliest_start' in line:
            # Parse the line
            parts = line.split()

            # Find the indices of keywords
            deadline_idx = parts.index('deadline')
            input_vol_idx = parts.index('input_vol')

            # Get the SD value for this task
            sd_value = sd_pattern[task_index % len(sd_pattern)]

            # Reconstruct the line with SD inserted
            new_parts = (
                parts[:deadline_idx+2] +  # Everything up to and including deadline value
                ['sd', str(sd_value)] +    # Add SD
                parts[input_vol_idx:]      # Rest of the line
            )

            output_lines.append(' '.join(new_parts) + '\n')
            task_index += 1
        else:
            output_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"Updated {task_index} tasks in {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python add_sd_values.py <gru|lstm> <job_file>")
        sys.exit(1)

    job_type = sys.argv[1].lower()
    job_file = sys.argv[2]

    if job_type == 'gru':
        add_sd_to_file(job_file, job_file, GRU_SD_PATTERN)
    elif job_type == 'lstm':
        add_sd_to_file(job_file, job_file, LSTM_SD_PATTERN)
    else:
        print(f"Unknown job type: {job_type}. Use 'gru' or 'lstm'")
        sys.exit(1)
