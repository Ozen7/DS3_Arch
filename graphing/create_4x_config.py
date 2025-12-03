#!/usr/bin/env python3
"""Generate 4x uniform SoC configuration from 3x template."""

import re

# Read 3x config
with open('../config_SoC/SoC.i3_g3_cv3_h3_e3_c3_el3.txt', 'r') as f:
    lines = f.readlines()

# Parse to find accelerator blocks and memory section
accelerator_templates = {}
current_family = None
current_block = []
memory_section = []
in_memory = False

for line in lines:
    if 'add_new_resource resource_type ACC' in line:
        # Save previous block if exists
        if current_family and current_block:
            if current_family not in accelerator_templates:
                accelerator_templates[current_family] = []
            accelerator_templates[current_family].append(current_block)

        # Start new block
        match = re.search(r'accelerator_family\s+(\S+)', line)
        if match:
            current_family = match.group(1)
            current_block = [line]
    elif 'add_new_resource resource_type MEM' in line:
        # Save last accelerator block
        if current_family and current_block:
            if current_family not in accelerator_templates:
                accelerator_templates[current_family] = []
            accelerator_templates[current_family].append(current_block)

        in_memory = True
        memory_section = [line]
    elif in_memory:
        memory_section.append(line)
    elif current_block is not None:
        current_block.append(line)
        if line.strip() == '':  # Empty line marks end of block
            if current_family:
                if current_family not in accelerator_templates:
                    accelerator_templates[current_family] = []
                accelerator_templates[current_family].append(current_block)
            current_block = []

# Generate 4x config
output = []
resource_id = 0

# Accelerator families in order
families = ['ISP', 'GRAYSCALE', 'CONVOLUTION', 'HARRIS_NON_MAX', 'EDGE_TRACKING', 'CANNY_NON_MAX', 'ELEM_MATRIX']

for family in families:
    if family not in accelerator_templates:
        continue

    # Use first instance as template
    template = accelerator_templates[family][0]

    # Generate 4 copies
    for i in range(4):
        for line in template:
            # Update resource_name and resource_ID
            if 'resource_name' in line and 'resource_ID' in line:
                # Replace name
                line = re.sub(r'resource_name\s+\S+', f'resource_name {family}_{i}', line)
                # Replace ID
                line = re.sub(r'resource_ID\s+\d+', f'resource_ID {resource_id}', line)
            output.append(line)

        resource_id += 1

# Add memory section with updated ID and comm_band
memory_id = resource_id
for line in memory_section:
    if 'resource_ID' in line:
        line = re.sub(r'resource_ID\s+\d+', f'resource_ID {memory_id}', line)
    elif 'comm_band' in line and 'comm_band_self' not in line:
        # Update comm_band entries
        parts = line.split()
        if len(parts) >= 4 and parts[0] == 'comm_band':
            # Check if ID needs updating
            src_id = int(parts[1])
            dst_id = int(parts[2])
            bandwidth = parts[3]

            # Only include if both IDs are <= memory_id
            if src_id <= memory_id and dst_id <= memory_id:
                output.append(line)
            continue
    output.append(line)

# Now regenerate all comm_band entries for 4x (28 accelerators + 1 memory = 29 total)
# Remove old comm_band lines
output_filtered = []
skip_comm_band = False
for line in output:
    if line.strip().startswith('comm_band') and 'comm_band_self' not in line:
        skip_comm_band = True
        continue
    output_filtered.append(line)

# Generate new comm_band matrix
comm_band_lines = []
for src in range(memory_id + 1):
    for dst in range(src, memory_id + 1):
        comm_band_lines.append(f'comm_band {src} {dst} 16000\n')

# Insert comm_band lines after comm_band_self
final_output = []
for line in output_filtered:
    final_output.append(line)
    if 'comm_band_self' in line:
        final_output.extend(comm_band_lines)

# Write output
with open('../config_SoC/SoC.i4_g4_cv4_h4_e4_c4_el4.txt', 'w') as f:
    f.writelines(final_output)

print(f"[SUCCESS] Generated SoC.i4_g4_cv4_h4_e4_c4_el4.txt")
print(f"[INFO] Created 4x of each accelerator family = {resource_id} accelerators")
print(f"[INFO] Memory resource ID = {memory_id}")
print(f"[INFO] Total resources = {memory_id + 1}")
