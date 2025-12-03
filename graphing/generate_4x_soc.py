#!/usr/bin/env python3
"""
Generate 4x SoC configuration file based on the 3x pattern.
"""

import re

# Read the 3x configuration
with open('../config_SoC/SoC.i3_g3_cv3_h3_e3_c3_el3.txt', 'r') as f:
    content_3x = f.read()

# Split into resource blocks
blocks = content_3x.strip().split('\n\n')

# Parse each block to understand the pattern
accelerator_families = {}
resource_id = 0

for block in blocks:
    lines = block.strip().split('\n')
    if not lines:
        continue

    # Parse the add_new_resource line
    first_line = lines[0]
    if 'add_new_resource' not in first_line:
        continue

    # Extract accelerator_family
    match = re.search(r'accelerator_family\s+(\S+)', first_line)
    if not match:
        continue

    family = match.group(1)

    # Store one representative block per family (without the _N suffix in name)
    if family not in accelerator_families:
        # Remove resource_name and resource_ID, we'll regenerate them
        template_lines = []
        for line in lines:
            if 'resource_name' in line or 'resource_ID' in line:
                # Extract everything except name and ID
                parts = line.split()
                new_parts = []
                skip_next = False
                for i, part in enumerate(parts):
                    if skip_next:
                        skip_next = False
                        continue
                    if part in ['resource_name', 'resource_ID']:
                        skip_next = True
                        continue
                    new_parts.append(part)
                template_lines.append(' '.join(new_parts))
            else:
                template_lines.append(line)

        accelerator_families[family] = '\n'.join(template_lines)

# Now generate 4x configuration
output_lines = []
resource_id = 0

# Count of each accelerator type in order
family_order = ['ISP', 'GRAYSCALE', 'CONVOLUTION', 'HARRIS', 'EDGE', 'CANNY', 'ELEMENT_WISE']

for family in family_order:
    if family not in accelerator_families:
        continue

    for i in range(4):  # Create 4 of each
        template = accelerator_families[family]

        # Insert resource_name and resource_ID
        lines = template.split('\n')
        first_line_parts = lines[0].split()

        # Find where to insert name and ID (after scratchpad_size value)
        insert_idx = None
        for idx, part in enumerate(first_line_parts):
            if part == 'scratchpad_size':
                insert_idx = idx + 2  # After scratchpad_size <value>
                break

        if insert_idx:
            resource_name = f"{family}_{i}"
            first_line_parts.insert(insert_idx, 'resource_name')
            first_line_parts.insert(insert_idx + 1, resource_name)
            first_line_parts.insert(insert_idx + 2, 'resource_ID')
            first_line_parts.insert(insert_idx + 3, str(resource_id))

            lines[0] = ' '.join(first_line_parts)

        output_lines.append('\n'.join(lines))
        resource_id += 1

# Add memory resource (copy from 3x)
memory_block = None
for block in blocks:
    if 'resource_type MEM' in block:
        memory_block = block
        break

if memory_block:
    # Update resource_ID for memory
    memory_lines = memory_block.split('\n')
    for i, line in enumerate(memory_lines):
        if 'resource_ID' in line:
            parts = line.split()
            for j, part in enumerate(parts):
                if part == 'resource_ID' and j + 1 < len(parts):
                    parts[j + 1] = str(resource_id)
            memory_lines[i] = ' '.join(parts)
            break
    output_lines.append('\n'.join(memory_lines))

# Write output
output_content = '\n\n'.join(output_lines)
with open('../config_SoC/SoC.i4_g4_cv4_h4_e4_c4_el4.txt', 'w') as f:
    f.write(output_content + '\n')

print("[SUCCESS] Generated SoC.i4_g4_cv4_h4_e4_c4_el4.txt")
print(f"[INFO] Created {resource_id} accelerators + 1 memory resource")
