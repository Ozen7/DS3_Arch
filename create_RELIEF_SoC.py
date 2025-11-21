#!/usr/bin/env python3
"""
DS3 SoC Configuration Generator

Generates custom SoC configuration files with variable numbers of accelerators.

Usage:
    python create_RELIEF_SoC.py <acc_type> <count> [<acc_type> <count> ...]

Accelerator types (shortnames):
    isp       - Image Signal Processor
    grayscale - Grayscale conversion
    conv      - Convolution operations
    harris    - Harris corner detection (non-maximum suppression)
    edge      - Edge tracking (Canny)
    canny     - Canny non-maximum suppression
    elem      - Element-wise matrix operations

Example:
    python create_RELIEF_SoC.py harris 3 edge 1 canny 1 elem 1 conv 3 isp 1 grayscale 1

Generates: config_SoC/SoC.i1_g1_cv3_h3_e1_c1_el1.txt
"""

import sys
import os
from collections import OrderedDict


# Accelerator type mapping: shortname -> (full_name, template_file)
ACCELERATOR_TYPES = OrderedDict([
    ('isp', ('ISP', 'isp.template')),
    ('grayscale', ('GRAYSCALE', 'grayscale.template')),
    ('conv', ('CONVOLUTION', 'conv.template')),
    ('harris', ('HARRIS_NON_MAX', 'harris.template')),
    ('edge', ('EDGE_TRACKING', 'edge.template')),
    ('canny', ('CANNY_NON_MAX', 'canny.template')),
    ('elem', ('ELEM_MATRIX', 'elem.template'))
])

# Shortnames for filename generation (order matters for consistent naming)
FILENAME_SHORTCUTS = OrderedDict([
    ('isp', 'i'),
    ('grayscale', 'g'),
    ('conv', 'cv'),
    ('harris', 'h'),
    ('edge', 'e'),
    ('canny', 'c'),
    ('elem', 'el')
])

TEMPLATE_DIR = 'config_SoC/templates'
OUTPUT_DIR = 'config_SoC'
DEFAULT_BANDWIDTH = 5460  # bytes/microsecond (16 GB/s)

# Scratchpad sizes per accelerator type (in bytes)
SCRATCHPAD_SIZES = {
    'isp': 115204,
    'grayscale': 180244,
    'conv': 196708,
    'harris': 196608,
    'edge': 98432,
    'canny': 262144,
    'elem': 262144
}


def print_usage():
    """Print usage information."""
    print(__doc__)
    sys.exit(1)


def parse_arguments(args):
    """
    Parse command-line arguments into accelerator counts.

    Args:
        args: List of command-line arguments (accelerator type, count pairs)

    Returns:
        OrderedDict mapping accelerator shortname to count

    Raises:
        ValueError: If arguments are invalid
    """
    if len(args) < 2 or len(args) % 2 != 0:
        raise ValueError("Arguments must be <acc_type> <count> pairs")

    acc_counts = OrderedDict()

    for i in range(0, len(args), 2):
        acc_type = args[i].lower()
        try:
            count = int(args[i + 1])
        except ValueError:
            raise ValueError("Count must be an integer: {}".format(args[i + 1]))

        if acc_type not in ACCELERATOR_TYPES:
            raise ValueError("Unknown accelerator type: {}. Valid types: {}".format(
                acc_type, ', '.join(ACCELERATOR_TYPES.keys())))

        if count < 0:
            raise ValueError("Count must be non-negative: {}".format(count))

        if count == 0:
            continue  # Skip accelerators with count 0

        # Accumulate counts if same accelerator specified multiple times
        if acc_type in acc_counts:
            acc_counts[acc_type] += count
        else:
            acc_counts[acc_type] = count

    if not acc_counts:
        raise ValueError("At least one accelerator with count > 0 required")

    return acc_counts


def read_template(template_name):
    """
    Read accelerator template file.

    Args:
        template_name: Name of the template file

    Returns:
        Template content as string
    """
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    if not os.path.exists(template_path):
        raise IOError("Template file not found: {}".format(template_path))

    with open(template_path, 'r') as f:
        return f.read()


def generate_accelerators(acc_counts):
    """
    Generate accelerator configuration strings with unique IDs and names.

    Args:
        acc_counts: OrderedDict mapping accelerator shortname to count

    Returns:
        List of (resource_id, config_string) tuples
    """
    accelerators = []
    resource_id = 0

    for acc_type, count in acc_counts.items():
        full_name, template_file = ACCELERATOR_TYPES[acc_type]
        template = read_template(template_file)

        # Get scratchpad size for this accelerator type
        scratchpad_size = SCRATCHPAD_SIZES[acc_type]

        for instance in range(count):
            # Generate unique name (e.g., ISP_0, ISP_1, or ISP if count == 1)
            if count == 1:
                acc_name = full_name
            else:
                acc_name = "{}_{}".format(full_name, instance)

            # Replace placeholders in template
            config = template.replace('{ID}', str(resource_id))
            config = config.replace('{NAME}', acc_name)
            config = config.replace('{FAMILY}', full_name)  # Use full_name as family
            config = config.replace('{SCRATCHPAD_SIZE}', str(scratchpad_size))

            accelerators.append((resource_id, config))
            resource_id += 1

    return accelerators, resource_id


def generate_comm_bandwidth_matrix(num_pes):
    """
    Generate communication bandwidth matrix (upper triangular with diagonal).

    Args:
        num_pes: Total number of processing elements (INCLUDING MEMORY)

    Returns:
        String containing bandwidth matrix lines
    """
    lines = []
    lines.append("# comm_band for all accelerators and memory")
    lines.append("# bandwidth is in terms of bytes/microsecond (to match up with the DAGs)")
    lines.append("# 16 GB/s")

    # Upper triangular matrix INCLUDING diagonal
    # Each PE i has entries from i to num_pes-1 (including itself)
    for i in range(num_pes):
        for j in range(i, num_pes):
            lines.append("comm_band {} {} {}".format(i, j, DEFAULT_BANDWIDTH))
        lines.append("")  # Add blank line after each PE's section

    return '\n'.join(lines)


def generate_memory_resource(memory_id):
    """
    Generate MEMORY resource configuration.

    Args:
        memory_id: Resource ID for MEMORY (should be last)

    Returns:
        String containing MEMORY resource configuration
    """
    config = "add_new_resource resource_type MEM resource_name MEMORY resource_ID {} capacity 1 num_supported_functionalities 1 DVFS_mode none\n".format(memory_id)
    config += "None 0"
    return config


def generate_filename(acc_counts):
    """
    Generate output filename based on accelerator counts.

    Args:
        acc_counts: OrderedDict mapping accelerator shortname to count

    Returns:
        Filename string (e.g., "SoC.i1_g1_cv3_h3_e1_c1_el1.txt")
    """
    parts = []

    # Iterate in fixed order for consistent naming
    for acc_type, shortcut in FILENAME_SHORTCUTS.items():
        if acc_type in acc_counts and acc_counts[acc_type] > 0:
            parts.append("{}{}".format(shortcut, acc_counts[acc_type]))

    if not parts:
        parts = ['custom']

    return "SoC.{}.txt".format('_'.join(parts))


def generate_soc_config(acc_counts):
    """
    Generate complete SoC configuration file content.

    Args:
        acc_counts: OrderedDict mapping accelerator shortname to count

    Returns:
        Complete SoC configuration as string
    """
    lines = []

    # Generate accelerator configurations
    accelerators, next_id = generate_accelerators(acc_counts)

    for resource_id, config in accelerators:
        lines.append(config)

    # Add MEMORY resource (must come before comm_band matrix)
    memory_id = next_id
    lines.append(generate_memory_resource(memory_id))

    # Add comment for comm_band_self
    lines.append("# comm_band when the source and destination are the same PE")
    lines.append("comm_band_self 1000000")

    # Generate inter-PE communication bandwidth matrix (including MEMORY)
    # Total PEs = accelerators + MEMORY
    total_pes = memory_id + 1
    lines.append(generate_comm_bandwidth_matrix(total_pes))

    return '\n'.join(lines)


def write_soc_file(filename, content):
    """
    Write SoC configuration to file.

    Args:
        filename: Output filename
        content: SoC configuration content
    """
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(content)

    print("Generated SoC configuration: {}".format(output_path))


def print_summary(acc_counts, filename):
    """
    Print summary of generated SoC.

    Args:
        acc_counts: OrderedDict mapping accelerator shortname to count
        filename: Output filename
    """
    total_accs = sum(acc_counts.values())

    print("\n" + "=" * 60)
    print("SoC Configuration Summary")
    print("=" * 60)
    print("Output file: {}".format(filename))
    print("Total accelerators: {}".format(total_accs))
    print("\nAccelerator breakdown:")

    for acc_type, count in acc_counts.items():
        full_name, _ = ACCELERATOR_TYPES[acc_type]
        print("  {:<20} ({:<10}): {}".format(full_name, acc_type, count))

    print("  MEMORY              (system)  : 1")
    print("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) < 3 or '--help' in sys.argv or '-h' in sys.argv:
        print_usage()

    try:
        # Parse command-line arguments
        acc_counts = parse_arguments(sys.argv[1:])

        # Generate filename
        filename = generate_filename(acc_counts)

        # Generate SoC configuration
        config_content = generate_soc_config(acc_counts)

        # Write to file
        write_soc_file(filename, config_content)

        # Print summary
        print_summary(acc_counts, filename)

    except (ValueError, IOError) as e:
        print("Error: {}".format(e), file=sys.stderr)
        print("\nRun with --help for usage information", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
