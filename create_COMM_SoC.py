#!/usr/bin/env python3
"""
DS3 Communication SoC Configuration Generator

Generates custom SoC configuration files with variable numbers of ARM cores and accelerators.

Usage:
    python create_COMM_SoC.py <pe_type> <count> [<pe_type> <count> ...]

PE types (shortnames):
    a7  - ARM A7 (LITTLE cores)
    a15 - ARM A15 (BIG cores)
    mm  - Matrix Multiplication accelerator
    fft - Fast Fourier Transform accelerator
    vit - Viterbi Decoder accelerator

Example:
    python create_COMM_SoC.py a7 1 a15 1 mm 1 fft 1 vit 1

Generates: config_SoC/SoC.COMM_a1_A1_m1_f1_v1.txt
"""

import sys
import os
from collections import OrderedDict


# PE type mapping: shortname -> (full_name, template_file, resource_type)
PE_TYPES = OrderedDict([
    ('a7', ('A7', 'a7.template', 'LTL')),
    ('a15', ('A15', 'a15.template', 'BIG')),
    ('mm', ('MM', 'mm.template', 'ACC')),
    ('fft', ('FFT', 'fft.template', 'ACC')),
    ('vit', ('VIT', 'vit.template', 'ACC'))
])

# Shortnames for filename generation (order matters for consistent naming)
FILENAME_SHORTCUTS = OrderedDict([
    ('a7', 'a'),
    ('a15', 'A'),
    ('mm', 'm'),
    ('fft', 'f'),
    ('vit', 'v')
])

TEMPLATE_DIR = 'config_SoC/templates'
OUTPUT_DIR = 'config_SoC'
BANDWIDTH_SATURATION = int((0.33) * 16000)  # bytes/microsecond = 5280

# Scratchpad sizes per PE type (in bytes)
SCRATCHPAD_SIZES = {
    'a7': 32768,      # 32 KB
    'a15': 65536,     # 64 KB
    'mm': 8192,       # 8 KB
    'fft': 16384,     # 16 KB
    'vit': 4096       # 4 KB
}


def print_usage():
    """Print usage information."""
    print(__doc__)
    sys.exit(1)


def parse_arguments(args):
    """
    Parse command-line arguments into PE counts.

    Args:
        args: List of command-line arguments (PE type, count pairs)

    Returns:
        OrderedDict mapping PE shortname to count

    Raises:
        ValueError: If arguments are invalid
    """
    if len(args) < 2 or len(args) % 2 != 0:
        raise ValueError("Arguments must be <pe_type> <count> pairs")

    pe_counts = OrderedDict()

    for i in range(0, len(args), 2):
        pe_type = args[i].lower()
        try:
            count = int(args[i + 1])
        except ValueError:
            raise ValueError("Count must be an integer: {}".format(args[i + 1]))

        if pe_type not in PE_TYPES:
            raise ValueError("Unknown PE type: {}. Valid types: {}".format(
                pe_type, ', '.join(PE_TYPES.keys())))

        if count < 0:
            raise ValueError("Count must be non-negative: {}".format(count))

        if count == 0:
            continue  # Skip PEs with count 0

        # Accumulate counts if same PE specified multiple times
        if pe_type in pe_counts:
            pe_counts[pe_type] += count
        else:
            pe_counts[pe_type] = count

    if not pe_counts:
        raise ValueError("At least one PE with count > 0 required")

    return pe_counts


def read_template(template_name):
    """
    Read PE template file.

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


def generate_pes(pe_counts):
    """
    Generate PE configuration strings with unique IDs and names.

    Resource ID assignment order: A7 → A15 → MM → FFT → VIT → MEMORY

    Args:
        pe_counts: OrderedDict mapping PE shortname to count

    Returns:
        List of (resource_id, config_string) tuples
    """
    pes = []
    resource_id = 0

    # Process in fixed order: A7, A15, MM, FFT, VIT
    for pe_type in ['a7', 'a15', 'mm', 'fft', 'vit']:
        if pe_type not in pe_counts:
            continue

        count = pe_counts[pe_type]
        full_name, template_file, resource_type = PE_TYPES[pe_type]
        template = read_template(template_file)

        # Get scratchpad size for this PE type
        scratchpad_size = SCRATCHPAD_SIZES[pe_type]

        for instance in range(count):
            # Generate unique name (e.g., A7, A7_0, A7_1, or MM if count == 1)
            if count == 1:
                pe_name = full_name
            else:
                pe_name = "{}_{}".format(full_name, instance)

            # Replace placeholders in template
            config = template.replace('{ID}', str(resource_id))
            config = config.replace('{NAME}', pe_name)
            config = config.replace('{FAMILY}', full_name)  # Use full_name as family
            config = config.replace('{SCRATCHPAD_SIZE}', str(scratchpad_size))

            pes.append((resource_id, config))
            resource_id += 1

    return pes, resource_id


def generate_comm_bandwidth_matrix(num_pes):
    """
    Generate communication bandwidth matrix (upper triangular with diagonal).

    Args:
        num_pes: Total number of processing elements (INCLUDING MEMORY)

    Returns:
        String containing bandwidth matrix lines
    """
    lines = []
    lines.append("# comm_band for all PEs and memory")
    lines.append("# bandwidth is in terms of bytes/microsecond (to match up with the DAGs)")
    lines.append("# 16 GB/s @ 33% saturation = 5280 bytes/microsecond")

    # Upper triangular matrix INCLUDING diagonal
    # Each PE i has entries from i to num_pes-1 (including itself)
    for i in range(num_pes):
        for j in range(i, num_pes):
            lines.append("comm_band {} {} {}".format(i, j, BANDWIDTH_SATURATION))
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


def generate_filename(pe_counts):
    """
    Generate output filename based on PE counts.

    Args:
        pe_counts: OrderedDict mapping PE shortname to count

    Returns:
        Filename string (e.g., "SoC.COMM_a1_A1_m1_f1_v1.txt")
    """
    parts = []

    # Iterate in fixed order for consistent naming
    for pe_type, shortcut in FILENAME_SHORTCUTS.items():
        if pe_type in pe_counts and pe_counts[pe_type] > 0:
            parts.append("{}{}".format(shortcut, pe_counts[pe_type]))

    if not parts:
        parts = ['custom']

    return "SoC.COMM_{}.txt".format('_'.join(parts))


def generate_soc_config(pe_counts):
    """
    Generate complete SoC configuration file content.

    Args:
        pe_counts: OrderedDict mapping PE shortname to count

    Returns:
        Complete SoC configuration as string
    """
    lines = []

    # Generate PE configurations
    pes, next_id = generate_pes(pe_counts)

    for resource_id, config in pes:
        lines.append(config)

    # Add MEMORY resource (must come before comm_band matrix)
    memory_id = next_id
    lines.append(generate_memory_resource(memory_id))

    # Add comment for comm_band_self
    lines.append("# comm_band when the source and destination are the same PE")
    lines.append("comm_band_self 1000000")

    # Generate inter-PE communication bandwidth matrix (including MEMORY)
    # Total PEs = all PEs + MEMORY
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


def print_summary(pe_counts, filename):
    """
    Print summary of generated SoC.

    Args:
        pe_counts: OrderedDict mapping PE shortname to count
        filename: Output filename
    """
    total_pes = sum(pe_counts.values())

    print("\n" + "=" * 60)
    print("SoC Configuration Summary")
    print("=" * 60)
    print("Output file: {}".format(filename))
    print("Total PEs: {}".format(total_pes))
    print("\nPE breakdown:")

    for pe_type, count in pe_counts.items():
        full_name, _, resource_type = PE_TYPES[pe_type]
        print("  {:<20} ({:<10}): {}".format(full_name, pe_type, count))

    print("  MEMORY              (system)  : 1")
    print("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) < 3 or '--help' in sys.argv or '-h' in sys.argv:
        print_usage()

    try:
        # Parse command-line arguments
        pe_counts = parse_arguments(sys.argv[1:])

        # Generate filename
        filename = generate_filename(pe_counts)

        # Generate SoC configuration
        config_content = generate_soc_config(pe_counts)

        # Write to file
        write_soc_file(filename, config_content)

        # Print summary
        print_summary(pe_counts, filename)

    except (ValueError, IOError) as e:
        print("Error: {}".format(e), file=sys.stderr)
        print("\nRun with --help for usage information", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
