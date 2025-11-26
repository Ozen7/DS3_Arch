#!/usr/bin/env python3
"""Quick test to verify SD values are being parsed correctly."""

import re

def check_job_file(filename, expected_sd_count):
    """Check that a job file has SD values in all earliest_start lines."""
    with open(filename, 'r') as f:
        content = f.read()

    earliest_start_lines = [line for line in content.split('\n') if 'earliest_start' in line]
    sd_lines = [line for line in earliest_start_lines if ' sd ' in line]

    print(f"\n{filename}:")
    print(f"  Total earliest_start lines: {len(earliest_start_lines)}")
    print(f"  Lines with SD values: {len(sd_lines)}")
    print(f"  Expected: {expected_sd_count}")

    if len(sd_lines) == expected_sd_count:
        print(f"  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL")
        return False

if __name__ == '__main__':
    all_pass = True

    all_pass &= check_job_file('config_Jobs/job_canny.txt', 12)
    all_pass &= check_job_file('config_Jobs/job_harris.txt', 18)
    all_pass &= check_job_file('config_Jobs/job_deblur.txt', 22)
    all_pass &= check_job_file('config_Jobs/job_gru.txt', 120)
    all_pass &= check_job_file('config_Jobs/job_lstm.txt', 144)

    print(f"\n{'='*50}")
    if all_pass:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")

    exit(0 if all_pass else 1)
