# DS3 WiFi Receiver Timing and Deadline Reference

## Purpose

This document provides correct timing, deadline, and runtime information for the DS3 WiFi receiver DAG (`job_WIFI_5RXM.txt`) and SoC configuration (`SoC_MULTIPLE_BAL_NoC.txt`). Use this to update the existing DS3 configuration files with realistic values.

---

## 1. What the DAG Represents

The `job_WIFI_5RXM.txt` DAG models a **5-branch parallel 802.11 OFDM WiFi receiver** processing a single PPDU (Physical Layer Protocol Data Unit).

### DAG Structure

```
                    ┌→ payload_extraction_1 → fft_1 → pilot_1 → qpsk_demod_1 → deinterleaver_1 → format_conv_1 ─┐
                    ├→ payload_extraction_2 → fft_2 → pilot_2 → qpsk_demod_2 → deinterleaver_2 → format_conv_2 ─┤
match_filter (0) ───├→ payload_extraction_3 → fft_3 → pilot_3 → qpsk_demod_3 → deinterleaver_3 → format_conv_3 ─┼→ viterbi_decoder (31) → descrambler (32) → message_decode (33)
                    ├→ payload_extraction_4 → fft_4 → pilot_4 → qpsk_demod_4 → deinterleaver_4 → format_conv_4 ─┤
                    └→ payload_extraction_5 → fft_5 → pilot_5 → qpsk_demod_5 → deinterleaver_5 → format_conv_5 ─┘
```

The 5 parallel branches represent **5 OFDM symbols** being processed concurrently.

### Task Descriptions

| Task | Function |
|------|----------|
| match_filter | Correlate incoming samples with known preamble to detect frame start |
| payload_extraction | Extract OFDM symbol samples from received frame, remove cyclic prefix |
| fft | 64-point FFT to convert time-domain samples to frequency-domain subcarriers |
| pilot | Remove pilot subcarriers used for channel estimation and phase tracking |
| qpsk_demodulation | Soft-decision demodulation: map subcarrier values to log-likelihood ratios |
| deinterleaver | Reverse bit interleaving applied at transmitter |
| format_conversion | Pack/align soft bits for Viterbi decoder input |
| viterbi_decoder | Convolutional code decoding (constraint length 7, rate 1/2) |
| descrambler | Reverse LFSR scrambling applied at transmitter |
| message_decode | Extract final MAC payload, verify FCS |

---

## 2. Frame Duration and Deadline Derivation

### What is a Frame?

A **frame** (PPDU) is the complete physical-layer transmission unit in 802.11. It consists of:

1. **Preamble**: Training symbols for synchronization and channel estimation (16 μs)
2. **SIGNAL field**: One OFDM symbol containing rate/length info (4 μs)  
3. **DATA field**: Variable number of OFDM symbols containing the payload

**Source**: IEEE 802.11-2016 Standard, Clause 17 (OFDM PHY)

### PPDU Duration Formula

```
T_PPDU = T_preamble + T_signal + T_data
T_PPDU = 16 μs + 4 μs + (N_symbols × 4 μs)
```

Where:
- Each OFDM symbol = 3.2 μs data + 0.8 μs guard interval = 4 μs total
- N_symbols = ceil((16 + 8×L_payload + 6) / N_DBPS)
- N_DBPS = data bits per symbol (depends on MCS)

| Data Rate | Modulation | Code Rate | N_DBPS |
|-----------|------------|-----------|--------|
| 6 Mbps | BPSK | 1/2 | 24 |
| 12 Mbps | QPSK | 1/2 | 48 |
| 24 Mbps | 16-QAM | 1/2 | 96 |
| 36 Mbps | 16-QAM | 3/4 | 144 |
| 48 Mbps | 64-QAM | 2/3 | 192 |
| 54 Mbps | 64-QAM | 3/4 | 216 |

### Example: 1500-byte payload at 54 Mbps

```
N_symbols = ceil((16 + 8×1500 + 6) / 216) = ceil(12022/216) = 56 symbols
T_data = 56 × 4 μs = 224 μs
T_PPDU = 16 + 4 + 224 = 244 μs
```

### Why Frame Duration = Deadline

**Throughput constraint**: To sustain maximum data rate, the receiver must complete processing frame N before frame N+1 finishes arriving. If processing exceeds frame duration:
- Frames queue indefinitely → buffer overflow → dropped frames
- Or transmitter must slow down → reduced throughput

Therefore: **T_processing ≤ T_PPDU**

### Recommended DAG-Level Deadline

| Scenario | Deadline |
|----------|----------|
| 1500-byte @ 54 Mbps | 250 μs |
| 1500-byte @ 24 Mbps | 520 μs |
| Short frames (5 symbols) | 40-60 μs |
| Conservative (with slack) | 300 μs |

**For general use: 250 μs**

---

## 3. Realistic Task Execution Times

The current DS3 execution times reflect software implementations with DMA overhead. Below are corrected values for both the existing DS3 model and realistic dedicated hardware.

### DS3 Software/FPGA Model (Current)

These are the profiled values from the DS3 SoC configuration, representing software execution on ARM cores with FPGA accelerators accessed via DMA.

| Task | A7 (μs) | A15 (μs) | FFT Acc (μs) | VIT Acc (μs) |
|------|---------|----------|--------------|--------------|
| match_filter | 16 | 5 | - | - |
| payload_extraction | 8 | 4 | - | - |
| fft | 289 | 114 | 12 | - |
| pilot | 6 | 4 | - | - |
| qpsk_demodulation | 191 | 95 | - | - |
| deinterleaver | 16 | 9 | - | - |
| format_conversion | 7 | 4 | - | - |
| viterbi_decoder | 1828 | 739 | - | 2 |
| descrambler | 3 | 2 | - | - |
| message_decode | 90 | 39 | - | - |

**Critical Path (A15 + Accelerators): 176 μs**

### Realistic Dedicated Hardware Model

For a standards-compliant WiFi chip with pipelined ASIC/FPGA implementation:

| Task | Realistic (μs) | Source |
|------|----------------|--------|
| match_filter | 1-2 | Pipelined correlator |
| payload_extraction | 0.5 | Memory operation |
| fft (64-point) | 0.16-0.8 | 16-80 cycles @ 100 MHz [1] |
| pilot | 0.1 | Simple subtraction |
| qpsk_demodulation | 0.5 | Table lookup + LLR computation |
| deinterleaver | 0.1 | Memory permutation |
| format_conversion | 0.05 | Bit packing |
| viterbi_decoder | 1-2 | Parallel ACS, 8-148 cycles [2][3] |
| descrambler | 0.05 | LFSR |
| message_decode | 0.5 | CRC check + extraction |

**Realistic Critical Path: 4-7 μs**

#### Sources

[1] "A 64 point FFT processor can finish one 64 point FFT computation in 16 clock cycles" at 100 MHz = 160 ns. 
    - Radix-4³ parallel FFT, UMC 130nm CMOS (ResearchGate, 2020)

[2] "Parallel Viterbi Decoder: 8 clock cycles to generate an output"
    - Microchip Viterbi Decoder User Guide

[3] "FPGA implementation results show that its core can run as high as 510 MHz, and has 80.39 ns latency"
    - IEEE ISCAS 2010, SystemVerilog Viterbi implementation

---

## 4. Per-Task Deadline Calculation

For EDF/ELF scheduling, each task needs an absolute deadline from job arrival. Calculate using proportional distribution along the critical path.

### Formula

```
deadline[task] = (cumulative_exec_time[task] / critical_path_length) × DAG_deadline
```

### Per-Task Deadlines (DAG deadline = 250 μs, DS3 model)

| Task ID | Task | Cumulative (μs) | Deadline (μs) |
|---------|------|-----------------|---------------|
| 0 | match_filter | 5 | 7 |
| 1,7,13,19,25 | payload_extraction_* | 9 | 13 |
| 2,8,14,20,26 | fft_* | 21 | 30 |
| 3,9,15,21,27 | pilot_* | 25 | 35 |
| 4,10,16,22,28 | qpsk_demodulation_* | 120 | 170 |
| 5,11,17,23,29 | deinterleaver_* | 129 | 183 |
| 6,12,18,24,30 | format_conversion_* | 133 | 189 |
| 31 | viterbi_decoder | 135 | 192 |
| 32 | descrambler | 137 | 195 |
| 33 | message_decode | 176 | 250 |

---

## 5. Updated Configuration Files

### job_WIFI_5RXM.txt Updates

Replace the arbitrary deadline values (2, 3, 4) with the calculated values above.

**Original format:**
```
task_name earliest_start 0 deadline X input_vol Y output_vol Z
```

**Updated deadlines (in μs, assuming 250 μs DAG deadline):**

```
match_filter earliest_start 0 deadline 7 input_vol 81920 output_vol 8224
payload_extraction_1 earliest_start 0 deadline 13 input_vol 8224 output_vol 8192
fft_1 earliest_start 0 deadline 30 input_vol 8192 output_vol 8192
pilot_1 earliest_start 0 deadline 35 input_vol 8192 output_vol 8192
qpsk_demodulation_1 earliest_start 0 deadline 170 input_vol 8192 output_vol 224
deinterleaver_1 earliest_start 0 deadline 183 input_vol 224 output_vol 224
format_conversion_1 earliest_start 0 deadline 189 input_vol 224 output_vol 224
... (repeat pattern for branches 2-5)
viterbi_decoder earliest_start 0 deadline 192 input_vol 224 output_vol 224
descrambler earliest_start 0 deadline 195 input_vol 224 output_vol 112
message_decode earliest_start 0 deadline 250 input_vol 1120 output_vol 640
```

### SoC Configuration Notes

The execution times in `SoC_MULTIPLE_BAL_NoC.txt` represent profiled values from actual hardware (Odroid-XU3 and Zynq ZCU-102). These are valid for DS3's design-space exploration purpose but do NOT represent a SIFS-compliant WiFi implementation.

If you need faster execution times for real-time analysis, scale all values by a factor derived from:

```
scale_factor = target_critical_path / current_critical_path
             = 10 μs / 176 μs 
             = 0.057
```

This would make FFT accelerator = 0.68 μs, Viterbi accelerator = 0.11 μs, etc.

---

## 6. Summary Table

| Parameter | Value | Derivation |
|-----------|-------|------------|
| DAG function | Decode one 802.11 OFDM frame (5 symbols) | DAG structure analysis |
| Frame duration (54 Mbps, 1500B) | 244 μs | IEEE 802.11 PPDU timing formula |
| **Recommended DAG deadline** | **250 μs** | Frame duration (throughput constraint) |
| Critical path (DS3 model) | 176 μs | Sum of task times on longest path |
| Critical path (realistic HW) | 4-7 μs | Published FPGA/ASIC implementations |
| SIFS (5 GHz) | 16 μs | IEEE 802.11-2016 |
| SIFS (2.4 GHz) | 10 μs | IEEE 802.11-2016 |

---

## 7. References

1. IEEE Std 802.11-2016, "Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications"

2. Arda et al., "DS3: A System-Level Domain-Specific System-on-Chip Simulation Framework," arXiv:2003.09016v1, 2020

3. "FPGA Design and Implementation of a Convolutional Encoder and a Viterbi Decoder Based on 802.11a for OFDM," Wireless Engineering and Technology, 2012

4. "Design and Implementation of a low latency Radix-8 FFT using FPGA Architecture," ResearchGate, 2020

5. Microchip, "Viterbi Decoder User Guide," 2023

6. NI, "LabVIEW Communications 802.11 Application Framework White Paper," 2015

---

## 8. Instructions for Claude Code

To update the DS3 configuration files:

1. **For `job_WIFI_5RXM.txt`**:
   - Replace all `deadline X` values with the per-task deadlines from Section 5
   - Keep `earliest_start 0` for all tasks
   - Keep `input_vol` and `output_vol` unchanged

2. **For `SoC_MULTIPLE_BAL_NoC.txt`**:
   - Execution times are already realistic for DS3's software/FPGA model
   - If faster times needed, apply scale_factor = 0.057 to all execution times
   - Do NOT modify the resource structure, DVFS settings, or communication bandwidths

3. **For scheduler configuration**:
   - Set DAG-level deadline to 250 μs
   - Use per-task deadlines for EDF/ELF scheduling algorithms
   - Critical path = 176 μs provides ~42% slack at 250 μs deadline
