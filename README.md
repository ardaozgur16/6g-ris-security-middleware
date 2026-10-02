# 6G-RIS-Security-Middleware: Latency-Aware Physical Layer Jamming Defense for O-RAN Near-RT RIC

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![O-RAN Alliance](https://img.shields.io/badge/O--RAN-Near--RT%20RIC%20xApp-orange.svg)](https://www.o-ran.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source, differentiable physical-layer security middleware designed as an O-RAN Near-RT RIC xApp. It autonomously detects active/reactive jamming attacks and reconfigures low-resolution (1-bit / 2-bit) Reconfigurable Intelligent Surfaces (RIS) in real time to steer spatial nulls toward the interferer while sustaining legitimate beamforming gains.

---

## 📌 Key Highlights

- **Near-RT RIC Compliant (< 10 ms):** Evaluated against the O-RAN Near-RT RIC control-loop budget ($p99 < 10\text{ ms}$ target).
- **Hardware-Aware Differentiable Optimization:** Uses a custom PyTorch layer with Straight-Through Estimators (STE) for Quantization-Aware Training (QAT) to handle discrete hardware constraints (1-bit / 2-bit phase shifts) without continuous relaxation loss.
- **Hysteresis Attack Detection & Probing:** Employs EWMA / expected-SINR degradation tracking with automated probing slots to prevent blind-spot traps during successful nulling.
- **Comprehensive Threat Modeling:** Benchmarked against continuous Barrage Jammers and energy/statistical-sensing Reactive Burst Jammers.

---

## 🏗️ System Architecture

```text
+------------------------------------------------------------------------+
|                         O-RAN Near-RT RIC                              |
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |             6G RIS Security Middleware (xApp)                    |  |
|  |                                                                  |  |
|  |  +------------------------+      +----------------------------+  |  |
|  |  |    AttackDetector      |      |   PyTorch Differentiable   |  |  |
|  |  | (EWMA / Probing Logic) | ---> |        Optimizer           |  |  |
|  |  |                        |      |    (STE-QAT + Polish)      |  |  |
|  |  +------------------------+      +----------------------------+  |  |
|  +------------------------------------------------------------------+  |
|                 ^ (CQI/SINR Telemetry)         | (E2 Control: Phase Shifts)
+-----------------|------------------------------|-----------------------+
                  |                              v
        +-------------------+          +-------------------+
        |  gNodeB / User    |          |    6G RIS Array   |
        |  (Legitimate Link)|          | (Discrete Elements|
        +-------------------+          +-------------------+
                  ^                              ^
                  |                              |
                  +------- [ Jammer Signal ] ----+
