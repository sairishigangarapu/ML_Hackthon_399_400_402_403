# 05 — HMM and Viterbi Decoding (Teammate 1 Scope)

This directory is reserved in the team repository structure for Teammate 1's HMM + Viterbi sequential decoder (Model 1).

Model 4 (`NN-GLS + RLCD`) is a **pure neural geostatistical + calibrated RL architecture** with zero HMM/Viterbi dependency: spatial correlation is handled by **NN-GLS (Vecchia NNGP spatial covariance whitening + Kriging graph layers)** and sequential route continuity + probability calibration are learned end-to-end via **RLCD (GRPO strictly proper Brier + directed road-network continuity rewards)**.
