# PathSnap Live Agent State

**Last Updated:** 2026-11-18 10:06 IST  
**Current Stage:** `00_env_setup` — Installing Python dependencies into `.venv`  
**Hardware:** Intel i9-13900H (20 threads), 15 GB RAM, NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, CUDA 13.2)

## Pipeline Progress
- [x] Read and audited all 5 official PathSnap PDFs (`Main Guidelines`, `Dataset Exploration`, `Models and Training Guidelines`, `Deliverables`, `Assessment and Plagiarism Policy`)
- [x] Initial raw dataset audit (`train`: 80 trips / 208,417 GPS pts; `student_test`: 10 trips / 19,160 GPS pts)
- [x] Created project directory layout following deliverable order + `agents/` + `theory_and_viva/`
- [ ] `00_env_setup`: Install Python dependencies into `.venv`
- [ ] `01_data_processing_eda`: Full EDA, missing values, CRS projection, graph connectivity, EDA plots & Folium maps
- [ ] `02_feature_engineering`: Candidate generation (local metric projection + spatial index), monotonic route alignment for ground-truth observation labeling, spatial/kinematic/topological feature extraction
- [ ] `03_base_model_training`: Train diverse classical/neural baselines (Logistic Regression, Random Forest, LightGBM Untuned, MLP-BCE) with 5-fold GroupKFold CV
- [ ] `04_bayesian_hpo`: Optuna TPE Bayesian Hyperparameter Optimisation for LightGBM (Model 1 classifier) + HMM transition parameters on 80 train trips CV
- [ ] `05_nn_gls_and_rlcd_training`: Train Model 2 (RLCD), Model 3 (NN-GLS with NNGP Vecchia GLS whitening + residual kriging), Model 4 (NN-GLS + RLCD hybrid) on GPU with 5-fold GroupKFold CV and epoch-wise train/val logs
- [ ] `06_hmm_viterbi_decoding`: Full HMM + Viterbi route reconstruction using road-network Dijkstra shortest paths, gap handling, and disconnected segment fallback — evaluate Classifier-only vs Classifier+HMM across ALL models
- [ ] `07_evaluation_and_plots`: Consolidated evaluation tables, confusion matrices, calibration curves (ECE/Brier), model comparison charts, training loss curves
- [ ] `08_folium_visualisations`: Interactive Folium HTML maps (OpenStreetMap free tiles) + rendered screenshots for EDA & route reconstruction comparisons
- [ ] `09_fastapi_backend_and_ui`: FastAPI inference & server-side evaluation backend (Laptop 1) + Leaflet/OpenStreetMap Web UI (Laptop 2)
- [ ] `10_deployment_and_e2e_test`: Two-laptop deployment scripts, live API+UI integration test, real UI + terminal log captures
- [ ] `11_technical_report_pdf`: Consolidated PDF Technical Report strictly matching `Deliverables.pdf` Sections 2–8
- [ ] `12_theory_and_viva_prep`: Comprehensive Theory Explanations & Viva Preparation guide in `theory_and_viva/`
