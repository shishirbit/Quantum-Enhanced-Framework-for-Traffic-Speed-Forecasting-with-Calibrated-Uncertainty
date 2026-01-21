Quantum-Enhanced-Framework-for-Traffic-Speed-Forecasting-with-Calibrated-Uncertainty/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
│
├── configs/
│   ├── metr_la.yaml
│   ├── quantum.yaml
│   └── uncertainty.yaml
│
├── data/
│   ├── METR-LA/
│   │   ├── adjacency.pkl
│   │   ├── speed.npy
│   │   └── README.md
│   └── custom_dataset_template/
│
├── src/
│   ├── models/
│   │   ├── st_gat.py
│   │   ├── st_transformer.py
│   │   ├── q_stgtf.py
│   │   └── quantum_layer.py
│   │
│   ├── training/
│   │   ├── train_phase1.py
│   │   ├── train_phase2.py
│   │   └── train_phase3.py
│   │
│   ├── uncertainty/
│   │   ├── mc_dropout.py
│   │   ├── ensemble.py
│   │   ├── quantile.py
│   │   └── conformal.py
│   │
│   ├── data_utils/
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   └── adjacency.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── evaluate.py
│
├── scripts/
│   ├── run_phase1.sh
│   ├── run_phase2.sh
│   ├── run_phase3.sh
│   └── run_uncertainty.sh
│
└── results/
    └── paper_tables/
