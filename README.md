# chicken-disaster-system-
Project Overview
Yeh project deep learning use karke chicken ki fecal images se 4 conditions detect karta hai:
ClassDescription🦠 CoccidiosisParasitic disease caused by Eimeria✅ HealthyNormal healthy chicken🦠 New Castle DiseaseViral disease (NDV)🦠 SalmonellaBacterial infection

🏗️ Project Architecture
Chicken-Disease-Classification/
├── src/cnnClassifier/
│   ├── components/
│   │   ├── data_ingestion.py         # Data download & extraction
│   │   ├── prepare_base_model.py     # VGG16 + custom head
│   │   ├── prepare_callbacks.py      # TensorBoard, Checkpoint, EarlyStopping
│   │   ├── model_trainer.py          # Training with augmentation
│   │   └── model_evaluation_mlflow.py # MLflow experiment tracking
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_trainer.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── predict.py               # Inference pipeline
│   ├── config/configuration.py       # Config manager
│   ├── entity/config_entity.py       # Dataclasses
│   ├── constants/__init__.py         # Path constants
│   └── utils/common.py              # Utility functions
├── config/config.yaml               # Project configuration
├── params.yaml                      # Model hyperparameters
├── dvc.yaml                         # DVC pipeline stages
├── main.py                          # Training orchestrator
├── app.py                           # Flask web application
├── templates/index.html             # Web UI
├── Dockerfile                       # Container setup
├── .github/workflows/main.yaml      # CI/CD pipeline
└── research/                        # Jupyter notebooks


