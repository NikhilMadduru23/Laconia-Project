# Laconia-Project: ATIS Chatbot

## Overview
The Laconia-Project aims to develop an intelligent chatbot for Airline Travel Information System (ATIS) queries, leveraging Rasa and DistilBERT. The chatbot is designed to assist users with tasks like finding flights, checking airfares, and identifying ground services using natural language processing (NLP).

### Project Goal
Develop a chatbot capable of:
- Classifying user intents (e.g., flight, airfare, airline).
- Extracting entities (e.g., cities, dates).
- Filling slots for dynamic, context-aware responses.

### Current Status
As of April 6, 2025, the project has completed foundational work (dataset preparation, model training setup) and designed a Rasa implementation with intent classification, entity extraction, and slot filling. The working code is in progress and will be fully implemented in [https://github.com/NikhilMadduru23/Laconia-Project](https://github.com/NikhilMadduru23/Laconia-Project).

## Technical Details

### Technology Stack
- **Programming Language**: Python 3.10
- **Framework**: Rasa 3.6.21 (planned)
- **Machine Learning**: DistilBERT (intent classification), DIETClassifier (entity extraction, planned)
- **Hardware**: NVIDIA RTX 4060 with CUDA 12.1 (optional CPU fallback)
- **Dependencies**: See `requirements.txt` for full list, key ones include:
  - `torch==2.5.1+cu121` (or CPU version if CUDA not supported)
  - `transformers==4.51.0`
  - `datasets==3.5.0`
  - `numpy==1.24.3`

### Achievements
#### Dataset Preparation
- **Source**: ATIS dataset (`data/processed/atis_train.csv`, `atis_test.csv`)
- **Intents**: 8 (abbreviation, aircraft, airfare, airline, flight, flight_time, ground_service, quantity)
- **Entities**: Cities (GPE), dates (DATE), airlines (ORG)
- **Processing**: Subsampled for efficient prototyping

#### Model Development
- **Model**: DistilBERT selected for intent classification
- **Training**: `transfer_learner.py` with CUDA support, ~99% accuracy on test set
- **Status**: Pre-trained model ready for Rasa integration

#### Environment Setup
- **Virtual Environment**: Created with Python 3.10 (`rasa_env`)
- **Dependencies**: Installed and documented in `requirements.txt`

### Rasa Implementation (In Progress)
- **Design**: 
  - Pipeline: `WhitespaceTokenizer`, `RegexFeaturizer`, `LexicalSyntacticFeaturizer`, `CountVectorsFeaturizer`, `DistilBertIntentClassifier`, `DIETClassifier`
  - Domain: 8 intents, 4 slots (`departure_city`, `destination_city`, `travel_date`, `airline_name`)
  - NLU: Annotated with entities (`data/nlu.yml`)
  - Rules: Simplified mappings (`data/rules.yml`)
  - Action: Slot filling (`actions/actions.py`)
- **Status**: Partially coded, full implementation pending

## Setup Instructions
1. **Clone the Repository**:
   ```bash
    git clone https://github.com/NikhilMadduru23/Laconia-Project.git
    cd Laconia-Project

2. **Create Virtual Environment**:
   ```bash
    py -3.10 -m venv rasa_env
    rasa_env\Scripts\activate

3. **Install Dependencies**:
  - ***For CUDA 12.1 Systems (e.g., RTX 4060):***:
      ```bash
      pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
      pip install -r requirements.txt

  - ***If CUDA 12.1 is Not Supported***:
      ```bash
      pip install torch==2.5.1 --no-cache-dir
      pip install -r requirements.txt

4. **Download Saved Models from Google Drive**:
   - Pre-trained `intent_classifier` model available at: [https://drive.google.com/drive/folders/1PTcZCWntpHZtVgpfKuVCqkG8TP0G2mHv?usp=drive_link](https://drive.google.com/drive/folders/1PTcZCWntpHZtVgpfKuVCqkG8TP0G2mHv?usp=drive_link)
   - Download the model files and extract them to the project root (e.g., `Laconia-Project/intent_classifier/`).
   - Ensure the Google Drive link permissions are set to "Anyone with the link can view" for accessibility.

5. **Train the Intent Classifier (Optional, if not using pre-trained model)**:
   - Ensure `data/processed/atis_train.csv` and `atis_test.csv` are in the `data/processed/` directory.
   - Run the training script:
     ```powershell
     python transfer_learner.py

