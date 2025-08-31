# QR-SAC Implementation for Assetto Corsa Gym  

## Overview  
This project extends the [Assetto Corsa Gym](https://[github.com/dasGringuen/assetto_corsa_gym)) framework by implementing **Quantile Regression Soft Actor-Critic (QR-SAC)** on top of the existing Soft Actor-Critic (SAC) baseline. The aim was to evaluate performance, stability, and risk-awareness of QR-SAC compared to SAC, with a focus on autonomous racing where reliability is often more critical than raw speed.  

## Key Contributions  
- **QR-SAC Implementation**: Added QR-SAC to the existing SAC codebase.  
- **Unified Hyperparameter Management**: Updated `config.yml` to ensure consistent hyperparameters across experiments.  
- **Critic Network Modification**: Altered `network.py` to incorporate quantile regression in the critic network.  
- **Backward Compatibility**: Updated `load_model` to allow models trained with SAC to remain compatible.  
- **Algorithm Selection**: Modified `train.py` to support selection between SAC and QR-SAC.  
- **Streamlined Configuration**: Added track and car specifications to `config.yml`, removing the need to pass them manually through the command line.  

## Usage  
1. Install and set up [Assetto Corsa Gym]([https://github.com/BakhshiZ/assetto_corsa_gym]) (ensure the environment runs correctly).
2. Download libraries in requirements.txt
3. Configure hyperparameters, algorithm, track, and car (if desired) in `config.yml`.  
4. Run training with `python train.py --algo [qrsac | sac] --config config.yml`.    
5. Load previously trained models (SAC or QR-SAC) using the updated `python train.py --test --load_path <path_to_model> --config config.yml`.  
