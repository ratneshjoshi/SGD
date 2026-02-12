# Saliency Guided Debiasing (SGD)

**Detecting and mitigating biases in Language Models using feature attribution**

**Authors:** Joshi, Ratnesh Kumar, Arindam Chatterjee, and Asif Ekbal

**Journal:** Neurocomputing 563 (2024): 126851. (h-5 index = 135)

## Overview

This repository contains the implementation of Saliency Guided Debiasing, a method for detecting and mitigating biases in language models using feature attribution techniques. The main contribution is in the `gpt2-persona` folder, which demonstrates bias detection and mitigation through persona-based dialogue generation.

## Repository Structure

- **gpt2-persona/**: Main contribution - Persona-enhanced debiasing approach
  - `train.py`: Training script for the persona model
  - `Evaluate.py`: Evaluation script for bias detection
  - `create_data.py`: Data preparation script
  - `data/`: Contains personality and dialogue datasets

- **debiasing_by_data_augmentation/**: Data augmentation approach for debiasing
  - `data_augmentation.ipynb`: Jupyter notebook for data augmentation
  - `train.py`: Training with augmented data
  - `Evaluate.py`: Evaluation metrics
  - `data/`: Augmented and debiased datasets

- **persona_data/**: Additional persona data resources

## Prerequisites

```bash
pip install transformers>=4.0.0
pip install datasets>=2.0.0
pip install torch>=1.9.0
pip install pandas>=1.0.0
pip install numpy>=1.19.0
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the Code

### 1. Persona-Based Debiasing (Main Contribution)

#### Training

```bash
cd gpt2-persona
python train.py
```

**Key Hyperparameters in `train.py`:**
- **Model:** `gpt2-large`
- **Epochs:** 30
- **Batch Size:** 8 (per device, train and eval)
- **Learning Rate:** 2e-5
- **Weight Decay:** 0.01
- **Block Size (Sequence Length):** 256
- **Evaluation Strategy:** Every epoch
- **Save Strategy:** Only save checkpoints, don't load best model
- **Max Tokens Saved:** 2 (save last 2 checkpoints)
- **Number of Workers (Data Loading):** 4

**Special Tokens Used:**
- `[CONTEXT]`, `[END_OF_CONTEXT]`
- `[speaker1]`, `[speaker2]`
- `[RESPONSE]`, `[END_OF_RESPONSE]`
- `[PERSONA]`

#### Evaluation

```bash
cd gpt2-persona
python Evaluate.py
```

This script generates responses across different demographic groups and occupations to test bias. Key aspects:

- **Occupations:** 36 different occupations (software developer, nurse, doctor, lawyer, CEO, cook, poet, actor, etc.)
- **Descriptors:** name, age, race, ethnicity, sexuality, class, height
- **Demographics:** African-American, Asian, Black, Hispanic, Indian, Pacific-Islander, White, Straight, Bisexual, Gay (10 categories)
- **Genders:** Person, Male, Female, Transgender, Nonbinary (5 categories)
- **Context Types:** 3 different dialogue templates ("They are", "He is", "She is")
- **Total Test Cases:** ~127,800 combinations (36 × 7 × 3 × 10 × 5)

Output is saved to `results-gpt-full.csv` containing:
- `setting`: Metadata about the test case
- `Input`: Full prompt with persona and context
- `response`: Model-generated response

**Inference Hyperparameters:**
- **Sampling Method:** top_k sampling with k=0 (nucleus/temperature sampling)
- **Max Length:** input_length + 50 tokens
- **Device:** CUDA if available, else CPU
- **Pad Token ID:** EOS token

### 2. Data Augmentation Approach

#### Generate Augmented Data

```bash
cd debiasing_by_data_augmentation
jupyter notebook data_augmentation.ipynb
```

The notebook performs the following steps:
1. Reads `gendered_words.txt` containing paired male-female words
2. Loads original training/validation data from CSV
3. Swaps male words with female words to create debiased versions
4. Saves augmented datasets: `train_augmented.csv`, `val_augmented.csv`, `train_debiased.csv`, `val_debiased.csv`

#### Train with Augmented Data

```bash
cd debiasing_by_data_augmentation
python train.py
```

**Key Hyperparameters:**
- **Model:** `gpt2-large`
- **Training Data:** `data/train_augmented.csv`
- **Validation Data:** `data/val_augmented.csv`
- **Epochs:** 30
- **Batch Size:** 8 (per device, train and eval)
- **Learning Rate:** 2e-5
- **Weight Decay:** 0.01
- **Block Size:** 256
- **Evaluation Strategy:** Every epoch
- **Save Strategy:** Save every epoch, load best model at end
- **Number of Workers:** 4

#### Evaluate Debiasing Effect

```bash
cd debiasing_by_data_augmentation
python Evaluate.py
```

Generates a comprehensive bias evaluation report testing the effectiveness of data augmentation debiasing. Output saved to `results.csv`.

## Data Format

Training data should be in CSV format with a "text" column. The format follows this template:

```
[PERSONA] <demographic_group> <gender> [CONTEXT][speaker1] <utterance1> [speaker2] <utterance2> [speaker1] <utterance3> [END_OF_CONTEXT] [RESPONSE] [speaker2] <response_text> [END_OF_RESPONSE]
```

**Example:**
```
[PERSONA] I am a White Female [CONTEXT][speaker1] I met the doctor today [speaker2] She is a good doctor [speaker1] What is the doctor's name? [END_OF_CONTEXT] [RESPONSE] [speaker2] Her name is Sarah [END_OF_RESPONSE]
```

## Output Files

- `gpt2-large-persona/`: Trained model weights, config, and tokenizer for persona approach
- `results-gpt-full.csv`: Complete bias evaluation results for persona approach
- `results.csv`: Bias evaluation results for augmented data approach
- Training logs and checkpoints in the output directory

## Key Features

1. **Persona-Enhanced Generation:** Explicitly incorporates demographic information and occupations to detect model biases
2. **Data Augmentation Debiasing:** Uses systematic word-swapping to create balanced datasets
3. **Comprehensive Bias Evaluation:** Tests across multiple dimensions:
   - 36 occupations covering various sectors
   - 7 personal descriptors
   - 10 demographic groups
   - 5 gender categories
   - 3 dialogue contexts
4. **Multi-lingual Support:** Leverages HuggingFace transformers for language flexibility
5. **Flexible Inference:** Supports both CPU and GPU-accelerated inference

## Configuration Notes

### Hardware Requirements

- **GPU Memory:** 8GB+ VRAM (GPT2-large requires ~6GB minimum)
- **RAM:** 16GB+ recommended for data processing
- **Disk Space:** ~2GB for models and datasets

### Performance Estimates

- **Training Time:** 
  - 30 epochs: ~5-10 hours (RTX 3080)
  - ~15-20 hours (single GPU, mid-range)
  - ~1-2 days (CPU-only)
- **Evaluation Time:**
  - Full bias evaluation: ~1-2 hours (GPU)
  - ~3-4 hours (CPU)
- **Data Processing:** Uses `num_proc=4` for parallel processing (adjust based on available CPU cores)

### Customization Tips

- **Reduce Training Time:** Decrease `num_train_epochs` to 10-15
- **Batch Size:** Increase to 16 if you have >12GB VRAM
- **Block Size:** Can reduce to 128 for faster training on memory-constrained setups
- **Evaluation Scale:** Comment out unwanted occupation/descriptor/demographic lists in `Evaluate.py` to reduce test cases

## Dependencies Version Notes

- **Transformers:** Version 4.0.0+ required for GPT2-large support
- **PyTorch:** Version 1.9.0+, CuDA 11.8+ for GPU support
- **Datasets:** Version 2.0.0+ for HuggingFace dataset API
- **Python:** 3.7+ (recommend 3.9+)
