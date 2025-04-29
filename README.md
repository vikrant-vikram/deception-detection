Here's a comprehensive `README.md` file for your Diplomacy Detection project:


Checkpoints Drive Link:
https://drive.google.com/file/d/1MGLtRMYIfsEEVG_e2OvjqL4dcTDPdEcr/view?usp=sharing


```markdown
# Diplomacy Detection: Lie Detection in Diplomacy Game Chats

This project implements a neural network model to detect lies in Diplomacy game chat messages using BERT and LSTM architectures with power-specific embeddings.

## Features

- **Contextual Analysis**: Uses previous messages as context for better prediction
- **Power-Specific Embeddings**: Incorporates player nation information
- **Two Detection Tasks**:
  - `ACTUAL_LIE`: Detects actual lies (sender perspective)
  - `SUSPECTED_LIE`: Detects suspected lies (receiver perspective)
- **Checkpointing**: Automatically saves model checkpoints after each epoch
- **Comprehensive Evaluation**: Provides accuracy, F1, precision, recall metrics
- **Label-Specific Analysis**: Can evaluate performance on truthful/deceptive messages separately

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- scikit-learn
- tqdm
- matplotlib

Install requirements:
```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

## Data Format

Input JSONL files should contain messages in this format:

```json
{
  "messages": ["message1", "message2", ...],
  "speakers": ["power1", "power2", ...],
  "sender_labels": [true/false ...],
  "receiver_labels": [true/false/"NOANNOTATION", ...]
}
```

## Usage

### Training
```bash
python diplomacy_detection.py
```
Select 't' for training when prompted.

### Inference
```bash
python diplomacy_detection.py
```
Select 'i' for inference when prompted, then choose which messages to evaluate:
- truthful
- deceptive
- all


## Model Architecture

1. **BERT Encoder**: Processes the combined context and target message
2. **Bidirectional LSTM**: Captures sequential patterns in BERT outputs
3. **Power Embeddings**: Learns nation-specific representations
4. **Classifier Head**: Makes final lie/truth prediction

```
Input Text → BERT → LSTM → +PowerEmb → Classifier → Prediction
```

## Checkpoints

The system automatically saves:
- Best model (`best_model_ACTUAL_LIE.pt`)
- Epoch checkpoints (`model_checkpoints/checkpoint_epoch_X.pt`)
- Training history plots (`training_history_ACTUAL_LIE.png`)

## Results

Typical performance metrics:

| Task          | Accuracy | F1 Score | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| ACTUAL_LIE    |0.9990     |0.9882   |0.9855      | 0.9909 |

## Configuration

Modify `Config` class in the code to adjust:
- Model hyperparameters
- Training settings
- Input lengths
- Batch sizes
