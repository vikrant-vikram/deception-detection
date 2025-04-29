# **5. Deception Detection [HARD]**

# - **Link to the dataset:** [[DATASET]](https://github.com/DenisPeskoff/2020_acl_diplomacy/tree/master/data)
# - **References:** [[TASK DESCRIPTION]](https://sites.google.com/view/qanta/projects/diplomacy?pli=1)
# - **Task explanation:** The QANTA Diplomacy project involves developing a model that predicts whether messages exchanged between players in the game **Diplomacy** are deceptive or truthful. The model analyzes in-game conversations and associated metadata to make its predictions. Its performance is evaluated based on how accurately it identifies deceptive and truthful messages.
# - **Evaluation Metric: The evaluation metric used is accuracy.**




# import all reuired lobrairies

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                            precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('diplomacy_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




# These are the configuration to train thr  model??
class Config:
    RANDOM_SEED: int = 42
    MAX_LEN: int = 128 
    BATCH_SIZE: int = 32 
    EPOCHS: int = 10 
    LEARNING_RATE: float = 2e-5
    NUM_CONTEXT: int = 3 
    POWER_EMBED_DIM: int = 32 
    LSTM_HIDDEN_DIM: int = 256 
    DROPOUT_PROB: float = 0.2
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME: str = 'bert-base-uncased'
    
    @staticmethod
    def set_seed():
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_SEED)

Config.set_seed()



# Enhanced dataset class with caching and better context handling

# Handles both "ACTUAL_LIE" (sender's intent) and "SUSPECTED_LIE" (receiver's perception) tasks
# Includes contextual messages (configurable number)
# Caches processed data for faster subsequent loads
# Adds speaker tags to messages for better context understanding


class DiplomacyContextDataset(Dataset):
    def __init__(
        self, 
        data: List[Dict],
        tokenizer: BertTokenizer,
        power_to_idx: Dict[str, int],
        max_len: int = Config.MAX_LEN,
        task: str = "ACTUAL_LIE",
        cache_dir: str = "data_cache"
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.power_to_idx = power_to_idx
        self.task = task
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        cache_file = self.cache_dir / f"{task}_samples.pt"
        if cache_file.exists():
            logger.info(f"Loading cached samples from {cache_file}")
            self.samples = torch.load(cache_file)
        else:
            self._prepare_samples(data)
            torch.save(self.samples, cache_file)
            logger.info(f"Cached samples saved to {cache_file}")

    def _prepare_samples(self, data: List[Dict]):
        logger.info(f"Preparing samples for task: {self.task}")
        for item in tqdm(data, desc="Processing data"):
            messages = item['messages']
            speakers = item['speakers']
            sender_labels = item['sender_labels']
            receiver_labels = item['receiver_labels']
            
            for i in range(Config.NUM_CONTEXT, len(messages)):
                context = messages[i-Config.NUM_CONTEXT:i]
                context_speakers = speakers[i-Config.NUM_CONTEXT:i]
                target = messages[i]
                target_speaker = speakers[i]

                if self.task == 'ACTUAL_LIE':
                    label = sender_labels[i]
                    if label == "NOANNOTATION":
                        continue
                    label = 1 if label == False else 0
                elif self.task == 'SUSPECTED_LIE':
                    label = receiver_labels[i]
                    if label == "NOANNOTATION":
                        continue
                    label = 1 if label else 0
                else:
                    raise ValueError(f"Unknown task: {self.task}")
                
                self.samples.append({
                    'context': context,
                    'context_speakers': context_speakers,
                    'target': target,
                    'target_speaker': target_speaker,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Enhanced context handling with speaker tags
        context_with_speakers = [
            f"[{speaker}] {msg}" 
            for speaker, msg in zip(sample['context_speakers'], sample['context'])
        ]
        context = " ".join(context_with_speakers)
        target = f"[{sample['target_speaker']}] {sample['target']}"
        
        combined_input = f"{context} [SEP] {target}"
        
        encoding = self.tokenizer.encode_plus(
            combined_input,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'power': torch.tensor(self.power_to_idx[sample['target_speaker']], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'original_text': sample['target']  # For interpretability
        }
    


# BERT Base: Extracts deep contextual word embeddings
# Power Embeddings: Learns representations for each diplomatic power (country)
# BiLSTM: Captures sequential patterns in the conversation
# Classifier Head: Makes final deception predictions

class ContextPowerModel(nn.Module):
    def __init__(self, n_powers, power_dim=32, lstm_dim=256, dropout_prob=0.2):
        super().__init__()
        
        # BERT base model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dim = self.bert.config.hidden_size
        
        # Power embedding with normalization
        # Each diplomatic power (e.g., France, England, Germany) gets assigned a unique integer index
        # The embedding layer converts this index into a dense vector representation
        # Layer normalization stabilizes the embeddings
        # Dropout prevents over-reliance on specific power features
        self.power_embedding = nn.Sequential(
            nn.Embedding(n_powers, power_dim),
            nn.LayerNorm(power_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=lstm_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if 2 > 1 else 0  # Dropout only between layers
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim*2 + power_dim, lstm_dim),
            nn.LayerNorm(lstm_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_dim, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Proper initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                param.data[self.lstm.hidden_size:2*self.lstm.hidden_size] = 1
                
        for module in [self.power_embedding, self.classifier]:
            if hasattr(module, 'weight') and module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, power):
        # BERT processing
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        sequence_output = bert_output.last_hidden_state
        
        # LSTM processing with packed sequence
        lengths = attention_mask.sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            sequence_output, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, (hidden, _) = self.lstm(packed_input)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Get power embeddings
        power_embed = self.power_embedding(power)
        
        # Combine features
        combined = torch.cat((hidden, power_embed), dim=1)
        
        return self.classifier(combined)

# Loss (CrossEntropy)
# ↑
# Classifier Weights
# ↑
# Combined Features ───┐
# ↑                    ↑
# LSTM Gradients    Power Embedding Gradients

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=Config.DEVICE):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = Path('model_checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train_epoch(self, loader: DataLoader, grad_accum_steps: int = 1) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct = 0, 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            power = batch['power'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                outputs = self.model(input_ids, attention_mask, power)
                loss = self.loss_fn(outputs, labels) / grad_accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            steps += 1
        
        return total_loss / steps, total_correct / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> Dict[str, Union[float, np.ndarray]]:
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                power = batch['power'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, power)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with additional metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_acc': val_metrics['accuracy']
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)


def load_jsonl(filepath: str) -> List[Dict]:
    try:
        with open(filepath) as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise



# Input: A list of dictionaries (presumably loaded from your JSONL files), where each dictionary represents one data point (e.g., a message or turn in the game).
# Output: A dictionary that maps each speaker (or “power”) to a unique integer index.
                                                

def get_power_map(data: List[Dict]) -> Dict[str, int]:
    all_speakers = set()
    for d in data:
        try:
            all_speakers.update(d['speakers'])
        except KeyError:
            logger.warning("Missing 'speakers' key in data item")
    return {p: i for i, p in enumerate(sorted(all_speakers))}

def plot_training_history(history: Dict[str, List[float]], task: str):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'Loss ({task})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'Accuracy ({task})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1 score plot
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title(f'Validation F1 Score ({task})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{task}.png')
    plt.close()

def run_label_specific_inference(
    model_path: str,
    input_data: Union[str, List[Dict]],
    power_to_idx: Dict[str, int],
    task: str = "ACTUAL_LIE",
    label_filter: str = "all",  # "truthful", "deceptive", or "all"
    num_examples: int = 5
) -> Dict:
    logger.info(f"Running inference for {label_filter} messages")
    
    # Load data if input is a file path
    if isinstance(input_data, str):
        input_data = load_jsonl(input_data)
    
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = ContextPowerModel(len(power_to_idx)).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    
    # Create dataset
    dataset = DiplomacyContextDataset(input_data, tokenizer, power_to_idx, task=task)
    
    # Filter samples by label 
    if label_filter != "all":
        label_value = 1 if label_filter == "deceptive" else 0
        filtered_samples = [
            sample for sample in dataset.samples 
            if sample['label'] == label_value
        ]
        dataset.samples = filtered_samples
    
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=lambda b: {
            'input_ids': torch.stack([item['input_ids'] for item in b]),
            'attention_mask': torch.stack([item['attention_mask'] for item in b]),
            'power': torch.stack([item['power'] for item in b]),
            'label': torch.stack([item['label'] for item in b]),
            'original_text': [item['original_text'] for item in b]
        }
    )
    
    # Run inference
    all_preds = []
    all_labels = []
    all_texts = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {label_filter} messages"):
            inputs = {
                'input_ids': batch['input_ids'].to(Config.DEVICE),
                'attention_mask': batch['attention_mask'].to(Config.DEVICE),
                'power': batch['power'].to(Config.DEVICE)
            }
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            all_texts.extend(batch['original_text'])
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'label_filter': label_filter,
        'num_samples': len(all_labels),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'examples': []
    }
    
    # Add example predictions
    for i in range(min(num_examples, len(all_texts))):
        metrics['examples'].append({
            'text': all_texts[i],
            'true_label': "Deceptive" if all_labels[i] == 1 else "Truthful",
            'predicted_label': "Deceptive" if all_preds[i] == 1 else "Truthful",
            'confidence': float(all_probs[i][all_preds[i]]),
            'probabilities': {
                'truthful': float(all_probs[i][0]),
                'deceptive': float(all_probs[i][1])
            }
        })
    
    return metrics

def main(task: str = "ACTUAL_LIE"):
    logger.info(f"Starting training for task: {task}")
    
    # Load data
    train_data = load_jsonl('train.jsonl')
    val_data = load_jsonl('validation.jsonl')
    test_data = load_jsonl('test.jsonl')

    # Create power mapping
    power_to_idx = get_power_map(train_data + val_data + test_data)
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create datasets with caching
    train_dataset = DiplomacyContextDataset(train_data, tokenizer, power_to_idx, task=task)
    val_dataset = DiplomacyContextDataset(val_data, tokenizer, power_to_idx, task=task)
    test_dataset = DiplomacyContextDataset(test_data, tokenizer, power_to_idx, task=task)

    # Create data loaders with collate_fn for dynamic padding
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'power': torch.stack([item['power'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch]),
            'original_text': [item['original_text'] for item in batch]
        }

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn
    )

    # Initialize model and training components
    model = ContextPowerModel(len(power_to_idx))
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=0.01  # Added weight decay
    )
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)

    # Training loop with early stopping
    best_f1 = 0
    no_improve = 0
    patience = 3  # Early stopping patience
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(Config.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        # Train and validate
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Check for improvement and save checkpoints
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            no_improve = 0
            torch.save(model.state_dict(), f'best_model_{task}.pt')
            logger.info("Saved new best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"No improvement for {patience} epochs, stopping early")
                break
        
        # Save checkpoint after each epoch
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
        logger.info(f"Saved checkpoint for epoch {epoch + 1}")

    # Final evaluation
    model.load_state_dict(torch.load(f'best_model_{task}.pt'))
    test_metrics = trainer.evaluate(test_loader)
    
    logger.info(f"\nTEST RESULTS ({task}):")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Plot training history
    plot_training_history(history, task)

    return test_metrics





if __name__ == '__main__':
    print("Project will run for ACTUAL_LIE task")
    action = input("Training or inference? (t/i): ").lower()
    task = "ACTUAL_LIE"
    if action == "t":
        main(task)
    elif action == "i":
        label_choice = input("Which messages to test? (truthful/deceptive/all): ").lower()
        # Validate label choice
                # Ask which messages the user wants to test the model on:
        # 'truthful' = only test truthful messages,
        # 'deceptive' = only test deceptive messages,
        # 'all' = test everything

        while label_choice not in ["truthful", "deceptive", "all"]:
            print("Invalid choice. Please enter 'truthful', 'deceptive', or 'all'")
            label_choice = input("Which messages to test? (truthful/deceptive/all): ").lower()
        
        # Load test data and power mapping
        test_data = load_jsonl('test.jsonl')
        power_to_idx = get_power_map(test_data)
        


        # Run the inference pipeline:
        # This loads the trained model, runs it on the filtered test data,
        # and returns performance metrics and sample predictions

        results = run_label_specific_inference(
            model_path=f'best_model_{task}.pt',
            input_data=test_data,
            power_to_idx=power_to_idx,
            task=task,
            label_filter=label_choice
        )
        
        # Print results
        print(f"\nResults for {label_choice} messages ({task}):")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
        
        # Print example predictions
        print("\nExample predictions:")
        for example in results['examples']:
            print(f"\nText: {example['text']}")
            print(f"True: {example['true_label']}, Predicted: {example['predicted_label']}")
            print(f"Confidence: {example['confidence']:.4f}")
            print(f"Probabilities: Truthful={example['probabilities']['truthful']:.4f}, "
                  f"Deceptive={example['probabilities']['deceptive']:.4f}")