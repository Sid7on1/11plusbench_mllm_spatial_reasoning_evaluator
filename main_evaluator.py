import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from typing import List, Dict, Tuple
from collections import defaultdict
import logging
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLLMEvaluator:
    """
    Central orchestrator for running complete MLLM evaluation pipeline on 11PLUS-BENCH.
    """
    def __init__(self, config: Dict):
        """
        Initialize the evaluator with the given configuration.

        Args:
        - config (Dict): Configuration dictionary containing model, dataset, and evaluation settings.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        """
        Load the pre-trained MLLM model and tokenizer.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model.to(self.device)

    def load_benchmark_data(self, data_path: str) -> Tuple[List, List]:
        """
        Load the 11PLUS-BENCH dataset from the given data path.

        Args:
        - data_path (str): Path to the dataset file.

        Returns:
        - texts (List): List of text samples.
        - labels (List): List of corresponding labels.
        """
        data = pd.read_csv(data_path)
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        return texts, labels

    def evaluate_model_batch(self, batch: List) -> List:
        """
        Evaluate the model on a batch of text samples.

        Args:
        - batch (List): List of text samples.

        Returns:
        - predictions (List): List of predicted labels.
        """
        inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy()

    def compute_metrics(self, predictions: List, labels: List) -> Dict:
        """
        Compute evaluation metrics (accuracy, classification report, confusion matrix).

        Args:
        - predictions (List): List of predicted labels.
        - labels (List): List of true labels.

        Returns:
        - metrics (Dict): Dictionary containing evaluation metrics.
        """
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        matrix = confusion_matrix(labels, predictions)
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': matrix
        }
        return metrics

    def save_results(self, results: Dict, output_path: str):
        """
        Save the evaluation results to the given output path.

        Args:
        - results (Dict): Dictionary containing evaluation metrics.
        - output_path (str): Path to save the results.
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from the evaluation results.

        Args:
        - results (Dict): Dictionary containing evaluation metrics.

        Returns:
        - report (str): Human-readable report.
        """
        report = f"Accuracy: {results['accuracy']:.4f}\n"
        report += results['classification_report']
        report += f"Confusion Matrix:\n{results['confusion_matrix']}"
        return report

    def run_evaluation(self, data_path: str, output_path: str):
        """
        Run the complete evaluation pipeline.

        Args:
        - data_path (str): Path to the dataset file.
        - output_path (str): Path to save the results.
        """
        texts, labels = self.load_benchmark_data(data_path)
        predictions = []
        for batch in tqdm([texts[i:i+32] for i in range(0, len(texts), 32)]):
            batch_predictions = self.evaluate_model_batch(batch)
            predictions.extend(batch_predictions)
        metrics = self.compute_metrics(predictions, labels)
        self.save_results(metrics, output_path)
        report = self.generate_report(metrics)
        logger.info(report)

def main():
    config = {
        'model_name': 'bert-base-uncased',
        'data_path': 'data/11plus-bench.csv',
        'output_path': 'results/evaluation_results.json'
    }
    evaluator = MLLMEvaluator(config)
    evaluator.run_evaluation(config['data_path'], config['output_path'])

if __name__ == '__main__':
    main()