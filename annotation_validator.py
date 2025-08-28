import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from threading import Lock

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants and configuration
class AnnotationValidatorConfig:
    def __init__(self, threshold: float = 0.7, max_outliers: int = 10):
        self.threshold = threshold
        self.max_outliers = max_outliers

class AnnotationType(Enum):
    SPATIAL_RELATION = 1
    SPATIAL_VISUALIZATION = 2
    FLEXIBILITY_OF_CLOSURE = 3

@dataclass
class Annotation:
    id: int
    type: AnnotationType
    label: str
    confidence: float

class AnnotationValidatorException(Exception):
    pass

class AnnotationValidator:
    def __init__(self, config: AnnotationValidatorConfig):
        self.config = config
        self.lock = Lock()

    def compute_inter_annotator_agreement(self, annotations: List[Annotation]) -> float:
        """
        Compute the inter-annotator agreement using Cohen's Kappa statistic.

        Args:
        - annotations (List[Annotation]): A list of annotations from different annotators.

        Returns:
        - float: The inter-annotator agreement score.
        """
        with self.lock:
            try:
                # Extract labels from annotations
                labels = [annotation.label for annotation in annotations]
                # Compute Cohen's Kappa statistic
                kappa = stats.cohen_kappa_score(labels, labels)
                return kappa
            except Exception as e:
                logger.error(f"Error computing inter-annotator agreement: {e}")
                raise AnnotationValidatorException("Failed to compute inter-annotator agreement")

    def validate_annotation_quality(self, annotations: List[Annotation]) -> bool:
        """
        Validate the quality of annotations based on the inter-annotator agreement score.

        Args:
        - annotations (List[Annotation]): A list of annotations from different annotators.

        Returns:
        - bool: True if the annotation quality is valid, False otherwise.
        """
        with self.lock:
            try:
                # Compute inter-annotator agreement score
                kappa = self.compute_inter_annotator_agreement(annotations)
                # Check if the score is above the threshold
                return kappa >= self.config.threshold
            except Exception as e:
                logger.error(f"Error validating annotation quality: {e}")
                raise AnnotationValidatorException("Failed to validate annotation quality")

    def detect_outliers(self, annotations: List[Annotation]) -> List[Annotation]:
        """
        Detect outlier annotations based on the confidence scores.

        Args:
        - annotations (List[Annotation]): A list of annotations from different annotators.

        Returns:
        - List[Annotation]: A list of outlier annotations.
        """
        with self.lock:
            try:
                # Extract confidence scores from annotations
                confidences = [annotation.confidence for annotation in annotations]
                # Compute the mean and standard deviation of confidence scores
                mean = np.mean(confidences)
                std = np.std(confidences)
                # Identify outlier annotations
                outliers = [annotation for annotation, confidence in zip(annotations, confidences) if abs(confidence - mean) > 2 * std]
                return outliers
            except Exception as e:
                logger.error(f"Error detecting outliers: {e}")
                raise AnnotationValidatorException("Failed to detect outliers")

    def generate_consensus_labels(self, annotations: List[Annotation]) -> List[str]:
        """
        Generate consensus labels from a list of annotations.

        Args:
        - annotations (List[Annotation]): A list of annotations from different annotators.

        Returns:
        - List[str]: A list of consensus labels.
        """
        with self.lock:
            try:
                # Extract labels from annotations
                labels = [annotation.label for annotation in annotations]
                # Compute the most frequent label
                consensus_label = stats.mode(labels)[0][0]
                return [consensus_label] * len(annotations)
            except Exception as e:
                logger.error(f"Error generating consensus labels: {e}")
                raise AnnotationValidatorException("Failed to generate consensus labels")

    def create_validation_report(self, annotations: List[Annotation]) -> Dict:
        """
        Create a validation report for a list of annotations.

        Args:
        - annotations (List[Annotation]): A list of annotations from different annotators.

        Returns:
        - Dict: A dictionary containing the validation report.
        """
        with self.lock:
            try:
                # Compute inter-annotator agreement score
                kappa = self.compute_inter_annotator_agreement(annotations)
                # Validate annotation quality
                is_valid = self.validate_annotation_quality(annotations)
                # Detect outlier annotations
                outliers = self.detect_outliers(annotations)
                # Generate consensus labels
                consensus_labels = self.generate_consensus_labels(annotations)
                # Create validation report
                report = {
                    "inter_annotator_agreement": kappa,
                    "annotation_quality": is_valid,
                    "outliers": outliers,
                    "consensus_labels": consensus_labels
                }
                return report
            except Exception as e:
                logger.error(f"Error creating validation report: {e}")
                raise AnnotationValidatorException("Failed to create validation report")

def main():
    # Create a list of annotations
    annotations = [
        Annotation(1, AnnotationType.SPATIAL_RELATION, "label1", 0.8),
        Annotation(2, AnnotationType.SPATIAL_VISUALIZATION, "label2", 0.9),
        Annotation(3, AnnotationType.FLEXIBILITY_OF_CLOSURE, "label3", 0.7)
    ]

    # Create an annotation validator
    config = AnnotationValidatorConfig()
    validator = AnnotationValidator(config)

    # Compute inter-annotator agreement score
    kappa = validator.compute_inter_annotator_agreement(annotations)
    logger.info(f"Inter-annotator agreement score: {kappa}")

    # Validate annotation quality
    is_valid = validator.validate_annotation_quality(annotations)
    logger.info(f"Annotation quality is valid: {is_valid}")

    # Detect outlier annotations
    outliers = validator.detect_outliers(annotations)
    logger.info(f"Outlier annotations: {outliers}")

    # Generate consensus labels
    consensus_labels = validator.generate_consensus_labels(annotations)
    logger.info(f"Consensus labels: {consensus_labels}")

    # Create validation report
    report = validator.create_validation_report(annotations)
    logger.info(f"Validation report: {report}")

if __name__ == "__main__":
    main()