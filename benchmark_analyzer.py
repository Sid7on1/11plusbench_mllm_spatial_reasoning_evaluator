import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = 'config.json'
DATA_FILE = 'data.csv'
TASK_DISTRIBUTION_PLOT = 'task_distribution.png'
COMPLEXITY_STATISTICS_PLOT = 'complexity_statistics.png'
DATASET_SUMMARY_PLOT = 'dataset_summary.png'
BIAS_PATTERNS_PLOT = 'bias_patterns.png'
DATA_QUALITY_REPORT = 'data_quality_report.txt'

# Enum for task types
class TaskType(Enum):
    SPATIAL_RELATION = 1
    SPATIAL_VISUALIZATION = 2
    FLEXIBILITY_OF_CLOSURE = 3

# Data class for task distribution
@dataclass
class TaskDistribution:
    task_type: TaskType
    count: int

# Data class for complexity statistics
@dataclass
class ComplexityStatistics:
    mean: float
    std_dev: float
    min: float
    max: float

# Data class for dataset summary
@dataclass
class DatasetSummary:
    total_tasks: int
    task_distribution: List[TaskDistribution]
    complexity_statistics: ComplexityStatistics

# Data class for bias patterns
@dataclass
class BiasPatterns:
    task_type: TaskType
    bias: float

# Exception classes
class InvalidConfigError(Exception):
    pass

class InvalidDataError(Exception):
    pass

# Lock for thread safety
lock = Lock()

class BenchmarkAnalyzer:
    def __init__(self, config_file: str = CONFIG_FILE, data_file: str = DATA_FILE):
        """
        Initialize the benchmark analyzer.

        Args:
        - config_file (str): The configuration file path.
        - data_file (str): The data file path.
        """
        self.config_file = config_file
        self.data_file = data_file
        self.config = self.load_config()
        self.data = self.load_data()

    def load_config(self) -> Dict:
        """
        Load the configuration from the configuration file.

        Returns:
        - config (Dict): The loaded configuration.
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Config file '{self.config_file}' not found.")
            raise InvalidConfigError(f"Config file '{self.config_file}' not found.")

    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the data file.

        Returns:
        - data (pd.DataFrame): The loaded data.
        """
        try:
            data = pd.read_csv(self.data_file)
            return data
        except FileNotFoundError:
            logger.error(f"Data file '{self.data_file}' not found.")
            raise InvalidDataError(f"Data file '{self.data_file}' not found.")

    def analyze_task_distribution(self) -> List[TaskDistribution]:
        """
        Analyze the task distribution.

        Returns:
        - task_distribution (List[TaskDistribution]): The task distribution.
        """
        with lock:
            task_distribution = []
            for task_type in TaskType:
                count = self.data[self.data['task_type'] == task_type.value].shape[0]
                task_distribution.append(TaskDistribution(task_type, count))
            return task_distribution

    def compute_complexity_statistics(self) -> ComplexityStatistics:
        """
        Compute the complexity statistics.

        Returns:
        - complexity_statistics (ComplexityStatistics): The complexity statistics.
        """
        with lock:
            complexity = self.data['complexity']
            mean = complexity.mean()
            std_dev = complexity.std()
            min = complexity.min()
            max = complexity.max()
            return ComplexityStatistics(mean, std_dev, min, max)

    def generate_dataset_summary(self) -> DatasetSummary:
        """
        Generate the dataset summary.

        Returns:
        - dataset_summary (DatasetSummary): The dataset summary.
        """
        with lock:
            total_tasks = self.data.shape[0]
            task_distribution = self.analyze_task_distribution()
            complexity_statistics = self.compute_complexity_statistics()
            return DatasetSummary(total_tasks, task_distribution, complexity_statistics)

    def identify_bias_patterns(self) -> List[BiasPatterns]:
        """
        Identify the bias patterns.

        Returns:
        - bias_patterns (List[BiasPatterns]): The bias patterns.
        """
        with lock:
            bias_patterns = []
            for task_type in TaskType:
                bias = self.data[self.data['task_type'] == task_type.value]['bias'].mean()
                bias_patterns.append(BiasPatterns(task_type, bias))
            return bias_patterns

    def create_data_quality_report(self) -> str:
        """
        Create the data quality report.

        Returns:
        - report (str): The data quality report.
        """
        with lock:
            report = ""
            dataset_summary = self.generate_dataset_summary()
            report += f"Total tasks: {dataset_summary.total_tasks}\n"
            report += "Task distribution:\n"
            for task_distribution in dataset_summary.task_distribution:
                report += f"  - {task_distribution.task_type.name}: {task_distribution.count}\n"
            report += f"Complexity statistics: mean={dataset_summary.complexity_statistics.mean}, std_dev={dataset_summary.complexity_statistics.std_dev}, min={dataset_summary.complexity_statistics.min}, max={dataset_summary.complexity_statistics.max}\n"
            bias_patterns = self.identify_bias_patterns()
            report += "Bias patterns:\n"
            for bias_pattern in bias_patterns:
                report += f"  - {bias_pattern.task_type.name}: {bias_pattern.bias}\n"
            return report

    def plot_task_distribution(self) -> None:
        """
        Plot the task distribution.
        """
        with lock:
            task_distribution = self.analyze_task_distribution()
            labels = [task_distribution.task_type.name for task_distribution in task_distribution]
            sizes = [task_distribution.count for task_distribution in task_distribution]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Task Distribution')
            plt.savefig(TASK_DISTRIBUTION_PLOT)

    def plot_complexity_statistics(self) -> None:
        """
        Plot the complexity statistics.
        """
        with lock:
            complexity_statistics = self.compute_complexity_statistics()
            plt.hist(self.data['complexity'], bins=10)
            plt.title('Complexity Statistics')
            plt.xlabel('Complexity')
            plt.ylabel('Frequency')
            plt.savefig(COMPLEXITY_STATISTICS_PLOT)

    def plot_dataset_summary(self) -> None:
        """
        Plot the dataset summary.
        """
        with lock:
            dataset_summary = self.generate_dataset_summary()
            plt.bar([task_distribution.task_type.name for task_distribution in dataset_summary.task_distribution], [task_distribution.count for task_distribution in dataset_summary.task_distribution])
            plt.title('Dataset Summary')
            plt.xlabel('Task Type')
            plt.ylabel('Count')
            plt.savefig(DATASET_SUMMARY_PLOT)

    def plot_bias_patterns(self) -> None:
        """
        Plot the bias patterns.
        """
        with lock:
            bias_patterns = self.identify_bias_patterns()
            plt.bar([bias_pattern.task_type.name for bias_pattern in bias_patterns], [bias_pattern.bias for bias_pattern in bias_patterns])
            plt.title('Bias Patterns')
            plt.xlabel('Task Type')
            plt.ylabel('Bias')
            plt.savefig(BIAS_PATTERNS_PLOT)

    def save_data_quality_report(self) -> None:
        """
        Save the data quality report.
        """
        with lock:
            report = self.create_data_quality_report()
            with open(DATA_QUALITY_REPORT, 'w') as f:
                f.write(report)

def main() -> None:
    analyzer = BenchmarkAnalyzer()
    analyzer.plot_task_distribution()
    analyzer.plot_complexity_statistics()
    analyzer.plot_dataset_summary()
    analyzer.plot_bias_patterns()
    analyzer.save_data_quality_report()

if __name__ == '__main__':
    main()