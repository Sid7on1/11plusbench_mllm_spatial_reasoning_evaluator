import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from git import Repo
from git.exc import GitCommandError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReproducibilityKit:
    """
    Ensures full reproducibility of evaluation results.

    Attributes:
        experiment_dir (Path): Directory where experiment results are stored.
        config (Dict): Configuration dictionary.
    """

    def __init__(self, experiment_dir: Path, config: Dict):
        """
        Initializes the ReproducibilityKit.

        Args:
            experiment_dir (Path): Directory where experiment results are stored.
            config (Dict): Configuration dictionary.
        """
        self.experiment_dir = experiment_dir
        self.config = config

    def compute_result_checksums(self, result_files: List[Path]) -> Dict[Path, str]:
        """
        Computes the checksums of the result files.

        Args:
            result_files (List[Path]): List of result files.

        Returns:
            Dict[Path, str]: Dictionary mapping result files to their checksums.
        """
        checksums = {}
        for file in result_files:
            try:
                with open(file, 'rb') as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                    checksums[file] = checksum
            except Exception as e:
                logger.error(f"Error computing checksum for {file}: {e}")
        return checksums

    def save_experiment_state(self, experiment_state: Dict) -> None:
        """
        Saves the experiment state to a file.

        Args:
            experiment_state (Dict): Dictionary containing the experiment state.
        """
        try:
            with open(self.experiment_dir / 'experiment_state.json', 'w') as f:
                json.dump(experiment_state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving experiment state: {e}")

    def validate_reproducibility(self, result_files: List[Path], expected_checksums: Dict[Path, str]) -> bool:
        """
        Validates the reproducibility of the experiment by checking the checksums of the result files.

        Args:
            result_files (List[Path]): List of result files.
            expected_checksums (Dict[Path, str]): Dictionary mapping result files to their expected checksums.

        Returns:
            bool: True if the experiment is reproducible, False otherwise.
        """
        actual_checksums = self.compute_result_checksums(result_files)
        for file, expected_checksum in expected_checksums.items():
            if file not in actual_checksums or actual_checksums[file] != expected_checksum:
                logger.error(f"Checksum mismatch for {file}: expected {expected_checksum}, got {actual_checksums.get(file, 'unknown')}")
                return False
        return True

    def generate_reproducibility_report(self, result_files: List[Path], expected_checksums: Dict[Path, str]) -> str:
        """
        Generates a reproducibility report.

        Args:
            result_files (List[Path]): List of result files.
            expected_checksums (Dict[Path, str]): Dictionary mapping result files to their expected checksums.

        Returns:
            str: Reproducibility report.
        """
        report = "Reproducibility Report:\n"
        if self.validate_reproducibility(result_files, expected_checksums):
            report += "Experiment is reproducible.\n"
        else:
            report += "Experiment is not reproducible.\n"
        return report

    def version_control_experiments(self) -> None:
        """
        Version controls the experiments using Git.
        """
        try:
            repo = Repo(self.experiment_dir)
            repo.index.add([str(file) for file in self.experiment_dir.iterdir()])
            repo.index.commit("Experiment results")
        except GitCommandError as e:
            logger.error(f"Error version controlling experiments: {e}")

class ExperimentState:
    """
    Represents the state of an experiment.

    Attributes:
        config (Dict): Configuration dictionary.
        result_files (List[Path]): List of result files.
    """

    def __init__(self, config: Dict, result_files: List[Path]):
        """
        Initializes the ExperimentState.

        Args:
            config (Dict): Configuration dictionary.
            result_files (List[Path]): List of result files.
        """
        self.config = config
        self.result_files = result_files

class ReproducibilityException(Exception):
    """
    Exception raised when there is an issue with reproducibility.
    """

    def __init__(self, message: str):
        """
        Initializes the ReproducibilityException.

        Args:
            message (str): Error message.
        """
        self.message = message
        super().__init__(message)

def main():
    # Create a ReproducibilityKit instance
    experiment_dir = Path("experiments")
    config = {
        "experiment_name": "example_experiment",
        "result_files": ["result1.txt", "result2.txt"]
    }
    kit = ReproducibilityKit(experiment_dir, config)

    # Compute result checksums
    result_files = [experiment_dir / file for file in config["result_files"]]
    checksums = kit.compute_result_checksums(result_files)
    print("Checksums:", checksums)

    # Save experiment state
    experiment_state = ExperimentState(config, result_files)
    kit.save_experiment_state(experiment_state.__dict__)

    # Validate reproducibility
    expected_checksums = checksums
    is_reproducible = kit.validate_reproducibility(result_files, expected_checksums)
    print("Is reproducible:", is_reproducible)

    # Generate reproducibility report
    report = kit.generate_reproducibility_report(result_files, expected_checksums)
    print("Reproducibility Report:")
    print(report)

    # Version control experiments
    kit.version_control_experiments()

if __name__ == "__main__":
    main()