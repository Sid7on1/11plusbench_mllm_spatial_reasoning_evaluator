import re
import pandas as pd
from typing import Dict, List, Tuple
import logging
from logging.config import dictConfig
import json
from enum import Enum

# Configure logging
logging_config = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console']
    }
}

dictConfig(logging_config)

class ResponseFormat(Enum):
    ORDINAL = 1
    LETTER = 2
    VERBALIZED = 3

class ResponseParser:
    def __init__(self, config: Dict):
        """
        Initialize the ResponseParser with a configuration dictionary.

        Args:
        - config (Dict): A dictionary containing configuration settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def parse_response_format(self, response: str) -> ResponseFormat:
        """
        Parse the response format from the given response string.

        Args:
        - response (str): The response string to parse.

        Returns:
        - ResponseFormat: The parsed response format.
        """
        if re.match(r'^\d+$', response):
            return ResponseFormat.ORDINAL
        elif re.match(r'^[a-zA-Z]+$', response):
            return ResponseFormat.LETTER
        else:
            return ResponseFormat.VERBALIZED

    def extract_choice_from_text(self, response: str) -> str:
        """
        Extract the choice from the given response text.

        Args:
        - response (str): The response text to extract the choice from.

        Returns:
        - str: The extracted choice.
        """
        # Use regular expression to extract the choice
        match = re.search(r'([a-zA-Z0-9]+)', response)
        if match:
            return match.group(1)
        else:
            return None

    def handle_ambiguous_responses(self, response: str) -> str:
        """
        Handle ambiguous responses by attempting to extract the choice.

        Args:
        - response (str): The ambiguous response to handle.

        Returns:
        - str: The handled response.
        """
        choice = self.extract_choice_from_text(response)
        if choice:
            return choice
        else:
            self.logger.warning(f"Unable to extract choice from response: {response}")
            return None

    def validate_parsed_answer(self, parsed_answer: str, correct_answer: str) -> bool:
        """
        Validate the parsed answer against the correct answer.

        Args:
        - parsed_answer (str): The parsed answer to validate.
        - correct_answer (str): The correct answer to validate against.

        Returns:
        - bool: True if the parsed answer is correct, False otherwise.
        """
        if parsed_answer == correct_answer:
            return True
        else:
            self.logger.error(f"Parsed answer '{parsed_answer}' does not match correct answer '{correct_answer}'")
            return False

    def generate_parsing_report(self, responses: List[str], correct_answers: List[str]) -> Dict:
        """
        Generate a parsing report for the given responses and correct answers.

        Args:
        - responses (List[str]): A list of responses to generate the report for.
        - correct_answers (List[str]): A list of correct answers to generate the report for.

        Returns:
        - Dict: A dictionary containing the parsing report.
        """
        report = {
            'correct': 0,
            'incorrect': 0,
            'ambiguous': 0
        }
        for response, correct_answer in zip(responses, correct_answers):
            parsed_answer = self.handle_ambiguous_responses(response)
            if parsed_answer:
                if self.validate_parsed_answer(parsed_answer, correct_answer):
                    report['correct'] += 1
                else:
                    report['incorrect'] += 1
            else:
                report['ambiguous'] += 1
        return report

class ResponseParserConfig:
    def __init__(self, config_file: str):
        """
        Initialize the ResponseParserConfig with a configuration file.

        Args:
        - config_file (str): The path to the configuration file.
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        Returns:
        - Dict: The configuration dictionary.
        """
        return self.config

class ResponseParserException(Exception):
    def __init__(self, message: str):
        """
        Initialize the ResponseParserException with a message.

        Args:
        - message (str): The error message.
        """
        self.message = message
        super().__init__(self.message)

def main():
    config_file = 'config.json'
    config = ResponseParserConfig(config_file).get_config()
    parser = ResponseParser(config)
    responses = ['1', 'a', 'The correct answer is b']
    correct_answers = ['1', 'a', 'b']
    report = parser.generate_parsing_report(responses, correct_answers)
    print(report)

if __name__ == '__main__':
    main()