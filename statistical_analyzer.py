import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalAnalyzer:
    """
    Performs advanced statistical analysis including correlation tests and regression modeling.

    Attributes:
    ----------
    data : pandas.DataFrame
        Input data for analysis.
    config : Dict
        Configuration settings for analysis.

    Methods:
    -------
    compute_pearson_correlation()
        Computes Pearson correlation between two variables.
    perform_regression_analysis()
        Performs linear regression analysis.
    calculate_intraclass_correlation()
        Calculates intraclass correlation coefficient.
    run_significance_tests()
        Runs significance tests for regression analysis.
    generate_statistical_report()
        Generates a statistical report based on analysis results.
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        """
        Initializes the StatisticalAnalyzer class.

        Parameters:
        ----------
        data : pandas.DataFrame
            Input data for analysis.
        config : Dict
            Configuration settings for analysis.
        """
        self.data = data
        self.config = config

    def compute_pearson_correlation(self, var1: str, var2: str) -> Tuple[float, float]:
        """
        Computes Pearson correlation between two variables.

        Parameters:
        ----------
        var1 : str
            Name of the first variable.
        var2 : str
            Name of the second variable.

        Returns:
        -------
        correlation_coefficient : float
            Pearson correlation coefficient.
        p_value : float
            p-value for the correlation test.
        """
        try:
            correlation_coefficient, p_value = pearsonr(self.data[var1], self.data[var2])
            logging.info(f"Pearson correlation between {var1} and {var2}: {correlation_coefficient}, p-value: {p_value}")
            return correlation_coefficient, p_value
        except Exception as e:
            logging.error(f"Error computing Pearson correlation: {str(e)}")
            return None, None

    def perform_regression_analysis(self, target_var: str, predictor_vars: List[str]) -> Dict:
        """
        Performs linear regression analysis.

        Parameters:
        ----------
        target_var : str
            Name of the target variable.
        predictor_vars : List[str]
            Names of the predictor variables.

        Returns:
        -------
        regression_results : Dict
            Dictionary containing regression results.
        """
        try:
            # Split data into training and testing sets
            X = self.data[predictor_vars]
            y = self.data[target_var]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Perform linear regression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate mean squared error
            mse = mean_squared_error(y_test, y_pred)

            # Calculate R-squared value
            r_squared = model.score(X_test_scaled, y_test)

            # Create a dictionary to store regression results
            regression_results = {
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "mse": mse,
                "r_squared": r_squared
            }

            logging.info(f"Regression analysis results: {regression_results}")
            return regression_results
        except Exception as e:
            logging.error(f"Error performing regression analysis: {str(e)}")
            return None

    def calculate_intraclass_correlation(self, var1: str, var2: str) -> float:
        """
        Calculates intraclass correlation coefficient.

        Parameters:
        ----------
        var1 : str
            Name of the first variable.
        var2 : str
            Name of the second variable.

        Returns:
        -------
        icc : float
            Intraclass correlation coefficient.
        """
        try:
            # Calculate intraclass correlation coefficient
            icc = stats.intraclass_correlation(self.data[var1], self.data[var2])
            logging.info(f"Intraclass correlation coefficient between {var1} and {var2}: {icc}")
            return icc
        except Exception as e:
            logging.error(f"Error calculating intraclass correlation coefficient: {str(e)}")
            return None

    def run_significance_tests(self, target_var: str, predictor_vars: List[str]) -> Dict:
        """
        Runs significance tests for regression analysis.

        Parameters:
        ----------
        target_var : str
            Name of the target variable.
        predictor_vars : List[str]
            Names of the predictor variables.

        Returns:
        -------
        significance_test_results : Dict
            Dictionary containing significance test results.
        """
        try:
            # Perform ANOVA test
            formula = f"{target_var} ~ {' + '.join(predictor_vars)}"
            model = ols(formula, data=self.data).fit()
            anova_table = anova_lm(model)

            # Create a dictionary to store significance test results
            significance_test_results = {
                "anova_table": anova_table
            }

            logging.info(f"Significance test results: {significance_test_results}")
            return significance_test_results
        except Exception as e:
            logging.error(f"Error running significance tests: {str(e)}")
            return None

    def generate_statistical_report(self) -> str:
        """
        Generates a statistical report based on analysis results.

        Returns:
        -------
        report : str
            Statistical report.
        """
        try:
            # Create a report string
            report = "Statistical Report:\n"

            # Add Pearson correlation results
            correlation_coefficient, p_value = self.compute_pearson_correlation("var1", "var2")
            report += f"Pearson correlation between var1 and var2: {correlation_coefficient}, p-value: {p_value}\n"

            # Add regression analysis results
            regression_results = self.perform_regression_analysis("target_var", ["predictor_var1", "predictor_var2"])
            report += f"Regression analysis results: {regression_results}\n"

            # Add intraclass correlation coefficient
            icc = self.calculate_intraclass_correlation("var1", "var2")
            report += f"Intraclass correlation coefficient between var1 and var2: {icc}\n"

            # Add significance test results
            significance_test_results = self.run_significance_tests("target_var", ["predictor_var1", "predictor_var2"])
            report += f"Significance test results: {significance_test_results}\n"

            logging.info(f"Statistical report: {report}")
            return report
        except Exception as e:
            logging.error(f"Error generating statistical report: {str(e)}")
            return None


class Configuration:
    """
    Configuration settings for statistical analysis.

    Attributes:
    ----------
    data_file : str
        Path to the input data file.
    target_var : str
        Name of the target variable.
    predictor_vars : List[str]
        Names of the predictor variables.
    """

    def __init__(self, data_file: str, target_var: str, predictor_vars: List[str]):
        """
        Initializes the Configuration class.

        Parameters:
        ----------
        data_file : str
            Path to the input data file.
        target_var : str
            Name of the target variable.
        predictor_vars : List[str]
            Names of the predictor variables.
        """
        self.data_file = data_file
        self.target_var = target_var
        self.predictor_vars = predictor_vars


def main():
    # Create a configuration object
    config = Configuration("data.csv", "target_var", ["predictor_var1", "predictor_var2"])

    # Load data
    data = pd.read_csv(config.data_file)

    # Create a statistical analyzer object
    analyzer = StatisticalAnalyzer(data, config.__dict__)

    # Generate a statistical report
    report = analyzer.generate_statistical_report()

    # Print the report
    print(report)


if __name__ == "__main__":
    main()