import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2, RMSE
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    """ 
    Evaluate the model using the provided data.

    Args:
        df: The data to evaluate the model with.
    """

    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)
        return r2,rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e