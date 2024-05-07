import logging
from abc import ABC, abstractmethod
import numpy as np

class Evaluation(ABC):
    """
    Abstract class for all models 
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ 
        Calculate the scores
        
        Args:
            y_true: True labels
            y_pred: Predicted labels    
        Returns:
            scores: Scores
        
        """
        pass

class MSE(Evaluation):
    """
    Evaluate model using Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores

        Args:   
            y_true: True labels
            y_pred: Predicted labels    
        Returns:    
            scores: Scores
        """

        try:
            logging.info("Calculating Mean Squared Error")
            mse = np.mean((y_true - y_pred)**2)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):
    """
    Evaluate model using R2
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores

        Args:   
            y_true: True labels
            y_pred: Predicted labels    
        Returns:    
            scores: Scores
        """

        try:
            logging.info("Calculating R2")
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = float(1 - (ss_res/ss_tot))
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2: {e}")
            raise e

class RMSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores

        Args:   
            y_true: True labels
            y_pred: Predicted labels    
        Returns:    
            scores: Scores
        """

        try:
            logging.info("Calculating Root Mean Squared Error")
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e

   