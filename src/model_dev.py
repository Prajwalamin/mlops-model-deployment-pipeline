import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models 
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """ 
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels    
        Returns:
            model: Trained model
        
        """
        pass

class LinearRegressionModel(Model): 
    """
    Linear regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model

        Args:   
            X_train: Training data
            y_train: Training labels    
        Returns:    
            None
        """

        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training complete")
            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e

