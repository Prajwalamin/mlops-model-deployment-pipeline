from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model name configuration
    """

    model_name: str = "LinearRegression"