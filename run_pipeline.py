from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
    #Run Pipeline
    train_pipeline(data_path="/Users/prajwalamin/Documents/Development/MLOps/MLOps-fcc/data/olist_customers_dataset.csv")