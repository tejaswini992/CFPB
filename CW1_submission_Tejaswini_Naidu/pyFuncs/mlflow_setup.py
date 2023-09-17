import mlflow

class MLflowConfig:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_id = None

    def configure_mlflow(self):
        # Set other MLflow settings as desired
        mlflow.set_experiment(self.experiment_name)

    def create_experiment(self):
        # Create a new MLflow experiment
        experiment = mlflow.create_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        print(f"Experiment created with ID: {self.experiment_id}")

    def get_experiment_id(self):
        # Get the ID of an existing MLflow experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        print(f"Experiment ID: {self.experiment_id}")
