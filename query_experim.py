import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

str_experim_name = "Breast cancer classification"
client = MlflowClient()

# Retrieve Experiment id by experiment name
int_experim_id = client.get_experiment_by_name(str_experim_name).experiment_id

# Retrieve Runs information (parameters and metrics)
dtf_runs = mlflow.search_runs(experiment_names=[str_experim_name])

dtf_runs_log_reg = mlflow.search_runs(experiment_ids=int_experim_id, filter_string='tags.mlflow.runName = "logistic regression"')
dtf_runs_rf = mlflow.search_runs(experiment_ids=int_experim_id, filter_string='tags.mlflow.runName = "random forest"')

# Retrieve the run with max accuracy in test set among all the runs
ser_run_opt = dtf_runs.iloc[dtf_runs['metrics.test accuracy_score'].idxmax()]
str_run_id_opt = ser_run_opt['run_id']
str_artifact_uri_opt = ser_run_opt['artifact_uri']

model = mlflow.sklearn.load_model(str_artifact_uri_opt + "/model")

dtf_class_table = pd.read_csv(str_artifact_uri_opt + "/classification_table.csv")