"""
Azure ML Model Monitor Configuration
Monitors model performance, data drift, and model drift
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataDriftThreshold,
    ModelPerformanceThreshold,
    PredictionDriftThreshold,
    MonitoringTarget,
    DeploymentModelConfiguration,
)
from azure.ai.ml.constants import (
    MonitorFeatureType,
    MonitorModelType,
    TriggerFrequency,
)
import os

# Configuration
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-sentinel-ai")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "mlw-sentinel-prod")

ml_client = MLClient(
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)


def create_data_drift_monitor(
    baseline_data_uri: str,
    production_data_uri: str,
    monitor_name: str = "biometric-data-drift"
):
    """Create data drift monitoring for input features"""
    
    data_drift_config = DataDriftThreshold(
        numerical_column_threshold=0.1,
        categorical_column_threshold=0.1,
    )
    
    monitoring_target = MonitoringTarget(
        prediction_data=production_data_uri,
        target_columns=["predicted_label", "confidence"],
        features=[
            "face_bbox_x1", "face_bbox_y1", "face_bbox_x2", "face_bbox_y2",
            "embedding_0", "embedding_1", "embedding_2", "embedding_3"
        ],
    )
    
    monitor = ml_client._ml_client.monitors.begin_create(
        name=monitor_name,
        monitoring_target=monitoring_target,
        data_drift_config=data_drift_config,
    )
    
    return monitor


def create_model_performance_monitor(
    model_name: str,
    baseline_data_uri: str,
    production_data_uri: str,
    monitor_name: str = "biometric-model-performance"
):
    """Create model performance monitoring"""
    
    performance_threshold = ModelPerformanceThreshold(
        accuracy_threshold=0.95,
        precision_threshold=0.95,
        recall_threshold=0.95,
        f1_score_threshold=0.95,
    )
    
    monitoring_target = MonitoringTarget(
        model=model_name,
        baseline_data=baseline_data_uri,
        prediction_data=production_data_uri,
        target_columns=["actual_label", "predicted_label", "confidence"],
    )
    
    monitor = ml_client._ml_client.monitors.begin_create(
        name=monitor_name,
        monitoring_target=monitoring_target,
        performance_thresholds=performance_threshold,
    )
    
    return monitor


def create_prediction_drift_monitor(
    baseline_predictions_uri: str,
    production_predictions_uri: str,
    monitor_name: str = "biometric-prediction-drift"
):
    """Create prediction drift monitoring"""
    
    drift_threshold = PredictionDriftThreshold(
        probability_threshold=0.1,
    )
    
    monitoring_target = MonitoringTarget(
        prediction_data=production_predictions_uri,
        baseline_prediction_data=baseline_predictions_uri,
        target_columns=["predicted_label", "confidence"],
    )
    
    monitor = ml_client._ml_client.monitors.begin_create(
        name=monitor_name,
        monitoring_target=monitoring_target,
        prediction_drift_config=drift_threshold,
    )
    
    return monitor


def setup_model_monitoring():
    """Setup all model monitoring for biometric service"""
    
    # Data drift monitor
    create_data_drift_monitor(
        baseline_data_uri="azureml://databases/sentinel_blobs/faces/baseline",
        production_data_uri="azureml://databases/sentinel_blobs/faces/production",
        monitor_name="biometric-data-drift"
    )
    
    # Model performance monitor
    create_model_performance_monitor(
        model_name="biometric-recognizer",
        baseline_data_uri="azureml://databases/sentinel_blobs/faces/validation",
        production_data_uri="azureml://databases/sentinel_blobs/faces/production",
        monitor_name="biometric-model-performance"
    )
    
    # Prediction drift monitor
    create_prediction_drift_monitor(
        baseline_predictions_uri="azureml://databases/sentinel_blobs/predictions/baseline",
        production_predictions_uri="azureml://databases/sentinel_blobs/predictions/production",
        monitor_name="biometric-prediction-drift"
    )
    
    print("All model monitors created successfully")


def get_monitoring_alerts(monitor_name: str):
    """Get alerts from model monitor"""
    
    alerts = ml_client._ml_client.monitors.list_alerts(
        name=monitor_name,
        alert_type="data_drift"
    )
    
    return alerts


if __name__ == "__main__":
    setup_model_monitoring()
