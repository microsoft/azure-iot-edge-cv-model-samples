from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Datastore
from azureml.core.dataset import Dataset

from azureml.train.dnn import TensorFlow

import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath("./MLOps/ml_service/util"))  # NOQA: E402
from workspace import get_workspace
from attach_compute import get_compute


def main():
    load_dotenv()
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = os.environ.get("BASE_NAME")+"-AML-RG"
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    sources_directory_train = os.environ.get("SOURCES_DIR_TRAIN")
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    model_name = os.environ.get("MODEL_NAME")
    build_id = os.environ.get("BUILD_BUILDID")
    pipeline_name = os.environ.get("TRAINING_PIPELINE_NAME")
    data_path = os.environ.get("DATA_PATH_DATASTORE")
    model_data_path = os.environ.get("MODEL_DATA_PATH_DATASTORE")

    # Get Azure machine learning workspace
    aml_workspace = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(
        aml_workspace,
        compute_name,
        vm_size)
    if aml_compute is not None:
        print(aml_compute)

    model_name = PipelineParameter(
        name="model_name", default_value=model_name)
    release_id = PipelineParameter(
        name="release_id", default_value="0"
    )
 
    ds = aml_workspace.get_default_datastore()

    dataref_folder = ds.path(data_path).as_mount()
    model_dataref = ds.path(model_data_path).as_mount()

    # NEED those two folders mounted on datastore and env variables specified in variable groups

    #ds.upload(src_dir='./VOCdevkit', target_path='VOCdevkit', overwrite=True, show_progress=True)
    #ds.upload(src_dir='./model_data', target_path='VOCmodel_data', overwrite=True, show_progress=True)

    yoloEstimator = TensorFlow(source_directory=sources_directory_train+'/training', 
                               compute_target=aml_compute,
                               entry_script=train_script_path,
                               pip_packages=['keras', 'pillow', 'matplotlib', 'onnxmltools', 'keras2onnx==1.5.1'], # recent versions of keras2onnx give conversion issues 
                               use_gpu=True,
                               framework_version='1.13')

    train_step = EstimatorStep(name="Train & Convert Model",
        estimator=yoloEstimator,
        estimator_entry_script_arguments=[
            "--release_id", release_id,
            "--model_name", model_name,
            "--data_folder", dataref_folder, 
            "--model_path", model_dataref
        ],
        runconfig_pipeline_params=None,
        inputs=[dataref_folder, model_dataref],
        compute_target=aml_compute,
        allow_reuse=False)
    print("Step Train & Convert created")

    train_pipeline = Pipeline(workspace=aml_workspace, steps=[train_step])
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="Model training/retraining pipeline",
        version=build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()