from azureml.core import Workspace
from azureml.core.model import Model
from workspace import get_workspace
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = os.environ.get("BASE_NAME")+"-AML-RG"
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    MODEL_NAME = os.environ.get('MODEL_NAME')
    model_data_path = os.environ.get("MODEL_DATA_PATH_DATASTORE")

    ws = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    modelName = MODEL_NAME.rstrip('h5')+'onnx'
    model = Model(workspace=ws, name=modelName)
    print(model)
    model.download()
    ds = ws.get_default_datastore()
    print(ds)
    ds.download(target_path='.', prefix=model_data_path, show_progress=True)

if __name__ == '__main__':
    main()