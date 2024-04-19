import numpy as np
import torch
import torch.nn.functional as F
from giza_actions.action import Action, action
from giza_actions.model import GizaModel
from giza_actions.task import task

from starkhouse.predict_onnx_action import preprocess_image

MODEL_ID = "Test"  # Update with your model ID
VERSION_ID = 1  # Update with your version ID


@task(name='Prediction with Cairo')
def prediction(image, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, request_id) = model.predict(
        input_feed={"image": image}, verifiable=True, output_dtype="Tensor<FP16x16>"
    )

    # Convert result to a PyTorch tensor
    probabilities = torch.tensor(result)
    # Use argmax to get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class, request_id


@action(name="Execution: Prediction with Cairo", log_prints=True)
def execution():
    image = preprocess_image("./zero.jpg")
    (result, request_id) = prediction(image, MODEL_ID, VERSION_ID)
    print("Result: ", result)
    print("Request id: ", request_id)

    return result, request_id


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-mnist-cairo-action")
    action_deploy.serve(name="pytorch-mnist-cairo-deployment")
