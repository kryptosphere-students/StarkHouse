# MNIST Example Template

Welcome to the Giza Actions SDK template! The [Giza Actions SDK](https://actions.gizatech.xyz/welcome/giza-actions-sdk) is tailored to assist you in designing your ZKML workflows efficiently. This project provides pre-configured actions ready for deployment on the Giza platform. The purpose of this template is to demonstrate how to construct your ZKML workflows using the Giza Actions SDK.

*Note: This template is based on the [MNIST tutorial](https://actions.gizatech.xyz/tutorials/build-a-verifiable-neural-network-with-giza-actions). Please be aware that certain steps, such as transpiling the model and deploying the generated model on Giza Plateform, are required between action executions. For a more comprehensive understanding, refer to the tutorial.*

## Requirements
- Python 3.11
- Poetry

## Get Started
```bash
$ poetry shell
$ poetry install
```

## Structure
Within the `starkhouse` directory, you'll discover multiple generated files:
- `train_action.py`: Contains actions for training your model.
- `predict_onnx_action.py`: Includes actions for making predictions with an ONNX model.
- `predict_cairo_action.py`: Includes actions for making verifiable predictions with the Orion Cairo model.

## Usage
To use this project, follow these steps:
1. Install the required dependencies.
2. Execute any of the provided action scripts using the command `python starkhouse/{action_file}.py`, for example, `python starkhouse/{train_action}.py`.

## Learn More
Explore more about the Giza Actions SDK [here](https://actions.gizatech.xyz/welcome/giza-actions-sdk).

## Acknowledgement
This template was generated using cookiecutter.

