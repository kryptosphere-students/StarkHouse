FROM python:3.11

WORKDIR /usr/src/app

RUN pip install numpy
RUN pip install scipy
RUN pip install torch
RUN pip install torchvision
RUN pip install giza-actions
RUN pipx install giza-cli

COPY . .

CMD ["giza", "create_api_key"]

# CMD [ "python", "./starkhouse/train_action.py"]
