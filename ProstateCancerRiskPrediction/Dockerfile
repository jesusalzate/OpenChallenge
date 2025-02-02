FROM python:3.9-slim
RUN apt-get update && apt-get install -y git

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt
RUN pip install -U torch==1.13.1 torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip install SimpleITK evalutils git+https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git#egg=bimcv_aikit


COPY --chown=user:user model_best_state.pth /opt/app/
COPY --chown=user:user EfficientNetMultimodal.py /opt/app/
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
