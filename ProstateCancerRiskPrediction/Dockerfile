FROM python:3.9-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
RUN python -m piptools sync requirements.txt
RUN pip install SimpleITK evalutils 'monai[itk]'

COPY --chown=user:user model_best_weights.pth /opt/app/
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
