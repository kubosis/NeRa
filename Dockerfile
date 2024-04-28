ARG BASE_IMAGE=python:3.11

FROM ${BASE_IMAGE} as base

WORKDIR /nera

ENV PYTHONUNBUFFERED TRUE

COPY . /nera

# Copy just the requirements file to the container
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install  torch==2.1.2 \
    && pip install  torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html \
    && pip install  torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cpu.html \
    && pip install  torch-geometric==2.4.0 \
    && pip install  torch-geometric-temporal==0.52.0 \
    && pip install  -r requirements.txt \
    && pip list

# Copy the rest of the application code
COPY . .

CMD ["python", "nera/evaluation.py"]

EXPOSE 3000


