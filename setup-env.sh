#!/bin/bash

if command -v nvidia-smi &> /dev/null; then
  ENVYAML=env-cuda.yaml
else
  ENVYAML=env.yaml
fi

conda env create --file "${ENVYAML}"
