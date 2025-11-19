FROM docker.io/cschranz/gpu-jupyter:v1.9_cuda-12.6_ubuntu-24.04_python-only

USER jovyan

RUN conda update --yes -n base -c conda-forge conda
RUN conda update --yes --prefix "${CONDA_DIR}" --all

RUN pip install cupy-cuda12x cutensor-cu12 nvidia-nccl-cu12 nvidia-cudnn-cu12 polars
RUN pip install \
      --extra-index-url=https://pypi.nvidia.com \
      "cudf-cu12==25.10.*" "dask-cudf-cu12==25.10.*" "cuml-cu12==25.10.*" \
      "cugraph-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cuxfilter-cu12==25.10.*" \
      "cucim-cu12==25.10.*" "pylibraft-cu12==25.10.*" "raft-dask-cu12==25.10.*" \
      "cuvs-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cudf-polars-cu12==25.10.*"

# typing
RUN pip install mypy types-tensorflow pandas-stubs

RUN mamba clean --all -f -y && \
  rm -rf "/home/${NB_USER}/.cache/yarn"

RUN fix-permissions "${CONDA_DIR}" && \
      fix-permissions "/home/${NB_USER}"

USER jovyan

