FROM docker.io/cschranz/gpu-jupyter:v1.9_cuda-12.6_ubuntu-24.04_python-only

USER jovyan

RUN conda update --yes -n base -c conda-forge conda
RUN conda update --yes --prefix "${CONDA_DIR}" --all

RUN pip install cupy-cuda13x cutensor-cu13 nvidia-nccl-cu13 nvidia-cudnn-cu13 polars xgboost h5py
RUN pip install \
      --extra-index-url=https://pypi.nvidia.com \
      "cudf-cu13==25.10.*" "dask-cudf-cu13==25.10.*" "cuml-cu13==25.10.*" \
      "cugraph-cu13==25.10.*" "nx-cugraph-cu13==25.10.*" "cuxfilter-cu13==25.10.*" \
      "cucim-cu13==25.10.*" "pylibraft-cu13==25.10.*" "raft-dask-cu13==25.10.*" \
      "cuvs-cu13==25.10.*" "nx-cugraph-cu13==25.10.*" "cudf-polars-cu13==25.10.*"

# typing
RUN pip install mypy tensorflow[and-cuda] types-tensorflow pandas-stubs
RUN pip uninstall jupyter_contrib_nbextensions -y

# Utils
RUN pip install tabulate pyvirtualdisplay swig gymnasium dlai-tools
# RUN pip install tabulate pyvirtualdisplay swig gymnasium[box2D]

RUN mamba clean --all -f -y && \
  rm -rf "/home/${NB_USER}/.cache/yarn"

RUN fix-permissions "${CONDA_DIR}" && \
      fix-permissions "/home/${NB_USER}"

USER jovyan

