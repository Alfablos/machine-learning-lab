# Machine Learning: Python Vs. Rust Vs. ?

Drafts, tests and projects implementations for the DS/ML course, rust tests and other tools.

## Usage

```shell
nix develop
```
```shell
docker compose up -d --build
```
The container method is advised since `cuml` libraries, for GPU support, cannot be found in nixpkgs.

Refer to [GitHub -  iot-salzburg/gpu-jupyter](https://github.com/iot-salzburg/gpu-jupyter/tree/master) if you want to further customize the container environment, since sourcing the built image is not the recommended way, unless you know what you're doing.

## Contents

## Python
Using `Python 3.12` as Tensorflow still seems incompatible with 3.13.

## Rust
The `1.91.0-nightly` is being used as some features used are still unstable.



## Sources

* zerotomastery.io - Complete A.I. Machine Learning and Data Science: Zero to Mastery

## TODOs

* Try out [duckdb](https://duckdb.org/) in python (non need for rust)





