# TOSA Adapter for Model Explorer

TOSA Adapter for [google-ai-edge/model-explorer](https://github.com/google-ai-edge/model-explorer) that enables visualization of [TOSA](https://www.mlplatform.org/tosa/) files.

![](https://raw.githubusercontent.com/arm/tosa-adapter-model-explorer/main/screenshots/tosa-adapter-readme-screenshot.png)

## Requirements

- Python >= 3.10

## Installation

### pip + PyPI
    pip install tosa-adapter-model-explorer

### GitHub

    gh release download \
    --repo arm/tosa-adapter-model-explorer \
    --pattern "*.whl" &&
    pip install *.whl

Or through the [GitHub Releases](https://github.com/arm/tosa-adapter-model-explorer/releases) UI.

## Usage

Install Model Explorer:

    pip install torch ai-edge-model-explorer==0.1.25

Launch Model Explorer with the TOSA adapter enabled:

    model-explorer --extensions=tosa_adapter_model_explorer

See the [Model Explorer wiki](https://github.com/google-ai-edge/model-explorer/wiki) for more information.

## Trademark notice

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

## Contributions

We are not accepting direct contributions at this time.
If you have any feedback or feature requests, please use the repository issues section.
