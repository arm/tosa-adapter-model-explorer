# Model Explorer TOSA Flatbuffer

A model explorer TOSA Flatbuffer plugin

## Getting Started

```
pip install -e .
```

Set PYTHONPATH:

```
PYTHONPATH="{absolute_path_to}/model-explorer-tosa-flatbuffer/src/model_explorer_tosa_flatbuffer
```

Run:

```
model-explorer --extensions=model_explorer_tosa_flatbuffer
```

## Generating Flatbuffer Objects

Requires `flatc`. `flatc` 25.2.10 was used to generate the initial objects.
Run sh script:

```
chmod +x scripts/flatbuffer-gen.sh && scripts/flatbuffer-gen.sh
```
