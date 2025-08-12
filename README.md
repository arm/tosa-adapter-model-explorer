# TOSA Adapter for Model Explorer

A TOSA adapter for Model Explorer plugin

## Getting Started

```
pip install -e .
```

Set PYTHONPATH:

```
PYTHONPATH="{absolute_path_to}/tosa-flatbuffer-model-explorer/src/model_explorer_tosa_flatbuffer
```

Run:

```
model-explorer --extensions=tosa_flatbuffer_model_explorer
```

## Generating Flatbuffer Objects

Requires `flatc`. `flatc` 25.2.10 was used to generate the initial objects.
Run sh script:

```
chmod +x scripts/flatbuffer-gen.sh && scripts/flatbuffer-gen.sh
```
