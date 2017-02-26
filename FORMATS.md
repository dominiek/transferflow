
_Work in progress. Feedback on this very welcome_

# Abstract

# Definitions

## 1. Model Package

### 1.1 Purpose

### 1.2 Model Package Directory Structure

|Path               | Required? | Description                            |
|-------------------|-----------|---------------------------------------|
|`/`                | Required  | Model Package Root |
|`/nnpackage.json`  | Required  | Model Package Meta-Data File |
|`/labels.jsons`    | Optional  | Default Labels Definition File |
|`/state`           | Optional  | Default Model State Directory |

### 1.3 Model Package Meta-Data File: `nnpackage.json`

Example:

```json
{
  "id": "58745c9fbd17c82dd4ff7c9c",
  "name": "Scene Type",
  "description": "Classify images into Indoor VS Outdoor scenes",
  "version": 0.1,
  "engines": {
    "tensorflow": ">=0.11"
  },
  "nodes": {
    "softmaxOutput": {
      "type": "tensor",
      "id": "retrained_layer:0"
    },
    "convolutionalRepresentation": {
      "type": "tensor",
      "id": "pool_3/_reshape:0"
    }
  },
  "author": {
    "name": "Dominiek Ter Heide",
    "email": "info@dominiek.com"
  },
  "labelsDefinitionFile": "labels.jsons",
  "stateDir": "state"
}
```

All attribute options:

| Attribute                | Description      | Required? |
|--------------------------|------------------|-----------|
|id                        |Unique model ID, must be lowercase, no spaces   |Required   |
|name                      |Human friendly name of model |Required   |
|version                   |Version of this model and state |Required   |
|engines                   |Required engines and versions. See _1.4 Engines Information_ |Required   |
|description               |Description of what the model does |Optional   |
|nodes                     |Information about key nodes/layers in neural net. See _1.5 Nodes Information_ |Optional   |
|author                    |Author information. See _1.6 Author Information_ |Optional   |
|labelsDefinitionFile      |Path of labels definition. Defaults to `labels.jsons`. See _3. Label Definitions_ |Optional   |
|stateDir                  |Directory where engine keeps state of model. Defaults to `state` |Optional   |

### 1.4 Engines Information

### 1.5 Nodes Information

### 1.6 Author Information

## 2. Scaffold Package

### 2.1 Purpose

### 2.2 Scaffold Package Directory Structure

|Path               |Description                            |
|-------------------|---------------------------------------|
|`/`                | Scaffold Root |
|`/nnscaffold.json` | Scaffold Meta-Data File |
|`/labels.jsons`    | Labels Definition File |
|`/images`          | Default Images Training Data Folder |
|`/cache`           | Default Scaffold Cache Directory |

### 2.3 Scaffold Package Meta-Data File: `nnscaffold.json`

Example:

```json
{
  "id": "58745c9fbd17c82dd4ff7c9c",
  "name": "Scene Type",
  "version": 0.1,
  "description": "Classify images into Indoor VS Outdoor scenes",
  "author": {
    "name": "Dominiek Ter Heide",
    "email": "info@dominiek.com"
  },
  "labelsDefinitionFile": "labels.json",
  "trainingDir": "images",
  "cacheDir": "cache"
}
```

| Attribute                | Description      | Required? |
|--------------------------|------------------|-----------|
|id                        |Unique model ID, must be lowercase, no spaces   |Required   |
|name                      |Human friendly name of model |Required   |
|version                   |Version of this scaffold and its labeled data |Required   |
|description               |Description of what the model does |Optional   |
|author                    |Author information. See _1.6 Author Information_ |Optional   |
|labelsDefinitionFile      |Path of labels definition. Defaults to `labels.jsons`. See _3. Label Definitions_ |Optional   |

### 2.4 Images Training Data Folder

### 2.4.1 Bounding Boxes for Object Detection

### 2.5 Scaffold Cache Directory

## 3. Label

### 3.1 Label Files

### 3.2 Label Definition

### 3.3 Hierarchical Labels
