# PyDataToolkit
A collection of Python scripts for data exploration, dataset management, image processing, and other essential manipulation tasks.

## Features

### Data resampling (with GPU acceleration)

The resampling script (`src/data_processing/resample_data.py`) now supports multiple backends with automatic selection:

- **MONAI + GPU** (fastest): 10-100x faster than CPU when CUDA is available
- **MONAI + CPU**: Optimized CPU processing with MONAI
- **nibabel + scipy** (fallback): Reliable CPU-based resampling

The script automatically detects and uses the best available backend.

## Installation


### Basic Installation (CPU only)
You can use either `uv` (recommended for speed) or `pip`:

**With uv:**
```bash
uv sync
```

**With pip:**
```bash
pip install -e .
```

### GPU-Accelerated Installation (Recommended)
For significantly faster resampling with GPU support:

**With uv:**
```bash
uv sync --extra gpu

# Or install all optional features
uv sync --extra all
```

**With pip:**
```bash
pip install -e ".[gpu]"

# Or install all optional features
pip install -e ".[all]"
```

**Note**: GPU acceleration requires NVIDIA GPU with CUDA support. On systems without GPU, MONAI will fall back to CPU processing.

## Usage


This project is designed to be run inside a Python environment. We recommend using `venv` (`conda` is not supported at this time).
Before any script execution, ensure you have a configuration YAML file (e.g., `configs/resample_config.yml`) specifying input/output directories, target spacing, and other options. 

### Dataset Indexing (JSON Creation)

Most processing scripts in PyDataToolkit rely on a JSON index of your dataset, which lists all images and (optionally) labels. Generate this index with:

```bash
python -m src.data_management.generate_json <images_dir> -l <labels_dir> -o <output_json>
```

- Supports configuration via CLI or OmegaConf YAML
- Automatically pairs images and labels (NIfTI formats)
- Output is a structured JSON file for downstream processing

### Dataset Resampling

```bash
# Basic usage
python -m src.data_processing.resample_data configs/resample_config.yml

# Dry run to preview operations
python -m src.data_processing.resample_data configs/resample_config.yml --dry-run

# Force overwrite existing files
python -m src.data_processing.resample_data configs/resample_config.yml --overwrite
```

The script will automatically show which backend is being used:
- `Using MONAI with GPU acceleration for resampling` - GPU mode
- `Using MONAI with CPU for resampling` - MONAI CPU mode
- `Using nibabel + scipy for resampling (CPU)` - Fallback mode

## Performance

Typical resampling time for a 512×512×300 CT scan:

| Backend | Approximate Time |
|---------|------------------|
| MONAI + GPU | 10-20 seconds |
| MONAI + CPU | 40-60 seconds |
| scipy + CPU | 1-2 minutes |

*Actual performance depends on hardware, volume size, and target spacing.*
