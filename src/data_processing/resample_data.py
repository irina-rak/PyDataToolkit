"""Generate a resampled copy of a dataset from a YAML config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
	from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	OmegaConf = None  # type: ignore[assignment]

try:
	import nibabel as nib  # type: ignore
	from nibabel.affines import voxel_sizes  # type: ignore
	from nibabel.processing import resample_to_output  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	nib = None  # type: ignore[assignment]
	voxel_sizes = None  # type: ignore[assignment]
	resample_to_output = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency check
	SCIPY_AVAILABLE = importlib.util.find_spec("scipy.ndimage") is not None
except ModuleNotFoundError:  # pragma: no cover
	SCIPY_AVAILABLE = False

try:
	from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	Progress = None  # type: ignore[assignment]

try:
	import torch  # type: ignore
	from monai.transforms import Spacing  # type: ignore
	MONAI_AVAILABLE = True
	CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover - optional dependency
	torch = None  # type: ignore[assignment]
	Spacing = None  # type: ignore[assignment]
	MONAI_AVAILABLE = False
	CUDA_AVAILABLE = False

INTERPOLATION_TO_ORDER = {
	"nearest": 0,
	"linear": 1,
	"quadratic": 2,
	"cubic": 3,
}

# MONAI interpolation mode mapping
INTERPOLATION_TO_MONAI_MODE = {
	"nearest": "nearest",
	"linear": "bilinear",
	"quadratic": "bilinear",  # MONAI doesn't have quadratic, use bilinear
	"cubic": "trilinear",
}


@dataclass(frozen=True)
class DatasetRecord:
	"""Single data sample with resolved paths for resampling."""

	image_path: Path
	image_relpath: Path
	label_path: Path | None
	label_relpath: Path | None


@dataclass(frozen=True)
class ResampleJobConfig:
	"""Resolved configuration for a resampling run."""

	dataset_index_path: Path
	base_dir: Path
	records: Sequence[DatasetRecord]
	output_images_dir: Path
	output_labels_dir: Path | None
	new_spacing: tuple[float, float, float]
	image_order: int
	label_order: int
	overwrite: bool
	dry_run: bool


def build_parser() -> argparse.ArgumentParser:
	"""
	Build and configure the argument parser for the resampling CLI.

	Returns:
		argparse.ArgumentParser: Configured parser with positional config argument
			and optional --dry-run and --overwrite flags.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Resample a dataset of NIfTI volumes based on an OmegaConf YAML configuration."
		)
	)
	parser.add_argument(
		"--config",
		type=Path,
		required=True,
		help="Path to the resampling configuration YAML file.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="List the operations without writing any files (overrides config).",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Force overwriting existing outputs (overrides config).",
	)
	return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""
	Parse command-line arguments for the resampling script.

	Args:
		argv: Optional sequence of command-line arguments. If None, uses sys.argv.

	Returns:
		argparse.Namespace: Parsed arguments containing config path and optional flags.
	"""
	return build_parser().parse_args(argv)


def load_config_file(config_path: Path) -> Mapping[str, Any]:
	"""
	Load and parse a YAML configuration file using OmegaConf.

	Args:
		config_path: Path to the YAML configuration file to load.

	Returns:
		Mapping[str, Any]: Dictionary-like container with resolved configuration values.

	Raises:
		ValueError: If OmegaConf is not installed, if the file cannot be read,
			if parsing fails, or if the root is not a mapping.
	"""
	if OmegaConf is None:
		raise ValueError(
			"OmegaConf is required to load configuration files. Install it via 'pip install omegaconf'."
		)
	try:
		config = OmegaConf.load(config_path)
	except OSError as error:
		raise ValueError(f"Unable to read config file {config_path}: {error}") from error
	except Exception as error:  # pragma: no cover - defensive catch
		raise ValueError(f"Failed to parse config file {config_path}: {error}") from error

	container = OmegaConf.to_container(config, resolve=True)
	if not isinstance(container, Mapping):
		raise ValueError("The configuration root must be a mapping of key/value pairs.")
	return container


def resolve_path(value: Any, *, base_dir: Path | None, allow_none: bool, field_name: str) -> Path | None:
	"""
	Resolve and validate a path configuration value.

	Converts relative paths to absolute paths relative to base_dir, expands user home
	directory (~), and validates the value is not empty (unless allow_none is True).

	Args:
		value: Configuration value to resolve (typically a string path or None).
		base_dir: Base directory for resolving relative paths. If None and path is
			relative, resolves relative to current working directory.
		allow_none: If True, allows None or empty string values to return None.
		field_name: Name of the configuration field for error messages.

	Returns:
		Path | None: Resolved absolute Path object, or None if value is None/empty
			and allow_none is True.

	Raises:
		ValueError: If value is None/empty and allow_none is False.
	"""
	if value is None:
		if allow_none:
			return None
		raise ValueError(f"Configuration value for '{field_name}' must not be null.")
	text = str(value).strip()
	if not text:
		if allow_none:
			return None
		raise ValueError(f"Configuration value for '{field_name}' must not be empty.")
	path = Path(text).expanduser()
	if not path.is_absolute() and base_dir:
		path = (base_dir / path).resolve()
	else:
		path = path.resolve()
	return path


def coerce_bool(value: Any, field_name: str) -> bool:
	"""
	Convert a configuration value to a boolean.

	Accepts boolean values, strings ('true', 'false', 'yes', 'no', '1', '0', etc.),
	and numeric types.

	Args:
		value: Value to coerce to boolean.
		field_name: Name of the configuration field for error messages.

	Returns:
		bool: The coerced boolean value.

	Raises:
		ValueError: If the value cannot be interpreted as a boolean.
	"""
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		lowered = value.strip().lower()
		if lowered in {"true", "1", "yes", "y", "on"}:
			return True
		if lowered in {"false", "0", "no", "n", "off"}:
			return False
	if isinstance(value, (int, float)):
		return bool(value)
	raise ValueError(f"Configuration value for '{field_name}' must be boolean-compatible.")


def normalise_spacing(values: Any, field_name: str) -> tuple[float, float, float]:
	"""
	Parse and validate a 3D spacing configuration value.

	Args:
		values: Iterable containing three numeric spacing values (x, y, z).
		field_name: Name of the configuration field for error messages.

	Returns:
		tuple[float, float, float]: Tuple of three positive floating-point spacing values.

	Raises:
		ValueError: If values is not iterable, doesn't contain exactly 3 elements,
			or contains non-positive values.
	"""
	if not isinstance(values, Iterable):
		raise ValueError(f"Configuration value for '{field_name}' must be a sequence of numbers.")
	floats = [float(component) for component in values]
	if len(floats) != 3:
		raise ValueError(f"Configuration value for '{field_name}' must contain exactly three numbers.")
	if any(component <= 0 for component in floats):
		raise ValueError(f"Configuration value for '{field_name}' must contain positive numbers.")
	return tuple(floats)  # type: ignore[return-value]


def interpolation_to_order(value: Any, field_name: str) -> int:
	"""
	Convert an interpolation method name to a scipy order integer.

	Args:
		value: Interpolation method name ('nearest', 'linear', 'quadratic', or 'cubic').
		field_name: Name of the configuration field for error messages.

	Returns:
		int: Scipy interpolation order (0 for nearest, 1 for linear, 2 for quadratic,
			3 for cubic).

	Raises:
		ValueError: If value is None or not a supported interpolation method.
	"""
	if value is None:
		raise ValueError(f"Configuration value for '{field_name}' must not be null.")
	key = str(value).strip().lower()
	if key not in INTERPOLATION_TO_ORDER:
		supported = ", ".join(sorted(INTERPOLATION_TO_ORDER))
		raise ValueError(f"Unsupported interpolation '{value}' for '{field_name}'. Use one of: {supported}.")
	return INTERPOLATION_TO_ORDER[key]


def resolve_dataset_path(
	value: Any,
	*,
	base_dir: Path,
	field_name: str,
) -> tuple[Path, Path]:
	"""
	Resolve a dataset path into absolute and relative path components.

	For absolute paths, tries to compute relative path from base_dir; if that fails,
	uses just the filename. For relative paths, treats the value as-is for the
	relative path and resolves absolute path from base_dir.

	Args:
		value: Path string from dataset index entry (image_path or label_path).
		base_dir: Base directory for resolving relative paths and computing
			relative paths from absolute ones.
		field_name: Name of the dataset field for error messages.

	Returns:
		tuple[Path, Path]: Tuple of (absolute_path, relative_path).

	Raises:
		ValueError: If value is None or empty string.
	"""
	if value is None:
		raise ValueError(f"Dataset entry for '{field_name}' must not be null.")
	text = str(value).strip()
	if not text:
		raise ValueError(f"Dataset entry for '{field_name}' must not be empty.")
	path = Path(text).expanduser()
	if path.is_absolute():
		abs_path = path.resolve()
		try:
			rel_path = abs_path.relative_to(base_dir)
		except ValueError:
			rel_path = Path(abs_path.name)
	else:
		rel_path = Path(text)
		abs_path = (base_dir / rel_path).resolve()
	return abs_path, rel_path


def load_dataset_records(index_path: Path, base_dir: Path) -> list[DatasetRecord]:
	"""
	Load and parse a dataset index JSON file into DatasetRecord objects.

	Reads a JSON array of dataset entries, each containing 'image_path' and optional
	'label_path'. Resolves all paths to absolute and relative components.

	Args:
		index_path: Path to the JSON dataset index file.
		base_dir: Base directory for resolving relative paths in the index.

	Returns:
		list[DatasetRecord]: List of parsed and validated dataset records with
			resolved paths.

	Raises:
		ValueError: If file cannot be read, JSON is invalid, not an array, missing
			required fields, or contains no records.
	"""
	try:
		with index_path.open("r", encoding="utf-8") as handle:
			entries = json.load(handle)
	except OSError as error:
		raise ValueError(f"Unable to read dataset index {index_path}: {error}") from error
	except json.JSONDecodeError as error:
		raise ValueError(f"Invalid JSON in dataset index {index_path}: {error}") from error

	if not isinstance(entries, list):
		raise ValueError("Dataset index must be a JSON array of records.")

	records: list[DatasetRecord] = []
	for idx, entry in enumerate(entries):
		if not isinstance(entry, Mapping):
			raise ValueError(f"Dataset record #{idx} must be a JSON object.")
		image_value = entry.get("image_path")
		if image_value is None or str(image_value).strip() == "":
			raise ValueError(f"Dataset record #{idx} is missing 'image_path'.")
		image_path, image_relpath = resolve_dataset_path(
			image_value,
			base_dir=base_dir,
			field_name=f"records[{idx}].image_path",
		)

		label_path_value = entry.get("label_path")
		label_path: Path | None = None
		label_relpath: Path | None = None
		if label_path_value not in (None, ""):
			label_path, label_relpath = resolve_dataset_path(
				label_path_value,
				base_dir=base_dir,
				field_name=f"records[{idx}].label_path",
			)

		records.append(
			DatasetRecord(
				image_path=image_path,
				image_relpath=image_relpath,
				label_path=label_path,
				label_relpath=label_relpath,
			)
		)

	if not records:
		raise ValueError(f"Dataset index {index_path} did not contain any records.")
	return records


def build_job_config(
	raw_config: Mapping[str, Any],
	*,
	config_path: Path,
	cli_overwrite: bool,
	cli_dry_run: bool,
) -> ResampleJobConfig:
	"""
	Build a complete ResampleJobConfig from parsed YAML configuration and CLI flags.

	Validates and resolves all configuration sections including paths, processing
	parameters, and options. CLI flags override configuration file settings.

	Args:
		raw_config: Parsed configuration mapping from YAML file.
		config_path: Path to the configuration file (used to resolve relative paths).
		cli_overwrite: Whether the --overwrite flag was set on the command line.
		cli_dry_run: Whether the --dry-run flag was set on the command line.

	Returns:
		ResampleJobConfig: Validated and resolved configuration ready for processing.

	Raises:
		ValueError: If required sections are missing, paths cannot be resolved,
			parameters are invalid, or the dataset index file is not found.
	"""
	base_dir = config_path.parent

	paths_section = raw_config.get("paths", {})
	if not isinstance(paths_section, Mapping):
		raise ValueError("'paths' section must be a mapping of path values.")

	base_override = paths_section.get("base_dir") or raw_config.get("base_dir")
	if base_override is not None:
		base_dir = resolve_path(base_override, base_dir=config_path.parent, allow_none=False, field_name="paths.base_dir")

	dataset_index_path = resolve_path(
		paths_section.get("dataset_index"),
		base_dir=base_dir,
		allow_none=False,
		field_name="paths.dataset_index",
	)
	if not dataset_index_path.exists():
		raise ValueError(f"Dataset index file not found: {dataset_index_path}")

	records = load_dataset_records(dataset_index_path, base_dir)

	output_images_dir = resolve_path(
		paths_section.get("output_images_dir"),
		base_dir=base_dir,
		allow_none=False,
		field_name="paths.output_images_dir",
	)
	output_labels_dir = resolve_path(
		paths_section.get("output_labels_dir"),
		base_dir=base_dir,
		allow_none=True,
		field_name="paths.output_labels_dir",
	)
	if output_labels_dir is None and any(record.label_path for record in records):
		output_labels_dir = output_images_dir

	processing_section = raw_config.get("processing", {})
	if not isinstance(processing_section, Mapping):
		raise ValueError("'processing' section must be a mapping of processing parameters.")

	new_spacing = normalise_spacing(processing_section.get("new_spacing", (1.0, 1.0, 1.0)), "processing.new_spacing")
	image_order = interpolation_to_order(
		processing_section.get("image_interpolation", "linear"),
		"processing.image_interpolation",
	)
	label_order = interpolation_to_order(
		processing_section.get("label_interpolation", "nearest"),
		"processing.label_interpolation",
	)

	options_section = raw_config.get("options", {})
	if not isinstance(options_section, Mapping):
		raise ValueError("'options' section must be a mapping of optional parameters.")

	overwrite = coerce_bool(options_section.get("overwrite", False), "options.overwrite")
	dry_run = coerce_bool(options_section.get("dry_run", False), "options.dry_run")

	if cli_overwrite:
		overwrite = True
	if cli_dry_run:
		dry_run = True

	return ResampleJobConfig(
		dataset_index_path=dataset_index_path,
		base_dir=base_dir,
		records=records,
		output_images_dir=output_images_dir,
		output_labels_dir=output_labels_dir,
		new_spacing=new_spacing,
		image_order=image_order,
		label_order=label_order,
		overwrite=overwrite,
		dry_run=dry_run,
	)


def resample_with_monai(
	input_path: Path,
	output_path: Path,
	new_spacing: tuple[float, float, float],
	mode: str,
	*,
	round_discrete: bool,
	use_gpu: bool = True,
) -> tuple[tuple[int, ...], tuple[float, float, float]]:
	"""
	Resample a NIfTI volume using MONAI (GPU-accelerated when available).

	Args:
		input_path: Path to input NIfTI file (.nii or .nii.gz).
		output_path: Path where resampled volume will be saved.
		new_spacing: Target voxel spacing in millimeters (x, y, z).
		mode: MONAI interpolation mode ('nearest', 'bilinear', 'trilinear').
		round_discrete: If True, rounds voxel values to nearest integer.
		use_gpu: If True and CUDA available, use GPU acceleration.

	Returns:
		tuple[tuple[int, ...], tuple[float, float, float]]: Tuple containing
			(output_shape, output_spacing) of the resampled volume.

	Raises:
		RuntimeError: If MONAI or required dependencies are not available.
		OSError: If input file cannot be read or output file cannot be written.
	"""
	if not MONAI_AVAILABLE or nib is None:
		raise RuntimeError("MONAI and nibabel are required for GPU-accelerated resampling.")

	# Load volume with nibabel
	volume = nib.load(str(input_path))
	original_dtype = volume.get_data_dtype()
	data = volume.get_fdata(dtype=np.float32)
	affine = volume.affine

	# Get original spacing from affine matrix
	original_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
	
	# Convert numpy array to torch tensor
	# NIfTI shape is (X, Y, Z), MONAI expects (C, X, Y, Z) where C is channels
	data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add channel dimension: [1, X, Y, Z]

	# Move to GPU if available and requested
	device = torch.device("cuda" if use_gpu and CUDA_AVAILABLE else "cpu")
	data_tensor = data_tensor.to(device)

	# Calculate target size based on spacing ratio
	# new_size = original_size * (original_spacing / new_spacing)
	zoom_factors = original_spacing / np.array(new_spacing)
	target_shape = tuple(int(round(dim * zoom)) for dim, zoom in zip(data.shape, zoom_factors))

	# Use PyTorch's interpolate for resampling
	# Add batch dimension for interpolate: [1, C, X, Y, Z]
	data_tensor = data_tensor.unsqueeze(0)  # [1, 1, X, Y, Z]
	
	from torch.nn.functional import interpolate  # type: ignore
	
	# Resample using trilinear/nearest interpolation
	if mode == 'nearest':
		resampled_tensor = interpolate(
			data_tensor,
			size=target_shape,
			mode='nearest',
		)
	else:
		# Use trilinear for bilinear/trilinear modes
		resampled_tensor = interpolate(
			data_tensor,
			size=target_shape,
			mode='trilinear',
			align_corners=True,
		)

	# Remove batch and channel dimensions and move back to CPU
	resampled_data = resampled_tensor.squeeze(0).squeeze(0).cpu().numpy()

	# Round if discrete labels
	if round_discrete:
		resampled_data = np.rint(resampled_data)

	# Convert back to original dtype
	resampled_data = resampled_data.astype(original_dtype, copy=False)

	# Compute new affine matrix
	# Scale each direction by the ratio of new/old spacing
	scale_factors = np.array(new_spacing) / original_spacing
	new_affine = affine.copy()
	for i in range(3):
		new_affine[:3, i] = affine[:3, i] * scale_factors[i]

	# Create new NIfTI image
	resampled_img = nib.Nifti1Image(resampled_data, new_affine, volume.header)
	resampled_img.set_data_dtype(original_dtype)

	# Save output
	output_path.parent.mkdir(parents=True, exist_ok=True)
	nib.save(resampled_img, str(output_path))

	# Return actual shape and spacing
	new_shape = resampled_data.shape
	actual_spacing = tuple(float(s) for s in np.sqrt((new_affine[:3, :3] ** 2).sum(axis=0)))

	return new_shape, actual_spacing


def resample_with_scipy(
	input_path: Path,
	output_path: Path,
	new_spacing: tuple[float, float, float],
	order: int,
	*,
	round_discrete: bool,
) -> tuple[tuple[int, ...], tuple[float, float, float]]:
	"""
	Resample a NIfTI volume using nibabel + scipy (CPU-based).

	Args:
		input_path: Path to input NIfTI file (.nii or .nii.gz).
		output_path: Path where resampled volume will be saved.
		new_spacing: Target voxel spacing in millimeters (x, y, z).
		order: Scipy interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic).
		round_discrete: If True, rounds voxel values to nearest integer.

	Returns:
		tuple[tuple[int, ...], tuple[float, float, float]]: Tuple containing
			(output_shape, output_spacing) of the resampled volume.

	Raises:
		RuntimeError: If nibabel is not available.
		OSError: If input file cannot be read or output file cannot be written.
	"""
	if nib is None or resample_to_output is None:
		raise RuntimeError("nibabel is not available; cannot resample volumes.")

	volume = nib.load(str(input_path))
	resampled = resample_to_output(volume, voxel_sizes=new_spacing, order=order)

	original_dtype = volume.get_data_dtype()
	data = resampled.get_fdata(dtype=np.float32)
	if round_discrete:
		data = np.rint(data)
	data = data.astype(original_dtype, copy=False)
	resampled = nib.Nifti1Image(data, resampled.affine, resampled.header)
	resampled.set_data_dtype(original_dtype)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	nib.save(resampled, str(output_path))

	new_shape = resampled.shape
	new_voxel_sizes = tuple(float(size) for size in voxel_sizes(resampled.affine)) if voxel_sizes else new_spacing
	return new_shape, new_voxel_sizes


def resample_and_save(
	input_path: Path,
	output_path: Path,
	new_spacing: tuple[float, float, float],
	order: int,
	*,
	round_discrete: bool,
) -> tuple[tuple[int, ...], tuple[float, float, float]]:
	"""
	Resample a NIfTI volume to a new voxel spacing and save the result.

	Automatically selects the best available backend:
	1. MONAI with GPU (fastest) - if CUDA is available
	2. MONAI with CPU - if MONAI is installed but no GPU
	3. nibabel + scipy (fallback) - reliable CPU-based method

	Args:
		input_path: Path to input NIfTI file (.nii or .nii.gz).
		output_path: Path where resampled volume will be saved (directory created if needed).
		new_spacing: Target voxel spacing in millimeters (x, y, z).
		order: Scipy interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic).
		round_discrete: If True, rounds voxel values to nearest integer after
			resampling (use for segmentation labels).

	Returns:
		tuple[tuple[int, ...], tuple[float, float, float]]: Tuple containing
			(output_shape, output_spacing) of the resampled volume.

	Raises:
		RuntimeError: If no resampling backend is available.
		OSError: If input file cannot be read or output file cannot be written.
	"""
	# Try MONAI first (GPU or CPU)
	if MONAI_AVAILABLE:
		# Map scipy order to MONAI mode
		mode = INTERPOLATION_TO_MONAI_MODE.get(
			{0: "nearest", 1: "linear", 2: "quadratic", 3: "cubic"}.get(order, "linear"),
			"bilinear"
		)
		try:
			return resample_with_monai(
				input_path=input_path,
				output_path=output_path,
				new_spacing=new_spacing,
				mode=mode,
				round_discrete=round_discrete,
				use_gpu=CUDA_AVAILABLE,
			)
		except Exception as e:
			# If MONAI fails, fall back to scipy
			print(f"[WARN] MONAI resampling failed ({e}), falling back to scipy", file=sys.stderr)

	# Fallback to scipy
	if nib is None or resample_to_output is None:
		raise RuntimeError("No resampling backend available. Install nibabel+scipy or MONAI.")

	return resample_with_scipy(
		input_path=input_path,
		output_path=output_path,
		new_spacing=new_spacing,
		order=order,
		round_discrete=round_discrete,
	)


def process_dataset(config: ResampleJobConfig) -> int:
	"""
	Process all dataset records according to the resampling configuration.

	Iterates through dataset index records, resamples images and labels to specified
	spacing, and saves outputs. Handles missing files, existing outputs (with overwrite
	option), and dry-run mode. Prints progress and summary statistics.

	Args:
		config: Complete resampling job configuration including dataset records,
			output paths, spacing, interpolation orders, and processing options.

	Returns:
		int: Exit code (0 for success).
	"""
	processed_images = 0
	processed_labels = 0
	skipped_existing = 0
	missing_images = 0
	missing_labels = 0

	# Use rich progress bar if available, otherwise fall back to simple print
	if Progress is not None and not config.dry_run:
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			BarColumn(),
			TaskProgressColumn(),
			TimeRemainingColumn(),
		) as progress:
			task = progress.add_task(
				f"Resampling {len(config.records)} volume(s)...",
				total=len(config.records)
			)

			for record in config.records:
				image_path = record.image_path
				progress.update(task, description=f"Processing {image_path.name}...")

				if not image_path.exists():
					progress.console.print(f"[yellow][WARN] Image not found: {image_path}[/yellow]")
					missing_images += 1
					progress.advance(task)
					continue

				output_image_path = config.output_images_dir / image_path.name

				if output_image_path.exists() and not config.overwrite:
					skipped_existing += 1
					progress.advance(task)
					continue

				label_input_path: Path | None = None
				label_output_path: Path | None = None
				if record.label_path is not None:
					if record.label_path.exists():
						label_input_path = record.label_path
						label_output_root = config.output_labels_dir or config.output_images_dir
						label_output_path = label_output_root / record.label_path.name
					else:
						progress.console.print(f"[yellow][WARN] Label not found: {record.label_path}[/yellow]")
						missing_labels += 1

				try:
					image_shape, image_spacing = resample_and_save(
						input_path=image_path,
						output_path=output_image_path,
						new_spacing=config.new_spacing,
						order=config.image_order,
						round_discrete=False,
					)
					processed_images += 1
				except Exception as error:  # pragma: no cover - runtime safeguard
					progress.console.print(f"[red][ERROR] Failed to resample image {image_path}: {error}[/red]")
					progress.advance(task)
					continue

				if label_input_path and label_output_path:
					try:
						resample_and_save(
							input_path=label_input_path,
							output_path=label_output_path,
							new_spacing=config.new_spacing,
							order=config.label_order,
							round_discrete=True,
						)
						processed_labels += 1
					except Exception as error:  # pragma: no cover - runtime safeguard
						progress.console.print(
							f"[red][ERROR] Failed to resample label {label_input_path}: {error}[/red]"
						)
				
				progress.advance(task)
				progress.console.print(
                    f"Resampled [green]{image_path.name}[/green] -> [green]{output_image_path.name}[/green] (shape={image_shape}, spacing={image_spacing})."
                )
	else:
		# Fallback to simple print statements if rich is not available or dry-run mode
		print(
			f"Loaded {len(config.records)} record(s) from {config.dataset_index_path}."
			+ (" (dry-run)" if config.dry_run else "")
		)

		for record in config.records:
			image_path = record.image_path
			if not image_path.exists():
				print(f"[WARN] Image not found: {image_path}", file=sys.stderr)
				missing_images += 1
				continue

			output_image_path = config.output_images_dir / image_path.name

			if output_image_path.exists() and not config.overwrite:
				skipped_existing += 1
				continue

			label_input_path: Path | None = None
			label_output_path: Path | None = None
			if record.label_path is not None:
				if record.label_path.exists():
					label_input_path = record.label_path
					label_output_root = config.output_labels_dir or config.output_images_dir
					label_output_path = label_output_root / record.label_path.name
				else:
					print(f"[WARN] Label not found: {record.label_path}", file=sys.stderr)
					missing_labels += 1

			if config.dry_run:
				print(f"[DRY-RUN] Would resample {image_path} -> {output_image_path}")
				if label_input_path and label_output_path:
					print(f"[DRY-RUN] Would resample {label_input_path} -> {label_output_path}")
				continue

			try:
				image_shape, image_spacing = resample_and_save(
					input_path=image_path,
					output_path=output_image_path,
					new_spacing=config.new_spacing,
					order=config.image_order,
					round_discrete=False,
				)
				processed_images += 1
			except Exception as error:  # pragma: no cover - runtime safeguard
				print(f"[ERROR] Failed to resample image {image_path}: {error}", file=sys.stderr)
				continue

			if label_input_path and label_output_path:
				try:
					resample_and_save(
						input_path=label_input_path,
						output_path=label_output_path,
						new_spacing=config.new_spacing,
						order=config.label_order,
						round_discrete=True,
					)
					processed_labels += 1
				except Exception as error:  # pragma: no cover - runtime safeguard
					print(
						f"[ERROR] Failed to resample label {label_input_path}: {error}",
						file=sys.stderr,
					)

			print(
				f"Resampled {image_path.name} -> {output_image_path.name} (shape={image_shape}, spacing={image_spacing})."
			)

	if skipped_existing:
		print(f"Skipped {skipped_existing} file(s) that already existed.")
	if missing_images:
		print(f"Warning: {missing_images} image file(s) were not found.", file=sys.stderr)
	if missing_labels:
		print(f"Warning: {missing_labels} label file(s) were not found.", file=sys.stderr)

	print(f"Finished processing {processed_images} image(s) and {processed_labels} label(s).")
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	"""
	Main entry point for the resampling CLI application.

	Parses command-line arguments, validates dependencies (nibabel, scipy), loads
	configuration, creates output directories, and initiates dataset processing.

	Args:
		argv: Optional sequence of command-line arguments. If None, uses sys.argv.

	Returns:
		int: Exit code (0 for success, 1 for errors).
	"""
	args = parse_args(argv)

	# Check for resampling backend availability
	if MONAI_AVAILABLE and CUDA_AVAILABLE:
		print("Using MONAI with GPU acceleration for resampling (FASTEST)")
	elif MONAI_AVAILABLE:
		print("Using MONAI with CPU for resampling")
	elif nib is not None and SCIPY_AVAILABLE:
		print("Using nibabel + scipy for resampling (CPU)")
	else:
		print(
			"No resampling backend available. Install nibabel+scipy or MONAI+torch.",
			file=sys.stderr,
		)
		return 1

	if nib is None:
		print(
			"nibabel is required to load NIfTI volumes. Install it via 'pip install nibabel'.",
			file=sys.stderr,
		)
		return 1

	# Scipy check is now optional if MONAI is available
	if not SCIPY_AVAILABLE and not MONAI_AVAILABLE:
		print(
			"Either scipy or MONAI is required for resampling. Install via 'pip install scipy' or 'pip install monai torch'.",
			file=sys.stderr,
		)
		return 1

	config_path = args.config.expanduser().resolve()
	if not config_path.exists():
		print(f"Config file not found: {config_path}", file=sys.stderr)
		return 1
	if config_path.is_dir():
		print(f"Config path must reference a file, not a directory: {config_path}", file=sys.stderr)
		return 1

	try:
		raw_config = load_config_file(config_path)
		job_config = build_job_config(
			raw_config,
			config_path=config_path,
			cli_overwrite=bool(getattr(args, "overwrite", False)),
			cli_dry_run=bool(getattr(args, "dry_run", False)),
		)
	except ValueError as error:
		print(error, file=sys.stderr)
		return 1

	if not job_config.dry_run:
		job_config.output_images_dir.mkdir(parents=True, exist_ok=True)
		if job_config.output_labels_dir is not None:
			job_config.output_labels_dir.mkdir(parents=True, exist_ok=True)

	return process_dataset(job_config)


if __name__ == "__main__":
	raise SystemExit(main())
