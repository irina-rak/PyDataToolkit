"""CLI tool to index dataset images (and optional labels) into a JSON file."""

from __future__ import annotations

import argparse
import json
import sys

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
	from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	OmegaConf = None  # type: ignore[assignment]

try:
	import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	pd = None  # type: ignore[assignment]


DEFAULT_IMAGE_EXTENSIONS = {
	".nii",
	".nii.gz",
}

DEFAULT_LABEL_EXTENSIONS = {
	".nii",
	".nii.gz",
}


@dataclass(frozen=True)
class DatasetEntry:
	"""Single record to be written to the output file."""

	image_path: Path
	label_path: Path | None = None


CONFIG_FIELDS_ORDER = (
	"images_dir",
	"labels_dir",
	"output",
	"image_extensions",
	"label_extensions",
	"recursive",
	"relative_to",
	"force",
)

CONFIG_FIELDS = set(CONFIG_FIELDS_ORDER)


def build_parser() -> argparse.ArgumentParser:
	"""
	Build and configure the argument parser for the dataset indexing CLI.

	Returns:
		argparse.ArgumentParser: Configured parser with optional config file path,
			positional images_dir argument, and various optional flags for labels,
			output path, extensions, recursion, and path formatting.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Generate a JSON dataset index listing image paths and, when available,"
			" corresponding label files."
		)
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=None,
		help=(
			"Optional OmegaConf-compatible config file providing the same keys as the"
			" CLI arguments. CLI values override config values when both are supplied."
		),
	)
	parser.add_argument(
		"images_dir",
		type=Path,
		nargs="?",
		help="Directory containing the dataset images.",
	)
	parser.add_argument(
		"-l",
		"--labels-dir",
		type=Path,
		default=None,
		help="Directory containing label files paired with the images.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path("dataset_index.json"),
		help="Path to the JSON file that will be generated (default: dataset_index.json).",
	)
	parser.add_argument(
		"--image-extensions",
		nargs="+",
		default=sorted(DEFAULT_IMAGE_EXTENSIONS),
		help=(
			"List of image extensions to index (case-insensitive). "
			"Defaults to medical imaging formats .nii and .nii.gz."
		),
	)
	parser.add_argument(
		"--label-extensions",
		nargs="+",
		default=sorted(DEFAULT_LABEL_EXTENSIONS),
		help=(
			"List of label extensions considered when pairing with images. "
			"Defaults to .nii and .nii.gz."
		),
	)
	parser.add_argument(
		"-r",
		"--recursive",
		action="store_true",
		help="Search directories recursively for images and labels.",
	)
	parser.add_argument(
		"--relative-to",
		type=Path,
		default=None,
		help=(
			"Make the stored paths relative to this directory when possible."
			" Defaults to absolute paths."
		),
	)
	parser.add_argument(
		"-f",
		"--force",
		action="store_true",
		help="Overwrite the output file if it already exists.",
	)
	return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""
	Parse command-line arguments for the dataset indexing script.

	Args:
		argv: Optional sequence of command-line arguments. If None, uses sys.argv.

	Returns:
		argparse.Namespace: Parsed arguments containing config path, images_dir,
			and various indexing options.
	"""
	return build_parser().parse_args(argv)


def collect_cli_overrides(
	parser: argparse.ArgumentParser, namespace: argparse.Namespace
) -> dict[str, Any]:
	"""
	Extract CLI arguments that differ from their defaults.

	Identifies which arguments were explicitly provided on the command line by
	comparing parsed values against their defaults. Used to determine which CLI
	values should override configuration file settings.

	Args:
		parser: The argument parser containing action definitions and defaults.
		namespace: Parsed command-line arguments from parse_args().

	Returns:
		dict[str, Any]: Dictionary mapping field names to explicitly provided
			CLI values (excluding defaults and help/config fields).
	"""
	overrides: dict[str, Any] = {}
	for action in parser._actions:
		dest = action.dest
		if not dest or dest in {"help", "config"}:
			continue
		if dest not in CONFIG_FIELDS:
			continue
		value = getattr(namespace, dest)
		default = action.default
		if dest == "images_dir" and value is None:
			continue
		if value is None and default is None:
			continue
		if value is default:
			continue
		if isinstance(value, list) and value == default:
			continue
		if value == default and not isinstance(value, bool):
			continue
		overrides[dest] = value
	return overrides


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


def resolve_path_from_config(
	value: Any, *, base_dir: Path | None, allow_none: bool, field_name: str
) -> Path | None:
	"""
	Resolve and validate a path configuration value.

	Converts relative paths to absolute paths relative to base_dir, expands user home
	directory (~), and validates the value is not empty (unless allow_none is True).

	Args:
		value: Configuration value to resolve (typically a string path or Path object).
		base_dir: Base directory for resolving relative paths. If None and path is
			relative, keeps path relative.
		allow_none: If True, allows None or empty string values to return None.
		field_name: Name of the configuration field for error messages.

	Returns:
		Path | None: Resolved Path object (may be relative or absolute), or None
			if value is None/empty and allow_none is True.

	Raises:
		ValueError: If value is None/empty and allow_none is False.
	"""
	if value is None:
		if allow_none:
			return None
		raise ValueError(f"Configuration value for '{field_name}' must not be null.")
	if isinstance(value, (str, Path)) and not str(value).strip():
		if allow_none:
			return None
		raise ValueError(f"Configuration value for '{field_name}' must not be empty.")
	if isinstance(value, Path):
		path = value
	else:
		path = Path(str(value)).expanduser()
	if not path.is_absolute() and base_dir:
		path = (base_dir / path).expanduser()
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


def coerce_extension_values(value: Any, field_name: str) -> list[str]:
	"""
	Parse and normalize file extension values from configuration.

	Accepts strings (comma or semicolon separated), lists, tuples, or sets of
	extension values. Filters out empty strings.

	Args:
		value: Extension value(s) to parse - can be a delimited string or sequence.
		field_name: Name of the configuration field for error messages.

	Returns:
		list[str]: List of normalized extension strings (whitespace stripped).

	Raises:
		ValueError: If value is None or not a supported type.
	"""
	if value is None:
		raise ValueError(f"Configuration value for '{field_name}' must not be null.")
	if isinstance(value, str):
		candidates = [part.strip() for part in value.replace(";", ",").split(",")]
		return [candidate for candidate in candidates if candidate]
	if isinstance(value, (list, tuple, set)):
		return [str(item).strip() for item in value if str(item).strip()]
	raise ValueError(
		f"Configuration value for '{field_name}' must be a string or a sequence of strings."
	)


def merge_parameters(
	parser: argparse.ArgumentParser,
	args: argparse.Namespace,
	config_values: Mapping[str, Any] | None,
	config_base_dir: Path | None,
) -> dict[str, Any]:
	"""
	Merge configuration file values with CLI arguments.

	CLI arguments take precedence over configuration file values. Validates and
	resolves paths, coerces types, and warns about unknown configuration keys.

	Args:
		parser: The argument parser (used to extract CLI overrides).
		args: Parsed command-line arguments.
		config_values: Optional dictionary of configuration file values.
		config_base_dir: Base directory for resolving relative paths from config file.

	Returns:
		dict[str, Any]: Merged configuration dictionary with CLI overrides applied
			and all values properly typed and resolved.
	"""
	config_values = config_values or {}

	unknown_keys = sorted(set(config_values) - CONFIG_FIELDS)
	if unknown_keys:
		print(
			f"[WARN] Ignoring unknown configuration keys: {', '.join(unknown_keys)}",
			file=sys.stderr,
		)

	cli_overrides = collect_cli_overrides(parser, args)
	merged: dict[str, Any] = {}

	for field in CONFIG_FIELDS_ORDER:
		if field in cli_overrides:
			merged[field] = cli_overrides[field]
			continue
		if field in config_values:
			value = config_values[field]
			if field in {"images_dir", "output"}:
				merged[field] = resolve_path_from_config(
					value, base_dir=config_base_dir, allow_none=False, field_name=field
				)
			elif field in {"labels_dir", "relative_to"}:
				merged[field] = resolve_path_from_config(
					value, base_dir=config_base_dir, allow_none=True, field_name=field
				)
			elif field in {"recursive", "force"}:
				merged[field] = coerce_bool(value, field)
			elif field in {"image_extensions", "label_extensions"}:
				merged[field] = coerce_extension_values(value, field)
			else:  # pragma: no cover - defensive default
				merged[field] = value
			continue
		merged[field] = getattr(args, field)

	return merged


def normalise_extensions(values: Iterable[str]) -> set[str]:
	"""
	Normalize file extensions to lowercase with leading dots.

	Args:
		values: Iterable of file extension strings.

	Returns:
		set[str]: Set of normalized extensions (lowercase, with leading dot).
	"""
	return {ensure_dot_prefix(value.strip().lower()) for value in values}


def ensure_dot_prefix(ext: str) -> str:
	"""
	Ensure a file extension string starts with a dot.

	Args:
		ext: File extension string (with or without leading dot).

	Returns:
		str: Extension string guaranteed to start with a dot.
	"""
	return ext if ext.startswith(".") else f".{ext}"


def iter_files(directory: Path, extensions: set[str], recursive: bool) -> Iterable[Path]:
	"""
	Iterate over files in a directory matching specified extensions.

	Args:
		directory: Directory to search for files.
		extensions: Set of normalized file extensions (lowercase, with dots).
		recursive: If True, search recursively in subdirectories.

	Yields:
		Path: Resolved absolute paths to matching files.
	"""
	walker = directory.rglob("*") if recursive else directory.glob("*")
	for path in walker:
		if path.is_file() and extension_matches(path, extensions):
			yield path.resolve()


def extension_matches(path: Path, extensions: set[str]) -> bool:
	"""
	Check if a file path's extension matches any in the provided set.

	Handles both simple extensions (.nii) and compound extensions (.nii.gz).
	Comparison is case-insensitive.

	Args:
		path: File path to check.
		extensions: Set of normalized extensions to match against.

	Returns:
		bool: True if the file's extension matches any in the set, False otherwise.
	"""
	if not extensions:
		return False
	suffixes = [suffix.lower() for suffix in path.suffixes if suffix]
	if not suffixes:
		return "" in extensions
	for index, suffix in enumerate(suffixes):
		if suffix in extensions:
			return True
		compound = "".join(suffixes[index:])
		if compound in extensions:
			return True
	return False


def build_label_lookup(
	labels: Iterable[Path],
	labels_root: Path,
) -> tuple[Mapping[str, list[Path]], Mapping[str, list[Path]]]:
	"""
	Build lookup dictionaries for efficiently matching labels to images.

	Creates two indexes: one based on relative paths (without extension) and one
	based on file stem (basename without extension). Both enable fast label lookup.

	Args:
		labels: Iterable of label file paths to index.
		labels_root: Root directory for computing relative paths.

	Returns:
		tuple[Mapping[str, list[Path]], Mapping[str, list[Path]]]: Tuple of
			(relative_path_lookup, stem_lookup) dictionaries mapping keys to
			lists of matching label paths.
	"""
	by_relative: dict[str, list[Path]] = {}
	by_stem: dict[str, list[Path]] = {}
	for label_path in labels:
		try:
			relative_key = label_path.relative_to(labels_root).with_suffix("")
		except ValueError:
			relative_key = label_path.name
		key_str = str(relative_key)
		by_relative.setdefault(key_str, []).append(label_path)
		by_stem.setdefault(label_path.stem.lower(), []).append(label_path)
	return by_relative, by_stem


def find_label_for_image(
	image: Path,
	images_root: Path,
	relative_lookup: Mapping[str, list[Path]],
	stem_lookup: Mapping[str, list[Path]],
) -> Path | None:
	"""
	Find a matching label file for an image.

	First attempts to match by relative path (preferred for hierarchical datasets),
	then falls back to matching by file stem. Warns if multiple matches are found.

	Args:
		image: Image file path to find a label for.
		images_root: Root directory for computing image's relative path.
		relative_lookup: Dictionary mapping relative paths to label file lists.
		stem_lookup: Dictionary mapping file stems to label file lists.

	Returns:
		Path | None: Path to the matching label file, or None if no match found.
			If multiple matches exist, returns the first and emits a warning.
	"""
	try:
		rel_key = image.relative_to(images_root).with_suffix("")
		candidates = relative_lookup.get(str(rel_key), [])
	except ValueError:
		candidates = []

	if len(candidates) == 1:
		return candidates[0]
	if len(candidates) > 1:
		_warn_duplicate_match(image, candidates)
		return candidates[0]

	stem_candidates = stem_lookup.get(image.stem.lower(), [])
	if len(stem_candidates) == 1:
		return stem_candidates[0]
	if len(stem_candidates) > 1:
		_warn_duplicate_match(image, stem_candidates)
		return stem_candidates[0]
	return None


def _warn_duplicate_match(image: Path, candidates: Sequence[Path]) -> None:
	"""
	Print a warning message when multiple label files match a single image.

	Args:
		image: The image file path that has multiple label matches.
		candidates: Sequence of label paths that matched the image.
	"""
	joined = ", ".join(str(candidate) for candidate in candidates)
	print(
		f"[WARN] Multiple label matches for {image}: {joined}. Using the first entry.",
		file=sys.stderr,
	)


def prepare_entries(
	images: Sequence[Path],
	images_root: Path,
	labels_root: Path | None,
	relative_lookup: Mapping[str, list[Path]] | None,
	stem_lookup: Mapping[str, list[Path]] | None,
) -> list[DatasetEntry]:
	"""
	Create dataset entries pairing images with their corresponding labels.

	Args:
		images: Sequence of image file paths to process.
		images_root: Root directory for computing image relative paths.
		labels_root: Optional root directory for label files. If None, entries
			will have no labels.
		relative_lookup: Optional dictionary for finding labels by relative path.
		stem_lookup: Optional dictionary for finding labels by file stem.

	Returns:
		list[DatasetEntry]: List of dataset entries with paired image and label
			paths (label may be None if not found or labels not provided).
	"""
	entries: list[DatasetEntry] = []
	for image_path in sorted(images):
		label_path = None
		if labels_root and relative_lookup and stem_lookup:
			label_path = find_label_for_image(
				image=image_path,
				images_root=images_root,
				relative_lookup=relative_lookup,
				stem_lookup=stem_lookup,
			)
		entries.append(DatasetEntry(image_path=image_path, label_path=label_path))
	return entries


def format_path(path: Path, relative_to: Path | None) -> str:
	"""
	Format a path as a string, optionally relative to a base directory.

	Args:
		path: Path to format.
		relative_to: Optional base directory to compute relative path from.
			If None or relative path computation fails, returns absolute path.

	Returns:
		str: String representation of the path (relative if possible, else absolute).
	"""
	if relative_to:
		try:
			return str(path.relative_to(relative_to))
		except ValueError:
			pass
	return str(path)


def ensure_writable_path(output_path: Path, force: bool) -> None:
	"""
	Validate output path is writable and create parent directories.

	Args:
		output_path: Path where output file will be written.
		force: If False, raises error if file already exists. If True, allows overwrite.

	Raises:
		FileExistsError: If output_path exists and force is False.
	"""
	if output_path.exists() and not force:
		raise FileExistsError(
			f"Output file {output_path} already exists. Use --force to overwrite."
		)
	output_path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
	"""
	Main entry point for the dataset indexing CLI application.

	Discovers image files (and optionally label files) in specified directories,
	pairs them based on filename matching, and generates a JSON index file with
	all dataset entries. Supports configuration files and CLI overrides.

	Args:
		argv: Optional sequence of command-line arguments. If None, uses sys.argv.

	Returns:
		int: Exit code (0 for success, 1 for errors).
	"""
	parser = build_parser()
	args = parser.parse_args(argv)

	if pd is None:
		print(
			"pandas is required to generate the dataset index. Install it via 'pip install pandas', or sync your environment with the pyproject.toml file.",
			file=sys.stderr,
		)
		return 1

	config_mapping: Mapping[str, Any] | None = None
	config_base_dir: Path | None = None

	if args.config:
		config_path = args.config.expanduser()
		if not config_path.exists():
			print(f"Config file not found: {config_path}", file=sys.stderr)
			return 1
		if config_path.is_dir():
			print(f"Config path must be a file: {config_path}", file=sys.stderr)
			return 1
		try:
			config_mapping = load_config_file(config_path)
		except ValueError as error:
			print(error, file=sys.stderr)
			return 1
		config_base_dir = config_path.parent

	try:
		merged = merge_parameters(parser, args, config_mapping, config_base_dir)
	except ValueError as error:
		print(error, file=sys.stderr)
		return 1

	if merged["images_dir"] is None:
		print(
			"Images directory must be provided via CLI or configuration file.",
			file=sys.stderr,
		)
		return 1

	images_dir = Path(merged["images_dir"]).expanduser().resolve()
	labels_dir = (
		Path(merged["labels_dir"]).expanduser().resolve() if merged["labels_dir"] else None
	)
	output_path = Path(merged["output"]).expanduser().resolve()
	relative_to = (
		Path(merged["relative_to"]).expanduser().resolve() if merged["relative_to"] else None
	)

	if not images_dir.is_dir():
		print(f"Images directory not found: {images_dir}", file=sys.stderr)
		return 1
	if labels_dir and not labels_dir.is_dir():
		print(f"Labels directory not found: {labels_dir}", file=sys.stderr)
		return 1

	image_exts = normalise_extensions(merged["image_extensions"])
	label_exts = normalise_extensions(merged["label_extensions"])

	recursive = bool(merged["recursive"])
	force = bool(merged["force"])

	images = list(iter_files(images_dir, image_exts, recursive))
	if not images:
		print(f"No images found in {images_dir} with extensions {sorted(image_exts)}.", file=sys.stderr)
		return 1

	relative_lookup = None
	stem_lookup = None
	labels = []

	if labels_dir:
		labels = list(iter_files(labels_dir, label_exts, recursive))
		if not labels:
			print(
				f"Warning: no labels found in {labels_dir} with extensions {sorted(label_exts)}.",
				file=sys.stderr,
			)
		relative_lookup, stem_lookup = build_label_lookup(labels, labels_dir)

	entries = prepare_entries(
		images=images,
		images_root=images_dir,
		labels_root=labels_dir,
		relative_lookup=relative_lookup,
		stem_lookup=stem_lookup,
	)

	ensure_writable_path(output_path, force)

	has_labels = labels_dir is not None
	rows: list[dict[str, Any]] = []
	for entry in entries:
		row: dict[str, Any] = {
			"image_path": format_path(entry.image_path, relative_to),
		}
		if has_labels:
			row["label_path"] = (
				format_path(entry.label_path, relative_to) if entry.label_path else None
			)
		rows.append(row)

	frame = pd.DataFrame(rows)
	records = frame.to_dict(orient="records")
	with output_path.open("w", encoding="utf-8") as json_file:
		json.dump(records, json_file, indent=2)

	print(
		(
			f"Indexed {len(entries)} images"
			+ (f" and {len(labels)} labels" if labels_dir else "")
			+ f" into {output_path}."
		)
	)

	unmatched = sum(1 for entry in entries if has_labels and not entry.label_path)
	if has_labels and unmatched:
		print(
			f"Warning: {unmatched} image(s) did not have a matching label.",
			file=sys.stderr,
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
