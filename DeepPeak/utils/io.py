from pathlib import Path
import re


def build_trace_files_from_folder(
    folder: str | Path,
    filename_regex: str = r"^dilution_(\d+)x_\d+\.csv$",
) -> list[tuple[str, int]]:
    """
    Find dilution trace files in a folder and build a PeakCountSeries-compatible
    trace_files list.

    Parameters
    ----------
    folder:
        Folder containing the trace CSV files.

    filename_regex:
        Regex used to identify trace files. It must contain one capture group
        corresponding to the dilution factor.

        Default matches filenames like:
            dilution_1000x_1.csv
            dilution_300x_1.csv
            dilution_10x_1.csv

    Returns
    -------
    trace_files:
        List of tuples:

            [
                ("dilution_1000x_1.csv", 1000),
                ("dilution_300x_1.csv", 300),
                ...
            ]

        Sorted from largest dilution factor to smallest.
    """
    folder = Path(folder)
    compiled_filename_regex = re.compile(filename_regex)

    trace_files: list[tuple[str, int]] = []

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        match = compiled_filename_regex.match(file_path.name)

        if match is None:
            continue

        dilution_factor = int(match.group(1))
        trace_files.append((file_path.name, dilution_factor))

    trace_files.sort(key=lambda item: item[1], reverse=True)

    return trace_files
