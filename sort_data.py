import argparse
import shutil
from pathlib import Path
from astropy.io import fits

def sort_hst_data(base_dir, target_id):

    """
    Identifies HST x1d.fits files by TARGNAME header and organizes them
    into target-specific directories in the current working directory.

    Args:
        base_dir (str): Path to the root MAST data download.
        target_id (str): 3-digit SDSS-RM target ID (e.g., '392').
    """

    base_path = Path(base_dir)
    # The full target name in the FITS header
    full_targname = f"SDSSRM-{target_id}"

    # 1. Source: Where the raw data lives
    data_search_path = base_path / "data" / "MAST_2025-05-02T1501" / "HST"

    # 2. Destination: set the destination destination to the CURRENT directory where the script runs
    current_dir = Path.cwd()
    dest_folder = current_dir / "hst_data_analysis" / f"rm{target_id}_x1d_files"

    # Create the folder if it doesn't exist
    dest_folder.mkdir(parents = True, exist_ok = True)

    # Use rglob to find all x1d files recursively
    x1d_files = list(data_search_path.rglob("*x1d.fits"))

    print(f"--- Processing Target: {full_targname} ---")
    count = 0

    for file_path in x1d_files:
        try:
            with fits.open(file_path) as hdul:
                header_targ = hdul[0].header.get('TARGNAME', 'UNKNOWN')

                if header_targ == full_targname:
                    shutil.copy(file_path, dest_folder / file_path.name)
                    count += 1
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print(f"Done! Copied {count} files to {dest_folder}\n")





if __name__ == "__main__":
    # Set up Command Line Arguments
    parser = argparse.ArgumentParser(description = "Sort HST x1d files by Target ID.")

    # Users can now pass target ID (e.g., --target 392 or --target 303)
    parser.add_argument(
        "--target",
        type = str,
        required = True,
        help = "The 3-digit target ID (e.g., 392 or 303)"
    )

    # Allow user to specify the base directory (i.e. the static location
    # on hard drive (or server) where the original, MAST downloads live.),
    # defaulting to current directory
    parser.add_argument(
        "--path",
        type = str,
        default = "./",
        help = "Base directory for the project (default: current directory)"
    )

    args = parser.parse_args()

    # Run the function with the provided arguments
    sort_hst_data(args.path, args.target)
