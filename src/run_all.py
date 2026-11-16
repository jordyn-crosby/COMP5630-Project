# Title: run_all.py
# Description: One command for the whole protocol run, in the order the roadmap fixes:
#
#   0. unzip data/data.zip  (only when no raw CPS files are present; the zip holds nov19/21/23pub.csv)
#   1. dataPrep.py           build data/predictor_data.csv and data/cohort_summary.csv from the raw files
#   2. unit tests            test_dataPrep + test_main + test_plot_results (skip with --skip-tests)
#   3. main.py               Experiment B (temporal) then Experiment A (pooled)
#   4. h3_feature_reduction  full vs reduced inputs, prespecified tolerances
#   5. main.py --experiment C  sensitivity run, unweighted and tuned on accuracy (skip with --skip-sensitivity)
#   6. plot_results.py       results/: model_performance.csv + the two figures
#
# Works in both layouts: scripts in src/ with tests/ beside it (the GitHub repo), or everything in one folder
# (the Revised_code working copy). All paths are relative to the project root, which is the parent of src/
# when this file lives in src/, otherwise this file's own folder.
#
# Run:  python src/run_all.py                        (raw files in data/, or data/data.zip)
#       python src/run_all.py --raw-dir /folder/with/CPS2019.csv
#       python src/run_all.py --quick                (tiny grids, smoke test only, results not for the paper)
#       python src/run_all.py --dry-run              (print the commands, run nothing)

import argparse
import glob
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
SRC_LAYOUT = os.path.basename(HERE) == "src"


def script(name):
    return os.path.join(HERE, name)


def raw_files_present(raw_dir):
    return bool(glob.glob(os.path.join(raw_dir, "CPS20*.csv")) or glob.glob(os.path.join(raw_dir, "nov*pub.csv")))


def unzip_raw_data(raw_dir, dry_run):
    """Extract data/data.zip (nov19pub.csv, nov21pub.csv, nov23pub.csv) into raw_dir, skipping macOS metadata."""
    zip_path = os.path.join(ROOT, "data", "data.zip")
    if not os.path.exists(zip_path):
        return False
    print(f"[0] Raw CPS files not found in {raw_dir}; extracting {zip_path} (about 620 MB on disk)")
    if dry_run:
        return True
    os.makedirs(raw_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = os.path.basename(member.filename)
            is_raw = name.lower().endswith(".csv") and (name.lower().startswith("nov") or name.upper().startswith("CPS20"))
            if member.is_dir() or "__MACOSX" in member.filename or not is_raw:
                continue  # only the raw survey files; the zip may also hold stale derived tables
            target = os.path.join(raw_dir, name)
            with z.open(member) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            print(f"      extracted {name}")
    return True


def test_command(py):
    if SRC_LAYOUT and os.path.isdir(os.path.join(ROOT, "tests")):
        return [py, "-m", "unittest", "tests.test_dataPrep", "tests.test_main", "tests.test_plot_results"]
    return [py, "-m", "unittest", "test_dataPrep", "test_main", "test_plot_results"]


def steps(raw_dir, quick, skip_tests, skip_sensitivity):
    py = sys.executable
    quick_flags = ["--quick", "--cv", "3"] if quick else []
    plan = [("Build the modeling table", [py, script("dataPrep.py"), "--raw-dir", raw_dir])]
    if not skip_tests:
        plan.append(("Unit tests", test_command(py)))
    plan.append(("Experiments B (temporal) and A (pooled)", [py, script("main.py")] + quick_flags))
    plan.append(("H3 feature reduction", [py, script("h3_feature_reduction.py")] + quick_flags))
    if not skip_sensitivity:
        plan.append(("Sensitivity run C (unweighted, accuracy-tuned)",
                     [py, script("main.py"), "--experiment", "C"] + quick_flags))
    plan.append(("Figures and combined table", [py, script("plot_results.py")]))
    return plan


def main():
    parser = argparse.ArgumentParser(description="Run the full digital exclusion pipeline.")
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data"),
                        help="folder holding CPS2019.csv, CPS2021.csv, CPS2023.csv (or nov19pub.csv ...); default data/")
    parser.add_argument("--quick", action="store_true", help="tiny grids for a smoke test (results not for the paper)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    os.chdir(ROOT)  # every script reads data/ and writes outputs/ and results/ relative to the project root
    print(f"Project root: {ROOT}\nScripts: {HERE}\nRaw files: {raw_dir}\n")

    if not raw_files_present(raw_dir) and not unzip_raw_data(raw_dir, args.dry_run):
        print(f"No raw CPS files in {raw_dir} and no data/data.zip to extract. "
              f"Place CPS2019.csv, CPS2021.csv and CPS2023.csv there or pass --raw-dir.")
        sys.exit(1)

    plan = steps(raw_dir, args.quick, args.skip_tests, args.skip_sensitivity)
    t_all = time.time()
    for i, (label, cmd) in enumerate(plan, start=1):
        shown = " ".join("python" if c == sys.executable else os.path.relpath(c, ROOT) if os.path.isabs(c) else c
                         for c in cmd)
        print(f"[{i}/{len(plan)}] {label}\n      $ {shown}")
        if args.dry_run:
            continue
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStopped: step {i} ({label}) exited with code {result.returncode}.")
            sys.exit(result.returncode)
        print(f"      done in {time.time() - t0:.0f}s\n")
    if not args.dry_run:
        print(f"All steps finished in {(time.time() - t_all) / 60:.1f} minutes. "
              f"Figures and table: {os.path.join(ROOT, 'results')}")


if __name__ == "__main__":
    main()
