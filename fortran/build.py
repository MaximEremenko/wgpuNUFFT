#!/usr/bin/env python3
"""Builds the Fortran module and its test against the C library, and runs it.

Usage: python fortran/build.py [--compiler gfortran|ifx|ifort] [--run cpu|gpu]

Builds capi/ with cargo in release mode, compiles fortran/wgpu_nufft.f90 and
fortran/tests/test_wgpu_nufft.f90 into fortran/build/, links them against
the shared C library and, with --run, runs the test on that backend. The
compiler defaults to $FC, then the first of gfortran, ifx and ifort found.
Intel compilers on Windows need the oneAPI environment (setvars.bat).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORTRAN = ROOT / "fortran"
BUILD = FORTRAN / "build"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def find_compiler(requested: str | None) -> str:
    for candidate in [requested, os.environ.get("FC"), "gfortran", "ifx", "ifort"]:
        if candidate and shutil.which(candidate):
            return candidate
    sys.exit("no Fortran compiler found; pass --compiler or set FC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compiler")
    parser.add_argument("--run", choices=["cpu", "gpu"])
    args = parser.parse_args()

    compiler = find_compiler(args.compiler)
    intel = Path(compiler).stem.lower() in ("ifx", "ifort")
    windows = platform.system() == "Windows"
    run(["cargo", "build", "--release", "--locked", "-p", "wgpu-nufft-c"], cwd=ROOT)
    library_dir = ROOT / "target" / "release"
    BUILD.mkdir(exist_ok=True)

    module = FORTRAN / "wgpu_nufft.f90"
    test = FORTRAN / "tests" / "test_wgpu_nufft.f90"
    executable = BUILD / ("test_wgpu_nufft.exe" if windows else "test_wgpu_nufft")
    if intel and windows:
        run([compiler, "/nologo", "/c", f"/module:{BUILD}", f"/object:{BUILD}\\", str(module)])
        run([compiler, "/nologo", f"/module:{BUILD}", f"/object:{BUILD}\\", f"/exe:{executable}", str(test),
             str(BUILD / "wgpu_nufft.obj"), str(library_dir / "wgpu_nufft_c.dll.lib")])
    else:
        flags = ["-O2", "-std=f2018", "-Wall"] if not intel else ["-O2"]
        module_flag = ["-J", str(BUILD)] if not intel else ["-module", str(BUILD)]
        run([compiler, *flags, *module_flag, "-c", str(module), "-o", str(BUILD / "wgpu_nufft.o")])
        link = [f"-L{library_dir}", "-lwgpu_nufft_c"]
        if not windows:
            link.append(f"-Wl,-rpath,{library_dir}")
        run([compiler, *flags, *module_flag, str(test), str(BUILD / "wgpu_nufft.o"), *link,
             "-o", str(executable)])
    if windows:
        # The executable finds the DLL next to itself.
        shutil.copy2(library_dir / "wgpu_nufft_c.dll", BUILD / "wgpu_nufft_c.dll")
    if args.run:
        run([str(executable), args.run])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
