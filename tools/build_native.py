"""Build the native batch tile codec (phantom/native/_tilecodec*.pyd) on Windows with Visual Studio.

    python -m tools.build_native

1. Downloads libjpeg-turbo 3.2.1 (source) and NASM 3.02 from their official sites into build/third_party/dl and
   checks them against pinned SHA-256 hashes (skipped when already there).
2. Builds libjpeg-turbo as a static library with SIMD (MSVC + NASM).
3. Compiles phantom/native/_tilecodec.cpp against it into an extension for this Python.

Needs Visual Studio (Community is fine) with "Desktop development with C++". Without the extension the demo
falls back to OpenCV, tile by tile (slower above 360p, same pictures).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "build" / "third_party"
DOWNLOADS = {
    "libjpeg-turbo-3.2.1.tar.gz": (
        "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.1/libjpeg-turbo-3.2.1.tar.gz",
        "c27ad72a68da151177929aa969ae773a2da5ac2af9ce946079314fdac8cece7f"),
    "nasm-3.02-win64.zip": (
        "https://www.nasm.us/pub/nasm/releasebuilds/3.02/win64/nasm-3.02-win64.zip",
        "161d0bfaff53c2f9e9f3e69fd0672323ebabafd1268976a5cec11be92a19aee7"),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch():
    dl = TP / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in DOWNLOADS.items():
        p = dl / name
        if not p.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, p)
        if sha256(p) != digest:
            p.unlink()
            sys.exit(f"{name}: checksum mismatch, deleted; run again")
    src = TP / "src"
    src.mkdir(exist_ok=True)
    if not (src / "libjpeg-turbo-3.2.1").exists():
        with tarfile.open(dl / "libjpeg-turbo-3.2.1.tar.gz") as t:
            t.extractall(src, filter="data")
    if not (src / "nasm-3.02").exists():
        zipfile.ZipFile(dl / "nasm-3.02-win64.zip").extractall(src)


def vcvars() -> Path:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        r = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                           capture_output=True, text=True)
        p = Path(r.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if r.stdout.strip() and p.exists():
            return p
    sys.exit("Visual Studio with C++ build tools not found (install 'Desktop development with C++')")


def run_bat(lines: list[str], name: str):
    bat = TP / name
    bat.write_text("@echo off\r\n" + "\r\n".join(lines) + "\r\n", encoding="utf-8")
    r = subprocess.run(["cmd", "/c", str(bat)], cwd=TP)
    if r.returncode:
        sys.exit(f"{name} failed (exit {r.returncode})")


def main():
    if sys.platform != "win32":
        sys.exit("this script builds with MSVC on Windows; on Linux compile phantom/native/_tilecodec.cpp with "
                 "g++ -O2 -shared -fPIC against libturbojpeg (see the comment at its top)")
    fetch()
    vc = vcvars()
    cmake = vc.parents[3] / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    lj = TP / "libjpeg-turbo"
    if not (lj / "lib" / "turbojpeg-static.lib").exists():
        src, nasm = TP / "src" / "libjpeg-turbo-3.2.1", TP / "src" / "nasm-3.02" / "nasm.exe"
        run_bat([f'call "{vc}" >nul || exit /b 1',
                 f'"{cmake}" -S "{src}" -B "{TP / "build-libjpeg"}" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release '
                 f'-DENABLE_SHARED=0 -DENABLE_STATIC=1 -DWITH_TURBOJPEG=1 -DREQUIRE_SIMD=1 -DWITH_CRT_DLL=1 '
                 f'-DCMAKE_ASM_NASM_COMPILER="{nasm}" -DCMAKE_INSTALL_PREFIX="{lj}" || exit /b 1',
                 f'"{cmake}" --build "{TP / "build-libjpeg"}" --target install || exit /b 1'], "build_libjpeg.bat")
    out = ROOT / "phantom" / "native" / ("_tilecodec" + sysconfig.get_config_var("EXT_SUFFIX"))
    obj = TP / "obj"
    obj.mkdir(exist_ok=True)
    py_inc, py_libs = sysconfig.get_paths()["include"], Path(sys.base_prefix) / "libs"
    run_bat([f'call "{vc}" >nul || exit /b 1',
             f'cl /nologo /O2 /EHsc /MD /std:c++17 /LD /W3 /I"{py_inc}" /I"{lj / "include"}" '
             f'/Fo"{obj}\\\\" "{ROOT / "phantom" / "native" / "_tilecodec.cpp"}" '
             f'/link /LIBPATH:"{py_libs}" "{lj / "lib" / "turbojpeg-static.lib"}" /OUT:"{out}" '
             f'/IMPLIB:"{obj / "_tilecodec.lib"}" || exit /b 1'], "build_tilecodec.bat")
    print("built", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
