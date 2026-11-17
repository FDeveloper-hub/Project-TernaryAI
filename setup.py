"""Empaquetado de TernaryAI.

La parte nativa (libternaryai) se compila con el Makefile del proyecto, sin cambiar sus banderas, y se copia
dentro del paquete junto con las fuentes en C (ternaryai/_csrc), que necesitan tn.build_engine_lib y tn.export_fused.

Variables de entorno (todas opcionales):
  TERNARYAI_MAKE_ARGS   argumentos extra para make, p. ej. para compilar de forma cruzada:
                        CC=arm-linux-gnueabihf-gcc CFLAGS="-std=c11 -O3 -ffp-contract=off -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -Wall -Wextra -Iinclude"
  TERNARYAI_WHEEL_PLAT  etiqueta de plataforma de la wheel (por defecto la de esta maquina, p. ej. linux_x86_64)
"""
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:                                   # pragma: no cover
    _bdist_wheel = None

HERE = Path(__file__).resolve().parent
TREE = ("Makefile", "src", "include")


class build_py(_build_py):
    def run(self):
        super().run()
        out = Path(self.build_lib) / "ternaryai"
        out.mkdir(parents=True, exist_ok=True)
        csrc = out / "_csrc"
        if csrc.exists():
            shutil.rmtree(csrc)
        csrc.mkdir()
        for name in ("src", "include"):
            shutil.copytree(HERE / name, csrc / name, ignore=shutil.ignore_patterns("*.o", "*.so", "__pycache__"))
        docs = out / "docs"                            # los manuales viajan dentro de la wheel (ternaryai/docs/)
        docs.mkdir(exist_ok=True)
        for name in ("MANUAL.md", "MANUAL.en.md"):
            if (HERE / "docs" / name).exists():
                shutil.copy2(HERE / "docs" / name, docs / name)
        with tempfile.TemporaryDirectory() as tmp:     # se compila en una copia: un .so viejo de otra arquitectura no se reutiliza
            work = Path(tmp)
            shutil.copy2(HERE / "Makefile", work / "Makefile")
            for name in ("src", "include"):
                shutil.copytree(HERE / name, work / name, ignore=shutil.ignore_patterns("*.o", "*.so", "__pycache__"))
            cmd = ["make", "libternaryai.so"] + shlex.split(os.environ.get("TERNARYAI_MAKE_ARGS", ""))
            subprocess.run(cmd, cwd=work, check=True)
            shutil.copy2(work / "libternaryai.so", out / "libternaryai.so")


class sdist(_sdist):
    def make_distribution(self):
        # la sdist lleva el arbol de C completo (MANIFEST.in); aqui solo se comprueba que exista
        for name in TREE:
            if not (HERE / name).exists():
                raise SystemExit(f"falta {name}: la sdist se construye desde la raiz del proyecto")
        super().make_distribution()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):                          # fuerza la carpeta platlib: el paquete lleva una biblioteca nativa
        return True


cmdclass = {"build_py": build_py, "sdist": sdist}
if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False                  # lleva una biblioteca nativa

        def get_tag(self):
            plat = os.environ.get("TERNARYAI_WHEEL_PLAT") or super().get_tag()[2]
            return "py3", "none", plat                 # no es una extension de CPython: sirve para cualquier Python 3

    cmdclass["bdist_wheel"] = bdist_wheel

setup(cmdclass=cmdclass, distclass=BinaryDistribution)
