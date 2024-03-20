from pathlib import Path
import re
from typing import List, Tuple

from setuptools import setup, find_packages


NAME = "cmehr"
DESCRIPTION = "Pytorch code and models for MSPG"

URL = "https://github.com/KaedeGo/MMMSPG"
AUTHOR = "***"
REQUIRES_PYTHON = ">=3.10.0"
HERE = Path(__file__).parent


try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def get_requirements(path: str = HERE / "requirements.txt") -> Tuple[List[str], List[str]]:
    requirements = []
    extra_indices = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\r\n")
            if line.startswith("--extra-index-url "):
                extra_indices.append(line[18:])
                continue
            requirements.append(line)
    return requirements, extra_indices


def get_package_version() -> str:
    with open(HERE / "cmehr/__init__.py") as f:
        result = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if result:
            return result.group(1)
    raise RuntimeError("Can't get package version")


requirements, extra_indices = get_requirements()
version = get_package_version()

setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=requirements,
)
