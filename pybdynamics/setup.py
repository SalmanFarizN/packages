import setuptools

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="pybdynamics",
    version="0.0.1",
    description="Functions for BD simulations and data-analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    author="Salman Fariz Navas",
    author_email="salmanfarizn@gmail.com",
    keywords=["molecular dynamics", "data analysis", "brownian dynamics"],
    install_requires=["numpy", "numba", "math", "cmath", "os"],
)


def main():
    setuptools.setup(**setup_args)
