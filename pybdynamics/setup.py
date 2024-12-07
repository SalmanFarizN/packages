import setuptools

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="pybdynamics",
    version="0.0.3",
    description="Python package for simulation and data analysis of interacting colloidal particle systems (Brownian Dynamics)",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    author="Salman Fariz Navas",
    author_email="salmanfarizn@gmail.com",
    keywords=["molecular dynamics", "data analysis", "brownian dynamics"],
    install_requires=["numpy", "numba"],
    url="https://github.com/SalmanFarizN/pybdynamics/tree/master",
)


if __name__ == "__main__":
    setuptools.setup(**setup_args)
