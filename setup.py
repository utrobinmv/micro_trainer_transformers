import os
import setuptools

description = "Micro version train modules Transformers trainer...."
long_description = description
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()


setuptools.setup(
    name="micro_trainer_transformers",
    version="0.0.6",
    author="Utrobin Mikhail",
    author_email="utrobinmv@yandex.ru",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utrobinmv/micro_trainer_transformers",
    packages=setuptools.find_packages(),
    license="Apache",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pytorch-lightning<=1.9.5','torch','datasets'],
    entry_points={
        "console_scripts": []
    }
)
