from setuptools import setup, find_packages

setup(
    name="faq-extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12, <3.13",
    entry_points={
        "console_scripts": [
            "faq-extractor=faq_extractor.main:run_pipeline",
        ],
    },
)
