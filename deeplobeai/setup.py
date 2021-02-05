import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplobeai", # Replace with your own username
    version="0.0.1",
    author="purnasai@soulpage",
    author_email="purnsai.gudikandula@soulpageit.com",
    description="A package for all your usecases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://soulpageit.com/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
