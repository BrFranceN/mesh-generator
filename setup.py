from setuptools import setup, find_packages

setup(
    name="your_package_name",  # Replace with your desired, descriptive name
    version="0.1.0",  # Start with version 0.1.0 for initial release
    packages=find_packages(),  # Automatically finds packages in a standard structure
    description="A brief description of your package's functionality",
    url="https://github.com/<your_username>/<your_repo_name>",  # Link to your GitHub repository
    author="Your Name or Organization",
    author_email="your_email@example.com",
    license="Specify your chosen license (e.g., MIT, Apache)",
    install_requires=[  # List your project's external dependencies here
        "dependency1",
        "dependency2",
        ...
    ],
    classifiers=[  # Optional: Classify your package for better discoverability
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
