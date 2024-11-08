FROM python:3.13-slim

# Set meta information
LABEL org.opencontainers.image.source="https://github.com/jansim/multiversum"
LABEL org.opencontainers.image.description="Container image to test parallel execution in a container environment"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Set up venv
RUN python -m venv venv

# Install package
RUN pip install -e '.[test]'

# Run multiverse_analysis.py when the container launches, passing the seed as an argument
CMD ["python", "-m", "pytest"]
