# Use Python 3.10.2
FROM python:3.10.2

# Set the working directory
WORKDIR /app

# Copy the poetry.lock and pyproject.toml files
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root

# Install gunicorn
RUN pip install gunicorn

# Copy the rest of the application code
COPY . /app

# Expose port 5000
EXPOSE 5000

# Use Gunicorn as the WSGI server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "src.serve.serveApi:app"]
