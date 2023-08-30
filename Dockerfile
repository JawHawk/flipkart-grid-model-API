# start by pulling the python image
FROM python:3.11

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

RUN python -m pip install --upgrade pip

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt
RUN pip3 install promptify --no-deps

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]