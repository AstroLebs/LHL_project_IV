FROM python:3.8
WORKDIR /opt/LHL_project_IV
COPY . .
RUN pip install -r requirements.txt
CMD ["python3","/OPT/LHL_project_IV/SRC/app.py"]