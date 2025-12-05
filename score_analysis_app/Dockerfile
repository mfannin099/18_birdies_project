FROM python:3.12

WORKDIR /score_analysis_app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]