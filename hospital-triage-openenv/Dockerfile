FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    pyyaml \
    flask \
    flask-cors

EXPOSE 7860

CMD ["python", "baseline_inference.py", "--agent", "greedy", "--seeds", "3"]