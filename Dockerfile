FROM anibali/pytorch:1.7.0-cuda11.0

WORKDIR /app

COPY *.py /app/

RUN  pip install scipy -i https://mirrors.nju.edu.cn/pypi/web/simple/
