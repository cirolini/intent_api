# Etapa base com Python 3.10
FROM python:3.10-slim

# Define diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do projeto para dentro do container
COPY . /app

# Instala dependências do sistema (inclui compiladores e bibliotecas usadas por faiss, datasets etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Define variáveis de ambiente necessárias para o Flask
ENV FLASK_APP=app
ENV PYTHONUNBUFFERED=1

# Executa o build do índice FAISS durante o build da imagem
RUN flask build-index

# Expõe a porta do Flask
EXPOSE 5000

# Comando padrão para iniciar a aplicação
CMD ["python", "run.py"]
