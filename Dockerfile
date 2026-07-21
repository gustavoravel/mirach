# Thin app image for Railway / production.
# CmdStan lives in the pre-built base (GHCR) so this layer is minutes, not hours.
#
# First time: run the "Build CmdStan base image" GitHub Action, then redeploy on Railway.
ARG BASE_IMAGE=ghcr.io/gustavoravel/mirach-base:cmdstan-2.33.1
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMDSTAN=/opt/cmdstan

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
