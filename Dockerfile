FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMDSTAN=/opt/cmdstan

WORKDIR /app

# System deps (Prophet/CmdStan need a C++ toolchain + curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Prophet ships an incomplete CmdStan tree in the wheel. Remove it and install
# a full CmdStan build that cmdstanpy / Prophet can actually use.
RUN python - <<'PY'
from pathlib import Path
import shutil
import cmdstanpy

stan_model = Path("/usr/local/lib/python3.11/site-packages/prophet/stan_model")
for d in stan_model.glob("cmdstan-*"):
    if d.is_dir():
        print(f"Removing incomplete bundled CmdStan: {d}")
        shutil.rmtree(d)

print("Installing CmdStan 2.33.1 ...")
ok = cmdstanpy.install_cmdstan(version="2.33.1", dir="/opt", overwrite=True, verbose=True, progress=False)
if not ok:
    raise SystemExit("cmdstanpy.install_cmdstan failed")

installed = Path(cmdstanpy.cmdstan_path())
print(f"CmdStan installed at {installed}")

# Place a complete copy where Prophet looks first (prophet/stan_model/cmdstan-X)
target = stan_model / installed.name
if target.exists():
    shutil.rmtree(target)
shutil.copytree(installed, target)
print(f"CmdStan mirrored for Prophet at {target}")
PY

# Project
COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
