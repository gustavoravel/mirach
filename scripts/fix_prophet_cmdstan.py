from pathlib import Path
import shutil
import cmdstanpy

stan_model = Path("/usr/local/lib/python3.11/site-packages/prophet/stan_model")
for d in stan_model.glob("cmdstan-*"):
    if d.is_dir():
        print(f"Removing incomplete bundled CmdStan: {d}")
        shutil.rmtree(d)

print("Installing CmdStan 2.33.1 into /opt ...")
ok = cmdstanpy.install_cmdstan(
    version="2.33.1",
    dir="/opt",
    overwrite=True,
    verbose=True,
    progress=False,
)
print("install ok:", ok)
installed = Path(cmdstanpy.cmdstan_path())
print("path:", installed)

target = stan_model / installed.name
if target.exists():
    shutil.rmtree(target)
shutil.copytree(installed, target)
print("mirrored to", target)
print("makefile exists:", (target / "makefile").exists())

from prophet import Prophet
m = Prophet()
print("stan_backend:", type(m.stan_backend).__name__)
