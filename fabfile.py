# fabfile.py
from fabric import task, Connection
import shlex

TARGETS = {
    "local": {"dir": "."}, 
    "khipu": {
        "host": "khipu",   # edit
        "dir":  "/home/diego.canez/research/jepa-intuitive-physics",    # edit
        "modules": ["python3/3.11.11"],  # add more if needed
    },
}

def _remote_line(cfg, cmd: str) -> str:
    module_cmds = [f"module load {m}" for m in cfg.get("modules", [])]
    parts = [f"cd {shlex.quote(cfg['dir'])}"] + module_cmds + [cmd]
    return " && ".join(parts)

@task
def run(c, target, cmd):
    cfg = TARGETS.get(target)
    if not cfg:
        raise ValueError(f"unknown target: {target}")
    if target == "local":
        c.local(cmd)
        return
    with Connection(cfg["host"]) as r:
        r.run("bash -lc " + shlex.quote(_remote_line(cfg, cmd)))
