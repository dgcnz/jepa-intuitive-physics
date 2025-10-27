TARGET = khipu

.PHONY: sync stderr stdout check-commit

sync:
	fab run --target=$(TARGET) --cmd "git pull && uv sync"

check-commit:
	@local_commit=$$(git rev-parse HEAD); \
	remote_commit=$$(fab run --target=$(TARGET) --cmd "git rev-parse HEAD" | tail -n 1); \
	if [ "$$local_commit" != "$$remote_commit" ]; then \
	  echo "Error: local HEAD ($$local_commit) != deployed HEAD ($$remote_commit) on $(TARGET)" 1>&2; \
	  exit 1; \
	fi

stderr:
	fab run --target=$(TARGET) --cmd "cat evaluation_code/logs/job_$(id)/$(id)_0_log.err"

stdout:
	fab run --target=$(TARGET) --cmd "cat evaluation_code/logs/job_$(id)/$(id)_0_log.out"

squeue:
	fab run --target=$(TARGET) --cmd "squeue"

wandb-sync:
	@test -n "$(ID)" || { echo "Usage: make $@ TARGET=<host> ID=<leaf-name>" >&2; exit 1; }
	@fab run --target=$(TARGET) --cmd ' \
	  dir=$$(find . -type d -name '"$(ID)"' -print -quit); \
	  [ -n "$$dir" ] || { echo "no match for '"$(ID)"'" >&2; exit 1; }; \
	  base=$${dir%/*/*}; \
	  set -- "$$base"/*/wandb/offline-run-*; \
	  [ -e "$$1" ] || { echo "no offline runs under $$base" >&2; exit 0; }; \
	  exec uv run wandb sync "$$@" \
	'
