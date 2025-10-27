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
	fab run --target=$(TARGET) --cmd "echo logs/intuitive-physics-eval/multiruns/$(dir)/*/wandb/offline-run-* | xargs wandb sync"