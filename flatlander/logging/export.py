import wandb
api = wandb.Api()
run = api.run("/wullli/flatland/runs/3ie4c1l7")
print(run.history())