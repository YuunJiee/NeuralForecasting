import wandb

# Entity and Project from your previous logs
ENTITY = "yuunjiee313-national-central-university"
PROJECT = "NSF neural forecasting"

api = wandb.Api()

print(f"Fetching runs from {ENTITY}/{PROJECT}...")
runs = api.runs(f"{ENTITY}/{PROJECT}")

if len(runs) == 0:
    print("No runs found.")
else:
    print(f"Found {len(runs)} runs. Deleting...")
    for run in runs:
        print(f"Deleting run: {run.name} ({run.id})")
        run.delete()

print("✅ All runs deleted successfully.")
