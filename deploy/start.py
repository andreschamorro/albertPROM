import subprocess
subprocess.run("uvicorn modules.app:app --host 0.0.0.0 --port 7860", shell=True)
