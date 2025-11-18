# start_server.ps1 - activate venv and start uvicorn
cd $PSScriptRoot\..
Write-Host "Activating venv..."
.\venv\Scripts\Activate.ps1

# Optionally set RAG_LOCAL_MODEL (uncomment to enforce)
# $env:RAG_LOCAL_MODEL = "distilgpt2"

# If you want OpenAI, export key here (DO NOT commit):
# $env:OPENAI_API_KEY = "sk-REPLACE_ME"

Write-Host "Starting uvicorn at http://127.0.0.1:8000 ..."
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
