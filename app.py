import os
import uvicorn
import sys

# Import the FastAPI app from the backend directory
from backend.main import app

if __name__ == "__main__":
    print("Starting modern FastAPI Clinic AI application...")
    port = int(os.environ.get("PORT", 8000))
    
    # If this was accidentally executed by Streamlit, it will crash due to port conflicts.
    # The user must ensure the start command is `python app.py` or `cd backend && python main.py`
    uvicorn.run(app, host="0.0.0.0", port=port)
