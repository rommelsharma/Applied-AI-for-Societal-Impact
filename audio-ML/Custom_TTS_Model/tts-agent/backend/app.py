from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from backend.api.routes import router
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_HERE = Path(__file__).resolve().parent.parent
_FRONTEND = _HERE / "frontend"

app = FastAPI(
    title="TTS Agent",
    description="Local TTS service using Kokoro-82M (built-in voices) and F5-TTS (voice cloning)",
    version="1.0.0",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(_FRONTEND / "static")), name="static")

templates = Jinja2Templates(directory=str(_FRONTEND / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("startup")
def on_startup():
    logger.info("TTS Agent started — visit http://localhost:8000")
