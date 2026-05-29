from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse

from backend.api.routes import router
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_HERE = Path(__file__).resolve().parent.parent
_FRONTEND = _HERE / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TTS Agent v3.0.0 started — visit http://localhost:8000")
    yield


app = FastAPI(
    title="TTS Agent",
    description="Local TTS service using Kokoro-82M (built-in voices) and XTTS v2 (voice cloning)",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(_FRONTEND / "static")), name="static")

templates = Jinja2Templates(directory=str(_FRONTEND / "templates"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = _FRONTEND / "static" / "favicon.ico"
    return FileResponse(str(ico), media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
