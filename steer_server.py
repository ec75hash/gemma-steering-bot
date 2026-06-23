"""FastAPI server exposing the SteerEngine over REST + SSE, and serving the web UI.

Run:
  python3 steer_server.py                # serves API + built web/dist on :8099
  KITCHEN_PORT=8099 python3 steer_server.py

Dev: run this for the API, and `npm run dev` in web/ (Vite proxies /api here).
"""
import json
import os
import threading
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

from steer_engine import HUM_PROMPT, SteerEngine, SteerError

ROOT = Path(__file__).parent
WEB_DIST = ROOT / "web" / "dist"
HOST = os.environ.get("KITCHEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("KITCHEN_PORT", "8099"))
DEFAULT_MODEL = os.environ.get("KITCHEN_MODEL", "it")
DEFAULT_PROMPT_MODE = os.environ.get("KITCHEN_PROMPT_MODE", "auto")

app = FastAPI(title="Gemma Test Kitchen")

# Single local session.
_engine = SteerEngine(DEFAULT_MODEL, DEFAULT_PROMPT_MODE)
_engine_lock = threading.Lock()


def engine() -> SteerEngine:
    return _engine


def guard(fn):
    """Run an engine call, mapping SteerError -> HTTP 400."""
    try:
        return fn()
    except SteerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --------------------------------------------------------------------------- #
# status / model lifecycle
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def status():
    return engine().status()


@app.get("/api/help")
def help_text():
    return {"humPrompt": HUM_PROMPT}


@app.post("/api/load")
def load(payload: dict = Body(default={})):
    global _engine
    model = payload.get("model", _engine.model_key)
    prompt_mode = payload.get("promptMode", "auto")
    with _engine_lock:
        eng = _engine
        if eng.loading:
            return JSONResponse(eng.status(), status_code=202)
        # switch model/prompt-mode by building a fresh engine
        if model != eng.model_key or (prompt_mode != "auto" and prompt_mode != eng.prompt_mode):
            eng = SteerEngine(model, prompt_mode)
            _engine = eng
        if eng.loaded:
            return eng.status()
        eng.loading = True
        eng.load_error = None

        def _load_async():
            try:
                eng.load()
            except Exception as exc:  # noqa: BLE001
                eng.load_error = str(exc)
            finally:
                eng.loading = False

        threading.Thread(target=_load_async, daemon=True).start()
        return JSONResponse(eng.status(), status_code=202)


# --------------------------------------------------------------------------- #
# catalogs
# --------------------------------------------------------------------------- #
@app.get("/api/catalog/features")
def cat_features(q: str = Query("")):
    return engine().list_features(q)


@app.get("/api/catalog/bundles")
def cat_bundles(q: str = Query("")):
    return engine().list_bundles(q)


@app.get("/api/catalog/configs")
def cat_configs(q: str = Query("")):
    return engine().list_configs(q)


@app.get("/api/catalog/aliases")
def cat_aliases(q: str = Query("")):
    return engine().list_aliases(q)


@app.get("/api/catalog/presets")
def cat_presets():
    return engine().presets()


@app.get("/api/catalog/hum-prompt")
def cat_hum():
    return {"humPrompt": HUM_PROMPT}


# --------------------------------------------------------------------------- #
# steering
# --------------------------------------------------------------------------- #
@app.get("/api/steering")
def get_steering():
    return engine().steering_snapshot()


@app.delete("/api/steering")
def clear_steering():
    return engine().clear_steer()


@app.delete("/api/steering/{index}")
def remove_steering(index: int):
    return guard(lambda: engine().remove_steer(index))


@app.post("/api/steering/set")
def steering_set(payload: dict = Body(...)):
    return guard(lambda: engine().add_named(payload["name"], payload.get("level"), replace=True))


@app.post("/api/steering/add")
def steering_add(payload: dict = Body(...)):
    return guard(lambda: engine().add_named(payload["name"], payload.get("level"), replace=False))


@app.post("/api/steering/config")
def steering_config(payload: dict = Body(...)):
    return guard(lambda: engine().add_config(payload["name"], payload.get("level"), replace=True))


@app.post("/api/steering/direct")
def steering_direct(payload: dict = Body(...)):
    return guard(lambda: engine().apply_direct(payload["name"], payload.get("level")))


@app.post("/api/steering/inject")
def steering_inject(payload: dict = Body(...)):
    replace = bool(payload.get("replace", False))
    if "name" in payload and payload["name"]:
        return guard(lambda: engine().add_inject_named(payload["name"], payload.get("strength"), replace))
    return guard(lambda: engine().add_inject_raw(payload["layer"], payload["feature"],
                                                 payload["strength"], replace))


@app.post("/api/steering/dim")
def steering_dim(payload: dict = Body(...)):
    if payload.get("preset"):
        return guard(lambda: engine().add_dim_preset(payload["preset"], payload["scale"]))
    return guard(lambda: engine().add_dim_raw(payload["layer"], payload["features"], payload["scale"]))


@app.post("/api/steering/no-hedge")
def steering_no_hedge(payload: dict = Body(default={})):
    return guard(lambda: engine().no_hedge(float(payload.get("scale", 0.0))))


@app.post("/api/steering/injectvec")
async def steering_injectvec(
    layer: int = Form(...),
    strength: float = Form(...),
    path: str = Form(None),
    file: UploadFile = File(None),
):
    if file is not None:
        raw = await file.read()
        return guard(lambda: engine().add_injectvec_bytes(layer, strength, raw, file.filename))
    if path:
        return guard(lambda: engine().add_injectvec_path(layer, strength, path))
    raise HTTPException(status_code=400, detail="provide a file upload or a server-side path")


# --------------------------------------------------------------------------- #
# generation config
# --------------------------------------------------------------------------- #
@app.put("/api/config")
def set_config(payload: dict = Body(...)):
    eng = engine()

    def apply():
        if "phase" in payload:
            eng.set_phase(payload["phase"])
        if "seed" in payload:
            eng.set_seed(payload["seed"])
        if "temp" in payload:
            eng.set_temp(payload["temp"])
        if "tokens" in payload or "soft" in payload:
            eng.set_tokens(payload.get("tokens", eng.tokens), payload.get("soft", eng.soft))
        if "contextTurns" in payload:
            eng.set_context(payload["contextTurns"])
        return eng.status()

    return guard(apply)


# --------------------------------------------------------------------------- #
# chat (SSE) + history
# --------------------------------------------------------------------------- #
@app.post("/api/chat")
def chat(payload: dict = Body(default={})):
    eng = engine()
    content = payload.get("content")
    use_hum = bool(payload.get("useHum", False))
    gen = guard(lambda: eng.chat_stream(content=content, use_hum=use_hum))

    def sse():
        try:
            for piece in gen:
                yield f"data: {json.dumps({'content': piece})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except SteerError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': repr(exc)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/chat/stop")
def chat_stop():
    engine().stop()
    return {"ok": True}


@app.get("/api/history")
def get_history():
    return engine().history


@app.delete("/api/history")
def reset_history():
    return engine().reset_history()


# --------------------------------------------------------------------------- #
# logits / analysis
# --------------------------------------------------------------------------- #
@app.post("/api/logits")
def logits(payload: dict = Body(default={})):
    return guard(lambda: engine().logits(
        top_n=int(payload.get("topN", 20)),
        source=payload.get("source", "current"),
        save=bool(payload.get("save", False)),
    ))


@app.get("/api/logits/snapshots")
def logits_snapshots():
    return engine().list_logits_snapshots()


@app.get("/api/logits/snapshots/{name}")
def logits_snapshot(name: str):
    return guard(lambda: engine().read_logits_snapshot(name))


# --------------------------------------------------------------------------- #
# static frontend (built web/dist) — must be registered last
# --------------------------------------------------------------------------- #
@app.get("/{full_path:path}")
def serve_spa(full_path: str, request: Request):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    if not WEB_DIST.exists():
        return JSONResponse(
            {"error": "frontend not built", "hint": "cd web && npm run build"}, status_code=503)
    target = (WEB_DIST / full_path).resolve()
    if WEB_DIST.resolve() in target.parents and target.is_file():
        return FileResponse(target)
    index = WEB_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="not found")


if __name__ == "__main__":
    print(f"Gemma Test Kitchen API: http://{HOST}:{PORT}")
    print(f"  model={_engine.model_key} prompt={_engine.prompt_mode} "
          f"sae_layers={sorted(_engine.sae_layers)}")
    print(f"  frontend: {'web/dist (built)' if WEB_DIST.exists() else 'NOT built — run: cd web && npm run build'}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
