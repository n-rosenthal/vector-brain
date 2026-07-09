"""
App FastAPI do vector-brain.

Uso:
    uvicorn src.api.App:app --reload --host 127.0.0.1 --port 8000

Docs interativas em /docs (Swagger) e /redoc.

Autenticação: desligada por padrão (uso local). Pra exigir uma API key em
toda requisição (recomendado antes de expor pra rede), defina a variável
de ambiente VECTOR_BRAIN_API_KEY e mande o header `X-API-Key` em todo
request.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import config
from .Routes import router

_PUBLIC_PATHS = {"/docs", "/redoc", "/openapi.json"}


def create_app() -> FastAPI:
    app = FastAPI(
        title="vector-brain API",
        description=(
            "API somente-leitura sobre a base org-roam indexada: busca "
            "semântica, navegação por nodes/tags e estatísticas do corpus."
        ),
        version="0.1.0",
    )

    # CORS liberado por padrão pra facilitar uso local (ex: um frontend
    # rodando em outra porta). Restrinja allow_origins se for expor a API
    # além de localhost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        if config.API_KEY and request.url.path not in _PUBLIC_PATHS:
            if request.headers.get("x-api-key") != config.API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key ausente ou inválida (header X-API-Key)"},
                )
        return await call_next(request)

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
