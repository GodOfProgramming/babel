import yaml
import json
from fastapi import Request, HTTPException
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ModelParser:
    def __init__(self, model: Type[T]):
        self._model = model

    async def __call__(self, request: Request) -> T:
        content_type = request.headers.get("Content-Type")
        body = await request.body()

        if not body:
            raise HTTPException(status_code=400, detail="Empty request")

        data: dict[str, Any]
        match content_type:
            case "application/json":
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid json: {e}")
            case "application/x-yaml":
                try:
                    data = yaml.full_load(body)
                except yaml.YAMLError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid yaml: {e}")
            case _:
                raise HTTPException(status_code=400, detail="Unsupported content type")

        try:
            return self._model(**data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())
