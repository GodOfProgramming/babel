import json
import yaml
from abc import abstractmethod
from fastapi import Request, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Any, Generic

T = TypeVar("T", bound=BaseModel)


class Converter:
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass

    @abstractmethod
    def headers(self) -> dict[str, str]:
        pass


class JsonConverter(Converter):
    def __call__(self, data: Any) -> Any:
        return json.dumps(data)

    def headers(self):
        return {"Content-Type": "application/json"}


class YamlConverter(Converter):
    def __call__(self, data: Any) -> Any:
        return yaml.dump(data, width=28)

    def headers(self):
        return {"Content-Type": "application/x-yaml"}


class Content(Generic[T]):
    def __init__(self, model: Type[T], converter: Converter):
        self.model = model
        self.converter = converter

    def validate(self) -> T:
        self.model


class ModelParser:
    def __init__(self, model: Type[T]):
        self._model = model

    async def __call__(self, request: Request) -> T:
        content_type = request.headers.get("Content-Type")
        body = await request.body()

        if not body:
            raise HTTPException(status_code=400, detail="Empty request")

        data: dict[str, Any]
        converter: Converter
        match content_type:
            case "application/json":
                try:
                    data = json.loads(body)
                    converter = JsonConverter()
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid json: {e}")
            case "application/x-yaml":
                try:
                    data = yaml.full_load(body)
                    converter = YamlConverter()
                except yaml.YAMLError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid yaml: {e}")
            case _:
                raise HTTPException(status_code=400, detail="Unsupported content type")

        try:
            return Content(self._model(**data), converter)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())
