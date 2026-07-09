import os
import string
import textwrap
import uvicorn
from fastapi import FastAPI, Depends, Response
from languages import LANGUAGES
from pydantic import BaseModel
from translators import TRANSLATORS, translate
from typing import Optional, TypeVar, Callable
from util import ModelParser, Content, log

app = FastAPI()

LANGUAGE_KEYS = list(LANGUAGES.keys())


def run_server(port: int):

    uvicorn.run(app, host="localhost", port=port, reload=False)


class TranslationRequest(BaseModel):
    text: str
    n: int


@app.post("/translate")
def app_translate(request: TranslationRequest):
    output = translate(request.text, TRANSLATORS, LANGUAGE_KEYS, iterations=request.n)
    return output


type Batch = dict[str, "str | Batch"]


class BatchTranslationRequest(BaseModel):
    n: int
    batch: Batch
    options: Optional[dict[str, int | float | str]] = None


@app.post("/translate/batch")
def app_translate_batch(
    request: Content[BatchTranslationRequest] = Depends(
        ModelParser(BatchTranslationRequest)
    ),
):
    wrap = None
    temperature = None
    top_k = None
    confidence_threshold = None
    if request.model.options is not None:
        opts = request.model.options
        wrap = get_opt(opts, "wrap", int)
        temperature = get_opt(opts, "temp", float)
        top_k = get_opt(opts, "top_k", int)
        confidence_threshold = get_opt(opts, "confidence", float)

    recursive_translate(
        request.model.batch,
        iterations=request.model.n,
        wrap=wrap,
        temp=temperature,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )

    return Response(
        content=request.converter(request.model.batch),
        headers=request.converter.headers(),
    )


def recursive_translate(
    batch: Batch,
    iterations: int,
    wrap=None,
    temp: Optional[float] = None,
    top_k: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
):

    for k, b in batch.items():
        if "_IGNORE_" not in k:
            if isinstance(b, dict):
                recursive_translate(
                    b,
                    iterations,
                    wrap,
                    temp=temp,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                )
            else:
                if b == "" or is_punctuation(
                    b
                ):  # skip empty strings and lines that are just punctuation
                    continue
                output = translate(
                    b,
                    TRANSLATORS,
                    LANGUAGE_KEYS,
                    iterations=iterations,
                    temp=temp,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                )

                if wrap is not None:
                    output = os.linesep.join(textwrap.wrap(output, width=wrap))

                batch[k] = output


def is_punctuation(s: str) -> bool:
    return bool(s) and all(c in string.punctuation for c in s)


T = TypeVar("T")


def get_opt(
    options: dict[str, int | float | str], key: str, expected: type[T]
) -> Optional[T]:
    if key in options:
        val = options[key]
        if isinstance(val, expected):
            return val
        else:
            log(f"ERROR: type mismatch with option {key}")
            return None
    else:
        return None
