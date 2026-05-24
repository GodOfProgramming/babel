import os
import textwrap
import uvicorn
from fastapi import FastAPI, Depends, Response
from languages import LANGUAGES
from pydantic import BaseModel
from translators import TRANSLATORS, translate
from typing import Optional
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
    options: Optional[dict[str, int | str]] = None


@app.post("/translate/batch")
def app_translate_batch(
    request: Content[BatchTranslationRequest] = Depends(
        ModelParser(BatchTranslationRequest)
    ),
):
    if request.model.options is not None and "wrap" in request.model.options:
        wrap = request.model.options["wrap"]
        if isinstance(wrap, str):
            wrap = int(wrap)
    else:
        wrap = None

    recursive_translate(request.model.batch, iterations=request.model.n, wrap=wrap)

    return Response(
        content=request.converter(request.model.batch),
        headers=request.converter.headers(),
    )


def recursive_translate(batch: Batch, iterations: int, wrap=None):

    for k, b in batch.items():
        if "_IGNORE_" not in k:
            if isinstance(b, dict):
                recursive_translate(b, iterations, wrap)
            else:
                output = translate(b, TRANSLATORS, LANGUAGE_KEYS, iterations=iterations)

                if wrap is not None:
                    output = os.linesep.join(textwrap.wrap(output, width=wrap))

                batch[k] = output
