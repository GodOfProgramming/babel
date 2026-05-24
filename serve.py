import textwrap
import uvicorn
from fastapi import FastAPI, Depends, Response
from languages import LANGUAGES
from pydantic import BaseModel
from translators import TRANSLATORS, translate
from typing import Optional
from util import ModelParser, Content

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


class BatchTranslationRequest(BaseModel):
    n: int
    options: Optional[dict[str, int | str]]
    batch: dict[str, str]


@app.post("/translate/batch")
def app_translate_batch(
    request: Content[BatchTranslationRequest] = Depends(
        ModelParser(BatchTranslationRequest)
    ),
):
    batch = dict()

    if request.model.options is not None and "wrap" in request.model.options:
        wrap = request.model.options["wrap"]
        if isinstance(wrap, str):
            wrap = int(wrap)
    else:
        wrap = None

    for k, t in request.model.batch.items():
        if "_IGNORE_" in k:
            batch[k] = t
        else:
            output = translate(
                t, TRANSLATORS, LANGUAGE_KEYS, iterations=request.model.n
            )

            if wrap is not None:
                output = "\n".join(textwrap.wrap(output, width=wrap))

            batch[k] = output

    return Response(
        content=request.converter(batch),
        headers=request.converter.headers(),
    )
