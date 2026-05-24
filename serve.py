import uvicorn
from fastapi import FastAPI, Depends, Response
from languages import LANGUAGES
from pydantic import BaseModel
from translators import TRANSLATORS, translate
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
    batch: dict[str, str]


@app.post("/translate/batch")
def app_translate_batch(
    request: Content[BatchTranslationRequest] = Depends(
        ModelParser(BatchTranslationRequest)
    ),
):
    batch = dict()

    for k, t in request.model.batch.items():
        if "_IGNORE_" in k:
            batch[k] = t
        else:
            batch[k] = translate(
                t, TRANSLATORS, LANGUAGE_KEYS, iterations=request.model.n
            )

    return Response(
        content=request.converter(batch),
        headers=request.converter.headers(),
    )
