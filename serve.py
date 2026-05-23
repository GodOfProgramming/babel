import uvicorn
from fastapi import FastAPI, Depends, Response
from languages import LANGUAGES
from pydantic import BaseModel
from translators import TRANSLATORS, translate
from util import ModelParser, Content

app = FastAPI()


def run_server(port: int):

    uvicorn.run(app, host="localhost", port=port, reload=False)


class TranslationRequest(BaseModel):
    text: str
    n: int


@app.post("/translate")
def translate(request: TranslationRequest):
    output = translate(
        request.text, TRANSLATORS, list(LANGUAGES.keys()), iterations=request.n
    )
    return output


class BatchTranslationRequest(BaseModel):
    n: int
    batch: dict[str, str]


@app.post("/translate/batch")
def translate_batch(
    request: Content[BatchTranslationRequest] = Depends(
        ModelParser(BatchTranslationRequest)
    ),
):
    batch = dict()

    for k, v in request.model.batch.items():
        batch[k] = translate(
            v, TRANSLATORS, list(LANGUAGES.keys()), iterations=request.model.n
        )

    return Response(
        content=request.converter(batch),
        headers=request.converter.headers(),
    )
