from abc import abstractmethod
from languages import LANGUAGES, Lang
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Translator:
    @abstractmethod
    def supports(self, src: Lang, target: Lang) -> bool:
        pass

    @abstractmethod
    def translate(self, text: str, src: Lang, target: Lang) -> str:
        pass


class MarianTranslator(Translator):
    def __init__(self, languages: list[Lang]):
        self._languages: dict[str, str] = {lang: lang.id for lang in languages}
        self._model_cache = {}
        print("Supported Marian", self._languages)

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(self, text: str, src: Lang, target: Lang) -> str:
        src = self._languages[src]
        target = self._languages[target]
        tokenizer, model = self.load_model(src, target)

        batch = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**batch)

        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def load_model(self, src: str, target: str) -> (MarianTokenizer, MarianMTModel):
        key = f"{src}->{target}"
        if key in self._model_cache:
            return self._model_cache[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{target}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        self._model_cache[key] = (tokenizer, model)
        return tokenizer, model


TRANSLATORS = [MarianTranslator(LANGUAGES)]
