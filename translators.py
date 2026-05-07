import torch
import subprocess
import os
from abc import abstractmethod
from languages import LANGUAGES, Lang, EN
from sacremoses import MosesDetokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Translator:
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def supports(self, src: Lang, target: Lang) -> bool:
        pass

    @abstractmethod
    def translate(self, text: str, src: Lang, target: Lang) -> str:
        pass


def _parse_langs(key: str, languages: dict[Lang, dict[str, str]]) -> dict[Lang, str]:
    return {k: v[key] for k, v in languages.items() if v[key] is not None}


class MarianTranslator(Translator):
    DICT_KEY = "marian"

    def __init__(self, languages: dict[Lang, dict[str, str]]):
        self._languages = _parse_langs(self.DICT_KEY, languages)
        self._model_cache: dict[Lang, tuple[MarianTokenizer, MarianMTModel]] = {}

    def name(self) -> str:
        return "Marian"

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(self, text: str, src: Lang, target: Lang) -> str:
        src = self._languages[src]
        target = self._languages[target]
        tokenizer, model = self.load_model(src, target)

        batch = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(
                **batch,
                # top_k=50,
                # temperature=1.2,
            )

        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def load_model(
        self, src: str, target: str
    ) -> tuple[MarianTokenizer, MarianMTModel]:
        key = f"{src}->{target}"
        if key in self._model_cache:
            return self._model_cache[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{target}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        self._model_cache[key] = (tokenizer, model)
        return tokenizer, model


class FacebookTranslator(Translator):
    DICT_KEY = "nllb"
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self, languages: dict[Lang, dict[str, str]]):
        self._languages = _parse_langs(self.DICT_KEY, languages)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME).to(DEVICE)

    def name(self) -> str:
        return "Nllb"

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(self, text: str, src: Lang, target: Lang) -> str:
        src = self._languages[src]
        target = self._languages[target]

        self._tokenizer.src_lang = src

        encoded = self._tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            out = self._model.generate(
                **encoded,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target),
                do_sample=True,
                # top_k=50,
                # temperature=1.2,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


class MosesTranslator(Translator):
    def __init__(self, bin_path: str, model_path: str):
        self.bin_path = bin_path
        self.model_path = model_path
        self._esc_cache: dict[Lang, MosesDetokenizer] = dict()

    def name(self) -> str:
        return "Moses"

    def supports(self, src: Lang, target: Lang) -> bool:
        if (src == EN and target != EN) or (src != EN and target == EN):
            return os.path.exists(self.ini_path(src, target))
        else:
            return os.path.exists(self.ini_path(src, EN)) and os.path.exists(
                self.ini_path(EN, target)
            )

    def translate(self, text: str, src: Lang, target: Lang) -> str:
        if src != EN:
            result = subprocess.run(
                [self.bin_path, "-f", self.ini_path(src, "en")],
                input=text,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            text = self.unesc(result.stdout)

        if target != EN:
            result = subprocess.run(
                [self.bin_path, "-f", self.ini_path("en", target)],
                input=text,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            text = self.unesc(result.stdout)

        return text

    def ini_path(self, src: Lang, target: Lang) -> str:
        return f"{self.model_path}/{src.id}-{target.id}/model/moses.ini"

    def unesc(self, text: str, lang: Lang):
        if lang in self._esc_cache:
            md = self._esc_cache[lang]
        else:
            md = MosesDetokenizer(lang=lang.id)
            self._esc_cache[lang] = md
        tokens = text.split()
        return md.detokenize(tokens)


TRANSLATORS = [MarianTranslator(LANGUAGES), FacebookTranslator(LANGUAGES)]
