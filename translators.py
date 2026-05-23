import gc
import os
import random
import subprocess
import sys
import torch
from abc import abstractmethod
from dataclasses import dataclass
from huggingface_hub import errors
from languages import LANGUAGES, Lang, EN
from sacremoses import MosesDetokenizer
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)
from unidecode import unidecode

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

    @dataclass
    class ModelData:
        tokenizer: MarianTokenizer
        model: MarianMTModel

    def __init__(self, languages: dict[Lang, dict[str, str]]):
        self._languages = _parse_langs(self.DICT_KEY, languages)
        self._model_cache: dict[Lang, MarianTranslator.ModelData] = {}

    def name(self) -> str:
        return "Marian"

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(self, text: str, src: Lang, target: Lang) -> str:
        src = self._languages[src]
        target = self._languages[target]

        data = self.load_model(src, target)
        batch = data.tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            generated = data.model.generate(
                **batch,
                # top_k=50,
                # temperature=1.2,
            )

        return data.tokenizer.decode(generated[0], skip_special_tokens=True)

    def load_model(self, src: str, target: str) -> MarianTranslator.ModelData:
        key = f"{src}->{target}"

        if DEVICE == "cuda":
            self.gc_check(key)

        if key in self._model_cache:
            return self._model_cache[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{target}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)

        data = MarianTranslator.ModelData(tokenizer, model)
        self._model_cache[key] = data

        return data

    def gc_check(self, current_key: str):
        used = self.get_vram_usage_percentage()
        if used > 0.70:
            for key in self._model_cache.keys():
                if key != current_key:
                    del self._model_cache[key]

            gc.collect()
            torch.cuda.empty_cache()

    def get_vram_usage_percentage(self):
        free_vram, total_vram = torch.cuda.mem_get_info()
        used_vram = total_vram - free_vram
        return used_vram / total_vram


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
                [self.bin_path, "-f", self.ini_path(src, EN)],
                input=ensure_newline(text),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            text = self.unesc(result.stdout, EN)

        if target != EN:
            result = subprocess.run(
                [self.bin_path, "-f", self.ini_path(EN, target)],
                input=ensure_newline(text),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            text = self.unesc(result.stdout, target)

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


def ensure_newline(text: str) -> str:
    if not text.endswith("\n"):
        return f"{text}\n"
    else:
        return text


TRANSLATORS = [MarianTranslator(LANGUAGES), FacebookTranslator(LANGUAGES)]


def translate(
    text: str, translators: list[Translator], languages: list[Lang], iterations=10
) -> str:
    print(f"Translating {text}", file=sys.stderr)

    src = EN
    i = 0
    translator = None
    while True:
        translator = random.choice(translators)
        target = random.choice(languages)
        result = try_translate(translator, src, target, text)

        if result is not None:
            text = result
            src = target
            i += 1

        if i >= iterations:
            if src != EN:
                result = try_translate(random.choice(translators), src, EN, text)

                if result is not None:
                    text = result
                else:
                    continue

            break

    return unidecode(text)


def try_translate(
    translator: Translator, src: Lang, target: Lang, text: str
) -> Optional[str]:
    try:
        if not translator.supports(src, target):
            print(
                f"{translator.name()} does not support {src}->{target}",
                file=sys.stderr,
            )
            return None

        text = translator.translate(text, src, target)

        print(
            f"Translated {src}->{target} with {translator.name()}: {text}",
            file=sys.stderr,
        )
    except EnvironmentError:
        return None
    except errors.RepositoryNotFoundError:
        print(
            f"Failed to acquire {src}-{target} with {translator.name()}",
            file=sys.stderr,
        )
        return None

    return text


def inject_moses(only_moses: bool):
    moses_bin = os.getenv("BABEL_MOSES_BIN")
    moses_models = os.getenv("BABEL_MOSES_MODELS")
    if moses_bin is not None:
        if not os.path.exists(moses_bin):
            print(
                "Set the BABEL_MOSES_BIN to set the path to where moses is located",
                file=sys.stderr,
            )
            exit(1)

        if not os.path.exists(moses_models):
            print(
                "Set the BABEL_MOSES_MODELS to set the path to where the models are located",
                file=sys.stderr,
            )
            exit(1)

        if only_moses:
            TRANSLATORS.clear()
            TRANSLATORS.append(MosesTranslator(moses_bin, moses_models))
        else:
            TRANSLATORS.append(MosesTranslator(moses_bin, moses_models))
