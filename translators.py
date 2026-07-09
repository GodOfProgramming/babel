import fasttext
import gc
import os
import random
import re
import subprocess
import sys
import textwrap
import torch
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from huggingface_hub import errors
from languages import LANGUAGES, LanguageMap, Lang, EN
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
    def translate(
        self,
        text: str,
        src: Lang,
        target: Lang,
        temp: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        pass


def _parse_langs(key: str, languages: LanguageMap) -> dict[Lang, str]:
    return {k: val for k, v in languages.items() if (val := v[key]) is not None}


class MarianTranslator(Translator):
    DICT_KEY = "marian"

    @dataclass
    class ModelData:
        tokenizer: MarianTokenizer
        model: MarianMTModel

    def __init__(self, languages: LanguageMap):
        self._languages = _parse_langs(self.DICT_KEY, languages)
        self._model_cache: OrderedDict[str, MarianTranslator.ModelData] = OrderedDict()

    def name(self) -> str:
        return "Marian"

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(
        self,
        text: str,
        src: Lang,
        target: Lang,
        temp: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        src_id = self._languages[src]
        target_id = self._languages[target]

        data = self.load_model(src_id, target_id)
        batch = data.tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            generated = data.model.generate(
                **batch,
                do_sample=temp is not None or top_k is not None,
                top_k=top_k,
                temperature=temp,
            )

        sentence = data.tokenizer.decode(generated[0], skip_special_tokens=True)

        if isinstance(sentence, list):
            return os.linesep.join(sentence)
        else:
            return sentence

    def load_model(self, src: str, target: str) -> MarianTranslator.ModelData:
        key = f"{src}->{target}"

        if key in self._model_cache:
            self._model_cache.move_to_end(key)

        if DEVICE == "cuda":
            self.gc_check()

        if key in self._model_cache:
            return self._model_cache[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{target}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)

        data = MarianTranslator.ModelData(tokenizer=tokenizer, model=model)
        self._model_cache[key] = data

        return data

    def gc_check(self):
        cleared = False
        while True:
            used = self.get_vram_usage_percentage()
            if used < 0.90 or len(self._model_cache) == 0:
                break

            key, data = self._model_cache.popitem(last=False)
            model_size = self.model_size(data.model)
            print(
                f"Freeing {model_size // (1024 ** 2)} MB from Marian cache ({key})",
                file=sys.stderr,
            )
            del data
            cleared = True

        if cleared:
            gc.collect()
            torch.cuda.empty_cache()

    def get_vram_usage_percentage(self):
        free_vram, total_vram = torch.cuda.mem_get_info()
        used_vram = total_vram - free_vram
        return used_vram / total_vram

    def model_size(self, model: MarianMTModel) -> int:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return param_size + buffer_size


class NllbTranslator(Translator):
    DICT_KEY = "nllb"
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self, languages: LanguageMap):
        self._languages = _parse_langs(self.DICT_KEY, languages)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME).to(DEVICE)

    def name(self) -> str:
        return "Nllb"

    def supports(self, src: Lang, target: Lang) -> bool:
        return src in self._languages and target in self._languages

    def translate(
        self,
        text: str,
        src: Lang,
        target: Lang,
        temp: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        src_id = self._languages[src]
        target_id = self._languages[target]

        self._tokenizer.src_lang = src_id

        encoded = self._tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            out = self._model.generate(
                **encoded,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_id),
                do_sample=temp is not None or top_k is not None,
                top_k=top_k,
                temperature=temp,
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

    def translate(
        self,
        text: str,
        src: Lang,
        target: Lang,
        temp: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if src != EN:
            parts = sanitize_for_moses(text)
            for i in range(len(parts)):
                result = subprocess.run(
                    [self.bin_path, "-f", self.ini_path(src, EN)],
                    input=ensure_newline(text),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

                out = self.unesc(result.stdout, EN)
                parts[i] = out
            text = " ".join(parts)

        if target != EN:
            parts = sanitize_for_moses(text)
            for i in range(len(parts)):
                result = subprocess.run(
                    [self.bin_path, "-f", self.ini_path(EN, target)],
                    input=ensure_newline(text),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

                out = self.unesc(result.stdout, target)
                parts[i] = out
            text = " ".join(parts)

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
    if not text.endswith(os.linesep):
        return f"{text}{os.linesep}"
    else:
        return text


NLLB_TRANSLATOR_INSTANCE = NllbTranslator(LANGUAGES)

TRANSLATORS = [MarianTranslator(LANGUAGES), NLLB_TRANSLATOR_INSTANCE]


class EnglishChecker:
    def __init__(self):
        # fetch from https://fasttext.cc/docs/en/language-identification.html
        self.model = fasttext.load_model("lid.176.bin")

    def validate(self, text: str, confidence_thresh: Optional[float] = None) -> str:
        if confidence_thresh is None:
            confidence_thresh = 0.85

        try:
            predictions = self.model.predict(text, k=1)

            if len(predictions) < 2:
                return text

            possible_languages = predictions[0]
            confidences = predictions[1]

            if len(possible_languages) == 0 or len(confidences) == 0:
                return text

            detected_lang = possible_languages[0].replace("__label__", "")
            confidence = confidences[0]

            if detected_lang == "en" and confidence > confidence_thresh:
                return text

            src = Lang(detected_lang)

            if NLLB_TRANSLATOR_INSTANCE.supports(src, EN):
                print(
                    f"Force translating to english, detected language {detected_lang} ({confidence * 100}% < {confidence_thresh * 100}%)",
                    file=sys.stderr,
                )

                output = try_translate(NLLB_TRANSLATOR_INSTANCE, src, EN, text)
                return output or text

            return text
        except:
            return text


CHECKER = EnglishChecker()


def translate(
    text: str,
    translators: list[Translator],
    languages: list[Lang],
    iterations=10,
    temp: Optional[float] = None,
    top_k: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
) -> str:
    print(f"Translating {text}", file=sys.stderr)

    src = EN
    i = 0
    translator = None
    while True:
        if i >= iterations:
            if src != EN:
                result = try_translate(
                    random.choice(translators), src, EN, text, temp=temp, top_k=top_k
                )

                if result is not None:
                    text = result
                else:
                    continue

            break

        translator = random.choice(translators)
        target = random.choice(languages)
        result = try_translate(translator, src, target, text, temp=temp, top_k=top_k)

        if result is not None:
            text = result
            src = target
            i += 1

    output = CHECKER.validate(unidecode(text), confidence_threshold)

    return output


def try_translate(
    translator: Translator,
    src: Lang,
    target: Lang,
    text: str,
    temp: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Optional[str]:
    try:
        if not translator.supports(src, target):
            print(
                f"{translator.name()} does not support {src}->{target}",
                file=sys.stderr,
            )
            return None

        text = translator.translate(text, src, target, temp=temp, top_k=top_k)

        print(
            f"Translated {src}->{target} with {translator.name()}: {text}",
            file=sys.stderr,
        )
    except errors.RepositoryNotFoundError:
        print(
            f"Failed to acquire {src}-{target} with {translator.name()}",
            file=sys.stderr,
        )
        return None
    except EnvironmentError:
        return None

    return text


def inject_moses(only_moses: bool):
    moses_bin = os.getenv("BABEL_MOSES_BIN")
    moses_models = os.getenv("BABEL_MOSES_MODELS")
    if moses_bin is not None and moses_models is not None:
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


def sanitize_for_moses(text: str) -> list[str]:
    # Force utf8
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Replace newlines with spaces so moses decodes whole lines
    text = text.replace("\n", " ").replace("\r", " ")

    # Strip ASCII control characters, pipes
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    text = text.replace("|", "")

    # Escape other characters that moses can accept in this form
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    text = text.replace("[", "&#91;")
    text = text.replace("]", "&#93;")

    # Replace duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure the phrase fits into moses's memory, split into multiple parts
    parts = chunkify(text, 100)

    return parts


def chunkify(text: str, max_words: int = 100) -> list[str]:
    words = text.split()

    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
