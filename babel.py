from transformers import MarianMTModel, MarianTokenizer
import torch
import random

DEVICE = "cuda"

_model_cache = {}


def load_model(src: str, target: str) -> (MarianTokenizer, MarianMTModel):
    key = f"{src}->{target}"
    if key in _model_cache:
        return _model_cache[key]

    model_name = f"Helsinki-NLP/opus-mt-{src}-{target}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    _model_cache[key] = (tokenizer, model)
    return tokenizer, model


def translate(text: str, src: str, target: str) -> str:
    print(f"translating {src} to {target}")
    tokenizer, model = load_model(src, target)

    batch = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(**batch)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def text_bable(text: str, lang_chain: list[str], iterations=10) -> str:
    current = text
    src = "en"

    i = 0
    while True:
        target = random.choice(lang_chain)
        try:
            current = translate(current, src, target)
            src = target
            i += 1
        except EnvironmentError:
            continue
        if i >= iterations:
            break

    if src != "en":
        current = translate(current, src, "en")

    return current


if __name__ == "__main__":
    from languages import LANGUAGES
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Babel", description="Nonsense")
    parser.add_argument("-t", "--text", help="Text to translate", required=True)
    parser.add_argument("-n", help="number of times to iterate", default=10)
    args = parser.parse_args()

    output = text_bable(args.text, LANGUAGES, iterations=args.n)

    print(output)
