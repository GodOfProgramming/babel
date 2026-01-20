from languages import Lang, EN
from translators import Translator
import random


def text_bable(
    text: str, translators: list[Translator], languages: list[Lang], iterations=10
) -> str:
    current = text
    src = EN

    i = 0
    translator = None
    while True:
        translator = random.choice(translators)
        target = random.choice(languages)
        try:
            if not translator.supports(src, target):
                print(f"{translator.name()} does not support {src}->{target}")
                continue
            print(f"Translating {src}->{target} with {translator.name()}")
            current = translator.translate(current, src, target)
            src = target
            i += 1
        except EnvironmentError:
            continue
        if i >= iterations:
            break

    if src != EN and translator is not None:
        current = translator.translate(current, src, EN)

    return current


if __name__ == "__main__":
    from argparse import ArgumentParser
    from languages import LANGUAGES
    from translators import TRANSLATORS

    parser = ArgumentParser(prog="Babel", description="Nonsense")
    parser.add_argument("-t", "--text", help="Text to translate", required=True)
    parser.add_argument("-n", help="number of times to iterate", default=10, type=int)
    args = parser.parse_args()

    output = text_bable(
        args.text, TRANSLATORS, list(LANGUAGES.keys()), iterations=args.n
    )

    print(output)
