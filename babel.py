from languages import Lang, EN
from translators import Translator, DEVICE, MosesTranslator
import os
import sys
import random
from huggingface_hub import errors


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
                print(
                    f"{translator.name()} does not support {src}->{target}",
                    file=sys.stderr,
                )
                continue
            print(
                f"Translating {src}->{target} with {translator.name()}", file=sys.stderr
            )
            current = translator.translate(current, src, target)
            src = target
            i += 1
        except EnvironmentError:
            continue
        except errors.RepositoryNotFoundError:
            print(
                f"Failed to acquire {src}-{target} with {translator.name()}",
                file=sys.stderr,
            )
            continue
        if i >= iterations:
            if src != EN and translator is not None:
                try:
                    current = translator.translate(current, src, EN)
                except errors.RepositoryNotFoundError:
                    print(
                        f"Failed to translate {src}->{EN}, rerolling with another language",
                        file=sys.stderr,
                    )
                    continue

            break

    return current


if __name__ == "__main__":
    from argparse import ArgumentParser
    from languages import LANGUAGES
    from translators import TRANSLATORS

    parser = ArgumentParser(prog="Babel", description="Nonsense")
    parser.add_argument("-t", "--text", help="Text to translate", required=False)
    parser.add_argument(
        "-i", "--input", help="Text to translate from a file", required=False
    )
    parser.add_argument(
        "-o", "--output", help="Where to write the text output", required=False
    )
    parser.add_argument("-n", help="Number of times to iterate", default=10, type=int)
    parser.add_argument("--moses", help="Enable the moses translator")
    parser.add_argument("--moses-models", help="Path to moses model location")
    parser.add_argument(
        "--cpu",
        help="Allow cpu to be used for translating",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()

    if DEVICE == "cpu" and not args.cpu:
        print("Use --cpu to allow for cpu translation", file=sys.stderr)
        exit(1)

    if len(args.moses) != 0:
        if len(args.moses_models) == 0:
            print(
                "Use --moses-models to set the path to where the models are located",
                file=sys.stderr,
            )
            exit(1)
        TRANSLATORS.append(MosesTranslator(args.moses, args.moses_models))

    text = args.text

    if args.input is not None:
        with open(args.input) as f:
            text = f.read()

    output = os.linesep.join(
        [
            text_bable(text, TRANSLATORS, list(LANGUAGES.keys()), iterations=args.n)
            for text in text.splitlines()
        ]
    )

    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)
