import os
import sys
from argparse import ArgumentParser
from util import log


def main():
    parser = ArgumentParser(prog="Babel", description="Nonsense")
    parser.add_argument("--serve", help="Run as a webserver", required=False, type=int)
    parser.add_argument(
        "--moses-only", help="Only use moses", required=False, action="store_true"
    )
    parser.add_argument("-t", "--text", help="Text to translate", required=False)
    parser.add_argument(
        "-i", "--input", help="Text to translate from a file", required=False
    )
    parser.add_argument(
        "-o", "--output", help="Where to write the text output", required=False
    )
    parser.add_argument("-n", help="Number of times to iterate", default=10, type=int)
    parser.add_argument(
        "--cpu",
        help="Allow cpu to be used for translating",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--no-newline",
        help="Strip the trailing newline",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--temp", help="Temperature setting", required=False, type=float
    )
    parser.add_argument("--topk", help="Top K result setting", required=False, type=int)
    args = parser.parse_args()

    from languages import LANGUAGES
    from translators import TRANSLATORS, DEVICE, translate, inject_moses

    if DEVICE == "cpu" and not args.cpu:
        log("Use --cpu to allow for cpu translation")
        exit(1)

    inject_moses(args.moses_only)

    if args.serve is None:
        text = args.text

        if args.input is not None:
            with open(args.input) as f:
                text = f.read()

        output = os.linesep.join(
            [
                translate(
                    text,
                    TRANSLATORS,
                    list(LANGUAGES.keys()),
                    iterations=args.n,
                    temp=args.temp,
                    top_k=args.topk,
                )
                for text in text.splitlines()
            ]
        )

        if args.output is not None:
            with open(args.output, "w") as f:
                f.write(output)
        else:
            end = None if args.no_newline else os.linesep
            print(output, end=end)
    else:
        import serve

        serve.run_server(args.serve)


if __name__ == "__main__":
    main()
