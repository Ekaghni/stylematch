"""Command line interface: `stylematch a.txt b.txt`."""

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .core import compare


def _read(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stylematch",
        description="Compare the writing style of two texts.",
        epilog=("Other modes: `stylematch detect FILE` checks one text for AI authorship "
                "(needs `pip install \"stylematch[ai]\"`), `stylematch gui` opens the desktop window."),
    )
    parser.add_argument("first", nargs="?", help="path to the first text file, or - for stdin")
    parser.add_argument("second", nargs="?", help="path to the second text file")
    parser.add_argument("--text", nargs=2, metavar=("TEXT1", "TEXT2"),
                        help="compare two strings given directly on the command line")
    parser.add_argument("--string", metavar="TEXT", help="with `detect`: check this string instead of a file")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="with `detect`: where to run the model (default: auto)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--version", action="version", version=f"stylematch {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)

    if args.first == "gui" and args.second is None and not args.text:
        from .gui import launch

        return launch()

    if args.first == "detect":
        return _run_detect(args, parser)

    try:
        if args.text:
            text1, text2 = args.text
        elif args.first and args.second:
            text1, text2 = _read(args.first), _read(args.second)
        else:
            parser.print_usage(sys.stderr)
            print("stylematch: give two files, or use --text, or run `stylematch gui`.",
                  file=sys.stderr)
            return 2
        result = compare(text1, text2)
    except (OSError, ValueError) as err:
        print(f"stylematch: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Similarity score : {result.score:.4f}")
        print(f"Reading          : {result.verdict}")
        print(f"Words            : {result.words_1} vs {result.words_2}")
        for note in result.warnings:
            print(f"Note             : {note}", file=sys.stderr)
    return 0


def _run_detect(args, parser) -> int:
    from .detect import MissingDependencyError, detect

    try:
        if args.string:
            text = args.string
        elif args.second:
            text = _read(args.second)
        else:
            parser.print_usage(sys.stderr)
            print("stylematch: detect needs a file (or - for stdin), or --string.", file=sys.stderr)
            return 2
        result = detect(text, device=args.device)
    except MissingDependencyError as err:
        print(f"stylematch: {err}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as err:
        print(f"stylematch: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"AI score       : {result.score:.3f}")
        print(f"Reading        : {result.verdict}")
        print(f"Model / device : {result.model} on {result.device}")
        print(f"Words          : {result.words}")
        for note in result.warnings:
            print(f"Note           : {note}", file=sys.stderr)
    return 0
