from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .cuda import detect_cuda, parse_cuda
from .errors import WheelgetError
from .http import HttpClient, download
from .providers import get_provider
from .wheels import Target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wheelget",
        description=(
            "Trova la wheel piu' recente di vllm/torch compatibile con il driver CUDA "
            "installato e ne stampa il link (url) o la scarica (get)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("url", "stampa il link della wheel"),
        ("get", "scarica la wheel nella directory corrente"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("package", help="pacchetto da cercare: vllm o torch")
        sub.add_argument(
            "version",
            nargs="?",
            default=None,
            help="versione del pacchetto (default: la piu' recente compatibile)",
        )
        sub.add_argument(
            "--cuda",
            metavar="VERSIONE",
            help="forza la versione CUDA (es. 12.6, 12, cu126); default: rilevata",
        )
        sub.add_argument(
            "--variant",
            metavar="cuXXX",
            help="forza la variante CUDA (es. cu128)",
        )
        sub.add_argument(
            "--python",
            dest="python_version",
            metavar="X.Y",
            help="versione Python target (default: quella corrente)",
        )
        sub.add_argument("--refresh", action="store_true", help="ignora la cache HTTP")
        sub.add_argument("-q", "--quiet", action="store_true", help="stampa solo il risultato")
        if command == "get":
            sub.add_argument(
                "-o",
                "--output",
                metavar="DIR",
                default=".",
                help="directory di destinazione (default: .)",
            )
            sub.add_argument("-f", "--force", action="store_true", help="sovrascrive file esistenti")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        return 130
    except WheelgetError as exc:
        print(f"wheelget: error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    target = Target.current(python=_parse_python(args.python_version))
    if args.cuda:
        cuda = parse_cuda(args.cuda)
        cuda_source = "da --cuda"
    else:
        detected = detect_cuda()
        if detected is None:
            raise WheelgetError(
                "CUDA non rilevata (nvidia-smi/nvcc non trovati); usa --cuda X.Y"
            )
        cuda, cuda_source = detected
    provider = get_provider(args.package)
    client = HttpClient(refresh=args.refresh, quiet=args.quiet)
    resolution = provider.resolve(
        client,
        version=args.version,
        variant=args.variant,
        cuda=cuda,
        target=target,
    )
    if not args.quiet:
        print(
            f"wheelget: {resolution.package} {resolution.version} "
            f"[{resolution.variant or 'cpu'}] per CUDA {cuda[0]}.{cuda[1]} "
            f"({cuda_source}), python {target.python[0]}.{target.python[1]} "
            f"{target.os}/{target.arch}",
            file=sys.stderr,
        )
        if resolution.note:
            print(f"wheelget: nota: {resolution.note}", file=sys.stderr)
    if args.command == "url":
        print(resolution.wheel.url)
        return 0
    output = Path(args.output).expanduser()
    path = download(
        resolution.wheel.url,
        output / resolution.wheel.filename,
        quiet=args.quiet,
        force=args.force,
        sha256=resolution.wheel.sha256,
    )
    print(path)
    return 0


def _parse_python(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if match is None:
        raise WheelgetError(f"invalid --python {value!r}: expected X.Y (e.g. 3.12)")
    return (int(match.group(1)), int(match.group(2)))
