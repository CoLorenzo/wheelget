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
from .providers.torch import TorchProvider
from .providers.vllm import torch_pin_from_wheel
from .wheels import Target, Wheel, parse_wheel


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
    compatibility = subparsers.add_parser(
        "torch-compability",
        aliases=["torch-compatibility"],
        help="stampa la versione Python richiesta da una wheel torch",
    )
    compatibility.add_argument(
        "wheel",
        nargs="?",
        help="percorso della wheel torch (default: torch*.whl nella directory corrente)",
    )
    backend = subparsers.add_parser(
        "torch-backend",
        help="stampa il valore da usare con --torch-backend di uv (es. cu130)",
    )
    backend.add_argument(
        "--cuda",
        metavar="VERSIONE",
        help="forza la versione CUDA (es. 12.6, 12, cu126); default: rilevata",
    )
    backend.add_argument("--refresh", action="store_true", help="ignora la cache HTTP")
    backend.add_argument("-q", "--quiet", action="store_true", help="stampa solo il risultato")
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
    if args.command in ("torch-compability", "torch-compatibility"):
        return _run_torch_compability(args)
    if args.command == "torch-backend":
        return _run_torch_backend(args)
    target = Target.current(python=_parse_python(args.python_version))
    cuda, cuda_source = _cuda_from_args(args)
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
    if resolution.package == "vllm":
        companion = _torch_companion(
            client=client,
            vllm_path=path,
            resolution=resolution,
            cuda=cuda,
            target=target,
            output=output,
            quiet=args.quiet,
            force=args.force,
        )
        if companion is not None:
            print(companion)
    return 0


def _torch_companion(
    *,
    client: HttpClient,
    vllm_path: Path,
    resolution,
    cuda: tuple[int, int],
    target: Target,
    output: Path,
    quiet: bool,
    force: bool,
) -> Path | None:
    pin = torch_pin_from_wheel(vllm_path)
    if pin is None:
        if not quiet:
            print(
                f"wheelget: nota: nessun pin torch== trovato nel METADATA di {vllm_path.name}",
                file=sys.stderr,
            )
        return None
    provider = TorchProvider()
    attempts = [resolution.variant, None] if resolution.variant else [None]
    last_error: WheelgetError | None = None
    for variant in attempts:
        try:
            torch_resolution = provider.resolve(
                client, version=pin, variant=variant, cuda=cuda, target=target
            )
        except WheelgetError as exc:
            last_error = exc
            continue
        if not quiet:
            print(
                f"wheelget: torch {torch_resolution.version} "
                f"[{torch_resolution.variant or 'cpu'}] richiesto da vllm {resolution.version}",
                file=sys.stderr,
            )
        return download(
            torch_resolution.wheel.url,
            output / torch_resolution.wheel.filename,
            quiet=quiet,
            force=force,
            sha256=torch_resolution.wheel.sha256,
        )
    if not quiet:
        print(
            f"wheelget: nota: torch=={pin} richiesto da vllm non e' disponibile "
            f"per CUDA {cuda[0]}.x ({last_error})",
            file=sys.stderr,
        )
    return None


def _cuda_from_args(args: argparse.Namespace) -> tuple[tuple[int, int], str]:
    if args.cuda:
        return parse_cuda(args.cuda), "da --cuda"
    detected = detect_cuda()
    if detected is None:
        raise WheelgetError(
            "CUDA non rilevata (nvidia-smi/nvcc non trovati); usa --cuda X.Y"
        )
    return detected


def _run_torch_backend(args: argparse.Namespace) -> int:
    cuda, cuda_source = _cuda_from_args(args)
    target = Target.current()
    client = HttpClient(refresh=args.refresh, quiet=args.quiet)
    resolution = TorchProvider().resolve(
        client, version=None, variant=None, cuda=cuda, target=target
    )
    if not args.quiet:
        print(
            f"wheelget: torch {resolution.version} [{resolution.variant}] "
            f"per CUDA {cuda[0]}.{cuda[1]} ({cuda_source}), "
            f"python {target.python[0]}.{target.python[1]} {target.os}/{target.arch}",
            file=sys.stderr,
        )
    print(resolution.variant)
    return 0


def _run_torch_compability(args: argparse.Namespace) -> int:
    path = _find_torch_wheel(args.wheel)
    wheel = parse_wheel(path.name, url=path.as_uri())
    if wheel is None:
        raise WheelgetError(f"{path.name} non e' un nome di wheel valido")
    if wheel.name != "torch":
        raise WheelgetError(f"{path.name} non e' una wheel di torch")
    version = python_version_for_wheel(wheel)
    if version is None:
        raise WheelgetError(f"nessun tag Python riconosciuto in {path.name}")
    print(version)
    return 0


def _find_torch_wheel(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_file():
            raise WheelgetError(f"file non trovato: {path}")
        return path
    matches = sorted(Path.cwd().glob("torch*.whl"))
    if not matches:
        raise WheelgetError(
            "nessun torch*.whl nella directory corrente; passa il percorso della wheel"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise WheelgetError(f"piu' wheel trovate ({names}); passa il percorso")
    return matches[0]


def python_version_for_wheel(wheel: Wheel) -> str | None:
    versions = []
    for tag in wheel.py_tags:
        match = re.fullmatch(r"cp(\d)(\d+)", tag)
        if match:
            versions.append((int(match.group(1)), int(match.group(2))))
    if not versions:
        for tag in wheel.py_tags:
            match = re.fullmatch(r"py(\d)(\d+)", tag)
            if match and match.group(1) == "3":
                versions.append((3, int(match.group(2))))
    if not versions:
        return "3" if "py3" in wheel.py_tags else None
    major, minor = min(versions)
    return f"{major}.{minor}"


def _parse_python(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if match is None:
        raise WheelgetError(f"invalid --python {value!r}: expected X.Y (e.g. 3.12)")
    return (int(match.group(1)), int(match.group(2)))
