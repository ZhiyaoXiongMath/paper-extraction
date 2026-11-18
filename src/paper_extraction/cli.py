import argparse
import os
import shutil
import subprocess
from pathlib import Path

from . import __version__
from .settings import PROVIDERS, probe, public_config


def doctor(args):
    ok = True
    for program in ("pdflatex","kpsewhich"):
        present = shutil.which(program)
        print(f"{program}: {present or 'missing'}")
        ok = ok and bool(present)
    if shutil.which("kpsewhich"):
        for package in ("newtxtext.sty","newtxmath.sty","t1xtt.tfm","needspace.sty","microtype.sty",
                        "mathrsfs.sty","enumitem.sty","titlesec.sty","hyperref.sty","etoolbox.sty"):
            found = subprocess.run(["kpsewhich",package],capture_output=True).returncode == 0
            print(f"{package}: {'ok' if found else 'missing'}")
            ok = ok and found
    config = public_config()
    provider = args.provider or config["default_provider"]
    for item in config["providers"]:
        print(f"{item['label']}: {item['text_model']} / {item['vision_model']} | {item['key_env']} {'configured' if item['configured'] else 'missing'}")
    if args.live:
        for mode in ("text","vision"):
            result = probe(provider,mode)
            print(f"{mode}: {result['message']}")
            ok = ok and result["ok"]
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(prog="paper-extraction")
    parser.add_argument("--version",action="version",version=__version__)
    parser.add_argument("--env-file",type=Path,help="path to the independent provider .env file")
    cmd = parser.add_subparsers(dest="command",required=True)
    serve = cmd.add_parser("serve",help="open the local workbench server")
    serve.add_argument("--port",type=int,default=8765)
    check = cmd.add_parser("doctor",help="inspect TeX installation and model configuration")
    check.add_argument("--live",action="store_true",help="test text and visual API calls")
    check.add_argument("--provider",choices=list(PROVIDERS))
    sample = cmd.add_parser("example", help="copy the bundled TeX example to a new folder")
    sample.add_argument("directory", nargs="?", type=Path, default=Path("compactness"))
    args = parser.parse_args()
    if args.env_file:
        os.environ["PAPER_EXTRACTION_ENV_FILE"] = str(args.env_file.resolve())
    if args.command == "doctor":
        return doctor(args)
    if args.command == "example":
        package = Path(__file__).resolve().parent
        source = package / "examples" / "compactness"
        if not source.is_dir():
            source = package.parents[1] / "examples" / "compactness"
        if args.directory.exists():
            parser.error("choose a new directory for the sample")
        shutil.copytree(source, args.directory)
        print(args.directory.resolve())
        return 0
    import uvicorn
    from .api import create_app
    print(f"Paper Extraction: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(),host="127.0.0.1",port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
