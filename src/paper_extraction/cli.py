import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import __version__
from .settings import PROVIDERS, data_home, probe, public_config


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
    backfill = cmd.add_parser("backfill-drafts", help="recover draft files from existing failed checkpoints locally")
    backfill.add_argument("--paper", help="recover one paper ID")
    prepare = cmd.add_parser("prepare-revision", help="build a source-bound engineering revision and render its pages for review")
    prepare.add_argument("--paper", required=True)
    prepare.add_argument("--attempt", type=int, required=True)
    prepare.add_argument("--round", type=int, required=True)
    prepare.add_argument("--draft-sha", required=True)
    prepare.add_argument("--patch", type=Path, required=True)
    accept = cmd.add_parser("accept-revision", help="record an operator's source and rendered-page inspection")
    accept.add_argument("--paper", required=True)
    accept.add_argument("--case", required=True, type=Path)
    accept.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    if args.env_file:
        os.environ["PAPER_EXTRACTION_ENV_FILE"] = str(args.env_file.resolve())
    if args.command == "doctor":
        return doctor(args)
    if args.command == "backfill-drafts":
        from .store import Store
        print(f"Recovered draft versions: {Store(data_home()).backfill_drafts(args.paper)}")
        return 0
    if args.command in {"prepare-revision", "accept-revision"}:
        from .reviewed_revision import accept_revision, prepare_revision
        from .store import Store
        store = Store(data_home())
        if args.command == "prepare-revision":
            outcome = prepare_revision(store, args.paper, args.attempt, args.round, args.draft_sha, args.patch)
        else:
            outcome = accept_revision(store, args.paper, args.case, args.audit)
        print(json.dumps(outcome, ensure_ascii=False))
        return 0
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
