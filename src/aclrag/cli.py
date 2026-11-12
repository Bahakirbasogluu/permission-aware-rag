from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aclrag.chat import ChatService, ExtractiveModel
from aclrag.corpus import load_corpus
from aclrag.evaluate import evaluate
from aclrag.models import Principal
from aclrag.retrieval import InMemoryRetriever


def _service(args) -> ChatService:
    if args.backend == "azure":
        from aclrag.chat import AzureOpenAIChatModel
        from aclrag.retrieval import AzureSearchRetriever

        return ChatService(AzureSearchRetriever(), AzureOpenAIChatModel())
    return ChatService(InMemoryRetriever(load_corpus(args.corpus)), ExtractiveModel())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aclrag")
    parser.add_argument("--backend", choices=("local", "azure"), default="local")
    parser.add_argument("--corpus", type=Path, default=Path("samples/corpus"))
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask a question as a member of the given groups")
    ask.add_argument("--groups", required=True, help="comma-separated group ids of the user")
    ask.add_argument("question")
    ev = sub.add_parser("eval", help="run the persona question set; exits 1 if anything leaked")
    ev.add_argument("--questions", type=Path, default=Path("samples/questions.csv"))
    args = parser.parse_args(argv)

    service = _service(args)
    if args.command == "ask":
        answer = service.ask(
            Principal("cli", tuple(g.strip() for g in args.groups.split(",") if g.strip())),
            args.question,
        )
        print(answer.text)
        for n, doc in enumerate(answer.cited, start=1):
            print(f"  source {n}: {doc.title}  {doc.source_url}")
        return 0
    report = evaluate(service, args.questions)
    print(report.summary())
    return 1 if report.leaks else 0


if __name__ == "__main__":
    sys.exit(main())
