# aclrag

A retrieval-augmented assistant for documents that not everyone may read. Two people can ask it the same question and get different answers, because it answers only from the documents the person asking has access to, and it says so when there are none.

The design comes from an intranet assistant I built as a proof of concept for a hospital group. The code here is written from scratch and runs on an invented corpus; it contains no client code or data.

## How a question is answered

```
user + groups ──> security filter ──> retrieve (filter applied in the index) ──> second check
                       │                                                            │
                 no usable groups:                                      nothing permitted matched:
                 refuse, no search                                      fixed answer, no model call
                                                                                    │
                                                    numbered sources ──> model ──> [n] citations ──> audit
```

## Design decisions

- **Access is decided at indexing time.** Every document names the groups that may read it, and those groups are stamped on each of its chunks. A document without groups is refused at ingestion rather than indexed as readable by nobody or by everybody.
- **The filter runs inside the search engine.** The user's groups become a filter on the query (`group_ids/any(g: search.in(g, 'a,b', ','))` for Azure AI Search), so an unauthorised passage is never retrieved. Filtering results after retrieval, or asking the model to respect permissions, would both put restricted text one mistake away from an answer.
- **Fail closed, twice.** A user with no groups, or a group id that could alter the filter expression, means no search at all. After retrieval the service checks every chunk against the filter again, so a retriever that ignores it is still contained.
- **No permitted source, no model call.** When nothing the user may read matches, the answer is a fixed sentence. The model is not asked, so it cannot fill the gap from general knowledge.
- **Citations are resolved in code.** The model sees numbered sources and cites `[n]`; the service maps the markers to documents and ignores numbers that do not exist.
- **A leak is what reached the model, not what was cited.** The evaluation fails a question if a forbidden document was among the passages sent to the model, even when the answer never mentions it.
- **Every answer is audited.** The log records which documents were sent to the model on whose behalf, and whether the question looked like an attempt to override the rules.

## Quick start

```bash
pip install -e ".[dev]"
aclrag ask --groups all-employees,hr "What is the salary band for a senior engineer?"
aclrag ask --groups all-employees "What is the salary band for a senior engineer?"
aclrag eval
pytest
```

The same question as an HR member and as any employee:

```
The salary band for a senior engineer runs from 95,000 to 130,000 TL per month. [1]
  source 1: Salary Bands 2026  https://intranet.example/docs/salary-bands

I could not find this in the documents you have access to.
```

## Evaluation

`samples/` holds 11 documents for four groups (all employees, HR, finance, IT) in English and Turkish, and 22 questions. Each question has a persona (the groups of the person asking) and a category:

| Category | Questions | What must happen |
| --- | --- | --- |
| normal | 11 | The expected document is cited |
| negative | 3 | The topic is not in the corpus: decline |
| authz | 5 | The answer exists, but in another group's document: decline, and that document must not reach the model |
| redteam | 3 | The question tries to override the rules: same requirement |

`aclrag eval` on the local backend:

```
unauthorised documents sent to the model: 0
expected source cited        11 / 11
declined (not in corpus)     3 / 3
declined (no access)         5 / 5
declined (injection attempt) 3 / 3
```

The command exits with 1 if the first number is not zero, so it can gate a release. One user is created per question, so nothing carries over between personas. The test suite also runs the evaluation against a deliberately broken service that skips the filter, and checks that the leaks are counted.

## Running on Azure

The local backend uses an in-memory keyword retriever and a deterministic extractive model, which keeps the evaluation reproducible. The Azure backend uses hybrid search with semantic ranking in Azure AI Search and a chat deployment in Azure OpenAI:

```bash
pip install -e ".[azure]"
cp .env.example .env   # fill in the search and model settings, then export them
aclrag --backend azure eval
```

The index needs `group_ids` as a filterable collection on every chunk, next to `id`, `doc_id`, `title`, `text`, `source_url` and a vector field with an integrated vectoriser. The user's groups normally come from the `groups` claim of a validated token; the service takes them as a `Principal`.

## Layout

```
src/aclrag/
  models.py      Principal, Chunk, Answer
  security.py    group filter, fail-closed rules, filter expression
  corpus.py      ingestion: front matter, chunking, groups on every chunk
  retrieval.py   in-memory retriever, Azure AI Search retriever
  chat.py        the question flow, citations, audit, model adapters
  evaluate.py    persona evaluation and leak counting
  cli.py         aclrag ask / aclrag eval
samples/         corpus and questions
tests/
```

## License

MIT
