ForgeIQ ⚙️

A Retrieval-Augmented Generation (RAG) assistant for industrial maintenance knowledge

ForgeIQ is a local prototype of an industrial knowledge assistant. It
searches a small maintenance knowledge base, retrieves relevant
passages, and uses a Gemini model to draft an answer grounded in those
passages. It also tracks source references so users can inspect where an
answer came from.

The project explores how retrieval, structured citations, evaluation,
and a simple web interface can be combined into a practical AI
application.

Project status: Educational / portfolio prototype. The documents
and facility are fictional demonstration material. ForgeIQ is not a
validated industrial safety system and must not be used to make real
maintenance, repair, or safety-critical decisions.

Contents:

What ForgeIQ does

How it works

Knowledge base

Technology stack

Project structure

Getting started

Running ForgeIQ

Evaluating the system

Evaluation results so far

Limitations and safety

Troubleshooting

Future improvements

What ForgeIQ does

Semantic retrieval: searches maintenance passages by meaning
rather than relying only on exact keyword matches.

Grounded answer generation: supplies retrieved passages to a
Gemini model to help it answer a user's question.

Source traceability: assigns temporary source IDs to retrieved
passages and validates generated citation IDs against the passages
actually retrieved.

Insufficient-information handling: supports responses that
acknowledge when the available documents do not specify an answer.

Retrieval evaluation: measures whether expected source sections
appear among the top retrieved results.

Answer evaluation: records generated answers, retrieved
evidence, citation checks, and generation errors in a report.

Interactive web interface: provides a Streamlit chat UI for
asking questions of the knowledge base.

How it works

flowchart TD
    A[Industrial reference documents] --> B[Ingestion and section-aware chunking]
    B --> C[Text chunks with metadata]
    C --> D[Sentence Transformers embeddings]
    D --> E[ChromaDB vector store]
    F[User question] --> G[Semantic retrieval]
    E --> G
    G --> H[Top relevant passages]
    H --> I[Gemini answer generation]
    I --> J[Structured answer and source IDs]
    J --> K[Validate citation IDs]
    K --> L[Streamlit interface]

Ingest: load the text documents and split them into
section-aware chunks.

Index: embed the chunks and store them in ChromaDB with metadata
such as source document and section.

Retrieve: embed a user's question and retrieve relevant passages
from the vector store.

Generate: pass the question and retrieved context to Gemini,
requesting a structured response.

Validate: check that each generated source ID corresponds to a
passage retrieved for that question.

Display: show the answer and its source information in the app.

Important distinction: citation-ID validation confirms that a
citation points to a retrieved passage. It does not, by itself, prove
that the passage fully supports the claim or that the answer is
factually correct.

Knowledge base

The current demo contains six fictional reference documents under
data/raw/:

Document                            Purpose

maintenance_manual.txt            General maintenance guidance

troubleshooting_guide.txt         Troubleshooting symptoms and
investigation guidance

equipment_datasheet.txt           Demonstration equipment information

inspection_checklist.txt          Inspection items and safety
restrictions

maintenance_policy.txt            Maintenance process and escalation
guidance

The documents are for demonstrating the workflow, not representing
specifications for a real machine. In particular, do not infer missing
limits, causes, repairs, or manufacturer requirements.

Technology stack

Python --- application and evaluation logic

Streamlit --- interactive web application

ChromaDB --- vector database

Sentence Transformers --- text embeddings
(sentence-transformers/all-MiniLM-L6-v2)

Google Gemini API --- answer generation

python-dotenv --- loading local environment variables

pytest --- automated tests

Project structure

ForgeIQ/
├── app.py                         # Streamlit web application
├── .env                           # Local API key; do not commit
├── .gitignore
├── requirements.txt               # Python dependencies
├── data/
│   ├── raw/                       # Six source documents
│   └── processed/                 # Generated/intermediate data, if used
├── evaluation/
│   ├── evaluation_dataset.json     # Retrieval evaluation questions
│   ├── answer_evaluation_dataset.json
│   └── ...                        # Generated evaluation reports
├── logs/                          # Runtime/evaluation logs, if generated
├── src/
│   ├── ingest.py                  # Chunk and index source documents
│   ├── retriever.py               # Load vector store and retrieve passages
│   ├── ask_forgeiq.py             # Generate answers and validate citations
│   ├── evaluate_retrieval.py      # Retrieval evaluation
│   └── evaluate_answers.py        # Answer/citation evaluation
└── tests/
    └── test_source_validation.py  # Source-validation tests

Generated databases, virtual environments, API keys, and runtime
artifacts may be excluded from version control. If the vector database
is not present after cloning, rebuild it by running the ingestion step
below.

Getting started

1. Prerequisites

Python 3.11 or 3.12 is recommended for a straightforward setup.

A Google Gemini API key with access to the model configured in
src/ask_forgeiq.py.

Internet access for the Gemini API and the first download of the
embedding model.

The original development environment used Windows and a local Python
virtual environment. The commands below use PowerShell.

2. Get the project

If you already have the project folder, open PowerShell and navigate to
it:

cd C:\Users\Royal\ForgeIQ

If you are cloning a repository, replace the placeholder with your
repository URL:

git clone <YOUR_REPOSITORY_URL>
cd ForgeIQ

3. Create and activate a virtual environment

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

If Python 3.12 is not installed, use another supported Python version
that is compatible with the dependencies.

4. Install dependencies

Install the packages used by the application and evaluation scripts:

pip install streamlit chromadb sentence-transformers google-genai python-dotenv pandas pytest

If requirements.txt is populated and maintained with the project's
tested dependency versions, you can instead install with:

pip install -r requirements.txt

5. Configure the Gemini API key

Create a file named .env in the project root and add:

GEMINI_API_KEY=your_gemini_api_key_here

Replace the placeholder with your own key. Keep .env private and
never commit or publish it. Do not paste API keys into source files,
screenshots, issues, or public repositories.

The configured model name is defined in src/ask_forgeiq.py. If you
change it, make sure your API key and account have access to that model.

6. Build the vector database

From the project root, with the virtual environment activated:

python src/ingest.py

This step processes the documents in data/raw/, creates chunks and
embeddings, and builds the vector store used by retrieval. The first run
may take longer while the embedding model is downloaded.

If your version of ingest.py prints a completion summary, check that
the expected documents and chunks were processed successfully before
continuing.

Running ForgeIQ

Start the Streamlit app from the project root:

streamlit run app.py

Streamlit should print a local address, usually:

http://localhost:8501

Open that address in your browser. Keep the terminal open while using
the application. To stop the server, focus the terminal and press
Ctrl + C.

Suggested smoke tests

Try these questions in the app:

Answerable: What should be monitored in the hydraulic system?

Incomplete work-order record:
What corrective action was completed for work order FIQ-WO-003?

Information not specified:
What is the exact hydraulic pressure limit?

The expected behavior is to answer from available evidence, distinguish
documented facts from unconfirmed details, and acknowledge when the
knowledge base does not provide the requested information. Exact wording
may vary.

Evaluating the system

Run retrieval evaluation:

python src/evaluate_retrieval.py

Run answer and citation evaluation:

python src/evaluate_answers.py

The answer evaluation calls the Gemini API. It may consume API quota,
and some cases may fail if the account reaches its rate or daily limit.
Check the generated report and error details rather than treating an API
error as a successful test.

Evaluation datasets and reports are stored in evaluation/. Keep
generated reports separate from the hand-written test datasets when
reviewing or publishing results.

Metrics used

Hit Rate@3: the proportion of scored questions for which at
least one expected source appears among the top three retrieved
results.

Expected-source coverage@3: how much of the expected source set
is retrieved within the top three results.

Citation-ID validity: whether generated citation IDs refer to
passages retrieved for the same question.

Generation success: whether the model produced a usable response
without an execution/API error.

These metrics measure different things. Strong retrieval results do not
automatically prove answer correctness, and valid citation IDs do not
guarantee that a cited passage entails the answer.

Evaluation results so far

The current development baseline produced the following results:

Evaluation                                  Result Interpretation

Retrieval questions                             11 One unanswerable
scored                                             question was excluded
from source-retrieval
scoring

Hit Rate@3                                    100% At least one expected
source appeared in
the top three for
each scored question

Mean expected-source                         92.4% Most expected sources
coverage@3                                         were found in the top
three

Answer-evaluation                                6 A small initial test
cases attempted                                    set

Successful answer                           4 of 6 Two cases were
generations in the                                 blocked by Gemini
recorded run                                       free-tier quota
exhaustion

These are early baseline results from a small, hand-curated dataset,
not evidence of production readiness. The two quota-blocked answer cases
were not successfully tested in that run; in particular, the
safety-related case must not be described as passed until it has
actually been evaluated.

Limitations and safety

ForgeIQ is a learning and portfolio project with important limitations:

Fictional source material: documents and work orders are
synthetic demonstration data.

Small evaluation set: the current questions are limited and
manually curated; performance may differ on unseen questions.

Retrieval is not calibrated confidence: vector similarity and
retrieval rank should not be interpreted as probabilities that an
answer is correct.

Citation validation is not fact verification: an ID can be valid
even when the source does not fully support the associated claim.

Generative errors remain possible: the model may omit details,
misunderstand context, or produce unsupported statements.

API availability and quotas: generation depends on an external
API and can fail because of rate limits, quota exhaustion, network
errors, or service availability.

No real equipment validation: no real machine specifications,
operating limits, or maintenance outcomes should be inferred from
the demo.

Never use ForgeIQ as the sole basis for real maintenance, isolation,
repair, operating-limit, or safety decisions. Follow approved site
procedures, equipment documentation, and qualified personnel. In
particular, never loosen or handle pressurized hydraulic connections
based on a chatbot response.

Troubleshooting

streamlit is not recognized

Activate the virtual environment and use:

python -m streamlit run app.py

The app cannot find the vector database

Rebuild it from the project root:

python src/ingest.py

Check that ingestion completes successfully and that the path used by
src/retriever.py matches the database location.

Gemini reports a quota or 429 RESOURCE_EXHAUSTED error

This usually indicates that the API request was rate-limited or the
available quota was exhausted. Wait until quota becomes available, check
your account's current limits, or use a model/account configuration with
available quota. Avoid repeatedly rerunning the full answer evaluation
while quota is exhausted.

The first run takes a while

The embedding model may need to be downloaded and initialized. Later
runs are generally faster once the model and vector database are
available locally.

Citations appear valid, but an answer still seems wrong

Citation-ID validation only checks that a source ID maps to a retrieved
passage. Read the cited section and verify that it actually supports the
claim. This is one reason ForgeIQ needs broader evaluation and stronger
factual-grounding checks.

Future improvements

Potential next steps include:

Maintain a pinned, reproducible requirements.txt.

Add more answerability, paraphrase, safety, and adversarial
evaluation cases.

Evaluate semantic support between each answer claim and its cited
passage.

Add automated regression tests for retrieval,
insufficient-information behavior, and citation validation.

Track latency, token usage, API errors, and retrieval diagnostics.

Add document versioning and stronger metadata filters.

Improve observability and user-facing error handling.

Test against a larger, independently reviewed knowledge base before
making stronger reliability claims.

Author's note

ForgeIQ was built as a hands-on exploration of retrieval-augmented
generation, semantic search, vector databases, structured outputs,
citation traceability, evaluation, and application development. The goal
is not just to generate an answer, but to make the evidence behind that
answer inspectable.

ForgeIQ --- find the evidence before you trust the answer.