"""
Diagnostic script: verify that Ollama is installed, running, and can serve
structured responses compatible with the evacuation-simulation agent pipeline.

Usage:
    conda run -n cs294 python -m agent.check_ollama
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _header(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def check_binary() -> bool:
    """1. Is the `ollama` CLI on PATH?"""
    _header("1. Ollama binary")
    path = shutil.which("ollama")
    if path is None:
        print("FAIL  -- `ollama` not found on PATH.")
        print("       Install it: https://ollama.com/download")
        return False

    version = subprocess.run(
        ["ollama", "--version"],
        capture_output=True, text=True,
    )
    ver_text = (version.stdout + version.stderr).strip()
    print(f"OK    -- {path}")
    print(f"       {ver_text}")
    return True


def check_server() -> bool:
    """2. Is the Ollama server reachable?"""
    _header("2. Ollama server")
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        print("OK    -- Server responding on localhost:11434")
        return True
    except Exception as exc:
        print(f"FAIL  -- Cannot reach Ollama server: {exc}")
        print("       Start it with:  ollama serve")
        return False


def check_models() -> str | None:
    """3. Are any models pulled?  Return the first available model name."""
    _header("3. Available models")
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        data = resp.json()
        models = data.get("models", [])
        if not models:
            print("FAIL  -- No models pulled.")
            print("       Pull one with:  ollama pull llama3.2")
            return None

        for m in models:
            name = m.get("name", m.get("model", "?"))
            size_bytes = m.get("size", 0)
            size_gb = size_bytes / 1e9
            print(f"  {name:30s}  ({size_gb:.1f} GB)")

        chosen = models[0].get("name", models[0].get("model"))
        print(f"\nOK    -- {len(models)} model(s) available.  Using '{chosen}' for test.")
        return chosen
    except Exception as exc:
        print(f"FAIL  -- Could not list models: {exc}")
        return None


def check_langchain_import() -> bool:
    """4. Can we import langchain-ollama?"""
    _header("4. Python package: langchain-ollama")
    try:
        import langchain_ollama  # noqa: F401
        print(f"OK    -- langchain-ollama {langchain_ollama.__version__}")
        return True
    except ImportError:
        print("FAIL  -- langchain-ollama not installed.")
        print("       Install it:  pip install langchain-ollama")
        return False


def check_structured_output(model_name: str) -> bool:
    """5. End-to-end test: send a prompt and parse structured JSON output."""
    _header("5. Structured-output test")
    from pydantic import BaseModel, Field
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    class TestOutput(BaseModel):
        greeting: str = Field(description="A short greeting")
        number: int = Field(description="Any integer between 1 and 100")

    llm = ChatOllama(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(TestOutput, include_raw=True)

    print(f"       Sending test prompt to '{model_name}' ...")
    try:
        result = structured_llm.invoke(
            [HumanMessage(content="Say hello and pick a number between 1 and 100.")]
        )
        parsed: TestOutput = result["parsed"]
        raw_msg = result["raw"]

        print(f"OK    -- Received valid structured output:")
        print(f"         greeting = {parsed.greeting!r}")
        print(f"         number   = {parsed.number}")

        meta = getattr(raw_msg, "response_metadata", {})
        if meta:
            eval_count = meta.get("eval_count", "?")
            eval_duration_ns = meta.get("eval_duration", 0)
            eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else "?"
            print(f"         tokens generated = {eval_count}")
            print(f"         generation time  = {eval_duration_s}s" if isinstance(eval_duration_s, float) else "")
        return True

    except Exception as exc:
        print(f"FAIL  -- Structured output failed: {exc}")
        return False


def main() -> None:
    print("Ollama + LangChain readiness check for evacuation-sim agent")

    results = {}

    results["binary"] = check_binary()
    if not results["binary"]:
        _summary(results)
        return

    results["server"] = check_server()
    if not results["server"]:
        _summary(results)
        return

    model_name = check_models()
    results["models"] = model_name is not None

    results["langchain"] = check_langchain_import()

    if model_name and results["langchain"]:
        results["structured"] = check_structured_output(model_name)
    else:
        results["structured"] = False

    _summary(results)


def _summary(results: dict) -> None:
    _header("Summary")
    all_ok = all(results.values())
    for key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {key}")

    print()
    if all_ok:
        print("All checks passed. Ollama is ready for use with the agent pipeline.")
    else:
        print("Some checks failed. See details above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
