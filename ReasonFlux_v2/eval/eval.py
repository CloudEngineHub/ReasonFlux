
import argparse
import json
import os
import regex as re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union

from openai import OpenAI
from tqdm import tqdm

# ---------- Configuration ----------
DATASET_MAP = {
    "AIME24": "dataset/AIME24/aime24_nofigures.jsonl",
    "AIME25": "dataset/AIME25/aime25_nofigures.jsonl",
    "MATH500": "dataset/MATH500/test.jsonl",
}

PROMPT_EXTRACT = """
Please analyze the problem below and first give **Problem Abstraction and Generalization**,
then based on the abstraction and initial analysis, give **Special Conditions and Applicability Analysis**.
Finally give **High‑Level Solution Strategy**.
""".strip()

PROMPT_SOLVE = """
Please follow the High-Level Steps below to solve the problem step by step,
and give the final answer within \\boxed{{}}.
""".strip()

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['11', '100', '50', '-5', '12', '10']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

5

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK

Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass


def extract_boxed_answer(text: str) -> str:
    """
    Extract the last answer from \boxed{} expressions in the text.
    
    Args:
        text: Input text containing \boxed{} expressions
        
    Returns:
        The content of the last \boxed{} expression, or empty string if none found
    """
    if not text:
        return ""
        
    n = len(text)
    answers = []
    i = 0
    
    while i < n:
        # Find \boxed
        if text.startswith(r'\boxed', i):
            i += 6  # Skip \boxed
            # Skip whitespace
            while i < n and text[i].isspace():
                i += 1
            # Must be followed by {
            if i >= n or text[i] != '{':
                continue
            i += 1  # Enter content
            depth = 1
            start = i
            while i < n and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                # i points to the character after the matching '}'
                answers.append(text[start:i-1].strip())
        else:
            i += 1
    
    return answers[-1] if answers else ""


def load_dataset(benchmarks: List[str]) -> List[Dict]:
    """
    Load dataset from specified benchmarks.
    
    Args:
        benchmarks: List of benchmark names
        
    Returns:
        List of problem dictionaries with 'source' field added
    """
    data = []
    for bench in benchmarks:
        path = DATASET_MAP.get(bench)
        if not path:
            raise PipelineError(
                f"Unsupported benchmark {bench!r}; choose from {', '.join(DATASET_MAP)}")
        
        if not Path(path).exists():
            raise PipelineError(f"Dataset file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    item["source"] = bench
                    data.append(item)
                except json.JSONDecodeError as e:
                    tqdm.write(f"[WARN] Invalid JSON at line {line_num} in {path}: {e}")
    
    return data


def build_messages(data: List[Dict], 
                  sys_prompt: Union[str, Callable], 
                  user_fmt: Union[str, Callable]) -> List[List[Dict]]:
    """
    Build message lists for API calls.
    
    Args:
        data: List of data dictionaries
        sys_prompt: System prompt (str or callable(d) -> str)
        user_fmt: User message format (str or callable(d) -> str)
        
    Returns:
        List of message lists for API calls
    """
    def _eval(x, d):
        return x(d) if callable(x) else x

    return [
        [
            {"role": "system", "content": _eval(sys_prompt, d)},
            {"role": "user", "content": _eval(user_fmt, d)},
        ]
        for d in data
    ]


def call_model(client: OpenAI, 
               model_id: str, 
               message: List[Dict], 
               enable_thinking: bool = True,
               temperature: float = 0.7,
               max_tokens: int = 32768) -> Tuple[object, str]:
    """
    Make API call to the model.
    
    Args:
        client: OpenAI client instance
        model_id: Model identifier
        message: Message list for the API call
        enable_thinking: Whether to enable thinking mode
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (response object, user prompt string)
    """
    try:
        if enable_thinking:
            resp = client.chat.completions.create(
                model=model_id,
                messages=message,
                temperature=temperature,
                top_p=0.8,
                max_tokens=max_tokens,
                presence_penalty=1.5,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
        else:
            resp = client.chat.completions.create(
                model=model_id,
                messages=message,
                temperature=temperature,
                top_p=0.8,
                max_tokens=min(max_tokens, 16384),
                presence_penalty=1.5,
            )
        return resp, message[1]["content"]
    except Exception as e:
        raise PipelineError(f"API call failed: {e}")


def extract_template_steps(content: str) -> str:
    """
    Extract solution steps from template content.
    
    Args:
        content: Template content string
        
    Returns:
        Extracted steps or empty string if extraction fails
    """
    try:
        parts = content.split("High-Level Solution Strategy", 1)
        if len(parts) < 2:
            return ""
        steps_part = parts[1].strip()
        if "\n\n" in steps_part:
            return steps_part.split("\n\n", 1)[1]
        return steps_part
    except Exception:
        return ""


def phase_extract(template_model: str,
                  data: List[Dict],
                  threads: int,
                  client: OpenAI,
                  max_retry: int = 5,
                  reasoning: bool = True) -> List[Dict]:
    """
    Phase 1: Extract templates from problems with retry mechanism.
    
    Args:
        template_model: Model ID for template extraction
        data: List of problem dictionaries
        threads: Number of concurrent threads
        client: OpenAI client instance
        max_retry: Maximum number of retry attempts
        reasoning: Whether to enable reasoning mode
        
    Returns:
        List of successfully extracted template records
    """
    remaining = data[:]  # Copy of items to process
    records = []  # Successfully processed samples
    last_reasoning_content = {}  # Store last reasoning content for fallback
    
    for attempt in range(1, max_retry + 1):
        if not remaining:
            break

        tqdm.write(f"🔁 Extract attempt {attempt}/{max_retry} "
                   f"(remaining {len(remaining)})")

        messages = build_messages(
            remaining,
            sys_prompt=PROMPT_EXTRACT,
            user_fmt=lambda d: d["problem"])

        failed = []
        with ThreadPoolExecutor(max_workers=threads) as pool, \
             tqdm(total=len(messages),
                  desc=f"Phase-1 Extracting [try {attempt}]") as pbar:

            futures = {pool.submit(call_model, client, template_model, m, reasoning): src
                       for m, src in zip(messages, remaining)}

            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    resp, _ = fut.result()
                except Exception as e:
                    tqdm.write(f"[WARN] Extract error: {e}")
                    failed.append(src)
                    pbar.update(1)
                    continue

                msg = resp.choices[0].message
                template_content = (msg.content or "").strip()
                reasoning_content = (msg.reasoning_content or "").strip() if reasoning else "No reasoning content."
                
                # Store latest reasoning content for fallback
                last_reasoning_content[src["problem"]] = reasoning_content

                if template_content:
                    # Success case
                    steps = extract_template_steps(template_content)
                    if steps:  # Only add if we successfully extracted steps
                        records.append({
                            "problem": src["problem"],
                            "answer": src.get("answer"),
                            "source": src["source"],
                            "thinking": reasoning_content,
                            "template": template_content,
                            "steps": steps,
                        })
                    else:
                        failed.append(src)
                else:
                    # Empty content - retry
                    failed.append(src)
                pbar.update(1)

        remaining = failed  # Next round processes only failed samples

    # Final fallback using reasoning content
    if remaining:
        tqdm.write(f"[INFO] Attempting fallback for {len(remaining)} remaining samples")
        for src in remaining:
            reasoning_content = last_reasoning_content.get(src["problem"], "").strip()
            
            if reasoning_content:
                steps = extract_template_steps(reasoning_content)
                if steps:
                    records.append({
                        "problem": src["problem"],
                        "answer": src.get("answer"),
                        "source": src["source"],
                        "thinking": reasoning_content,
                        "template": reasoning_content,  # Use reasoning as template
                        "steps": steps,
                    })
                else:
                    tqdm.write(f"[WARN] Sample '{src['problem'][:50]}...' "
                             f"failed after {max_retry} retries; skipped.")
            else:
                tqdm.write(f"[WARN] Sample '{src['problem'][:50]}...' "
                           f"has no reasoning content; skipped.")

    return records


def phase_solve(solver_model: str,
                template_records: List[Dict],
                threads: int,
                client: OpenAI,
                max_retries: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """
    Phase 2: Solve problems using extracted templates with retry mechanism.
    
    Args:
        solver_model: Model ID for problem solving
        template_records: List of template records from phase 1
        threads: Number of concurrent threads
        client: OpenAI client instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (successful results, failed samples)
    """
    if not template_records:
        return [], []

    all_results: List[Dict] = []
    pending: List[Dict] = template_records[:]  # Samples still to solve
    retry = 0

    while pending and retry < max_retries:
        retry += 1
        tqdm.write(f"[INFO] Solve round {retry}/{max_retries} — {len(pending)} samples")

        # Build messages only for pending samples
        messages = build_messages(
            pending,
            sys_prompt=PROMPT_SOLVE,
            user_fmt=lambda d: f"Problem:\n{d['problem']}\nHigh-Level Steps:\n{d['steps']}",
        )

        failed_this_round: List[Dict] = []

        with ThreadPoolExecutor(max_workers=threads) as pool, \
             tqdm(total=len(messages), desc=f"Phase-2 Solving (round {retry})") as pbar:

            futures = {
                pool.submit(call_model, client, solver_model, m, True): rec
                for m, rec in zip(messages, pending)
            }

            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    resp, _ = fut.result()
                    msg = resp.choices[0].message

                    # Treat empty content as failure
                    if not msg or not getattr(msg, "content", "").strip():
                        raise ValueError("Empty response content")

                    extracted_answer = extract_boxed_answer(msg.content)

                    all_results.append({
                        "problem": rec["problem"],
                        "thought_template": rec["template"],
                        "solver_thinking": getattr(msg, "reasoning_content", ""),
                        "solver_response": msg.content,
                        "gt_answer": rec.get("answer"),
                        "extracted_answer": extracted_answer,
                        "source": rec["source"],
                    })

                except Exception as e:
                    tqdm.write(f"[WARN] Solve failed: {e}")
                    failed_this_round.append(rec)

                finally:
                    pbar.update(1)

        pending = failed_this_round  # Re-queue only failed samples

    # Log final failures
    if pending:
        tqdm.write(f"[INFO] {len(pending)} samples still failed after {max_retries} retries")

    return all_results, pending


def phase_judge(judge_model: str,
                sample_records: List[Dict],
                threads: int,
                client: OpenAI) -> List[Dict]:
    """
    Phase 3: Judge the correctness of solutions using LLM-as-a-Judge.
    
    Args:
        judge_model: Model ID for judging
        sample_records: List of sample records from phase 2
        threads: Number of concurrent threads
        client: OpenAI client instance
        
    Returns:
        List of judged records with correctness scores
    """
    def build_judge_prompt(rec):
        """Build judge prompt for a single record"""
        opts = rec["gt_answer"]
        # Ensure options are in list literal string format
        if isinstance(opts, list):
            opts_str = str(opts)
        else:
            opts_str = f"['{opts}']"
        return EXTRACTION_TEMPLATE_IDX % {
            "expression1": opts_str,
            "expression2": rec["extracted_answer"]
        }

    messages = build_messages(
        sample_records,
        sys_prompt=lambda d: build_judge_prompt(d),  # System prompt contains entire template
        user_fmt=lambda d: ""  # Judge prompt only in system, user empty
    )

    judged = []
    with ThreadPoolExecutor(max_workers=threads) as pool, \
         tqdm(total=len(messages), desc="Phase-3 Judging") as pbar:
        
        futures = {pool.submit(call_model, client, judge_model, m, False): rec
                   for m, rec in zip(messages, sample_records)}

        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                resp, _ = fut.result()
                output = (resp.choices[0].message.content or "").strip()
                
                # Extract numeric score
                try:
                    match = re.match(r"-?\d+", output)
                    score = int(match.group()) if match else -1
                except Exception:
                    score = -1
                
                rec["judge_score"] = output
                rec["correct"] = 1 if score != -1 else -1
                
            except Exception as e:
                tqdm.write(f"[WARN] Judge failed: {e}")
                rec["judge_score"] = "ERROR"
                rec["correct"] = -1
            
            judged.append(rec)
            pbar.update(1)
    
    return judged


def write_accuracy_log(records: List[Dict], template_model: str, solver_model: str):
    """
    Write accuracy statistics to log file.
    
    Args:
        records: List of judged records
        template_model: Template model identifier
        solver_model: Solver model identifier
    """
    # Group by source benchmark
    grouped = defaultdict(list)
    for r in records:
        grouped[r["source"]].append(r)

    lines = []
    for bench, items in grouped.items():
        correct_cnt = sum(1 for r in items if r["correct"] == 1)
        acc = correct_cnt / len(items) if items else 0
        lines.append(f"{bench}: {acc:.4%} ({correct_cnt}/{len(items)})")

    # Overall accuracy
    overall_correct = sum(1 for r in records if r["correct"] == 1)
    overall_acc = overall_correct / len(records) if records else 0
    lines.append(f"OVERALL: {overall_acc:.4%} ({overall_correct}/{len(records)})")

    # Write to log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{template_model.replace('/', '_')}_{solver_model.replace('/', '_')}_accuracy.log"
    
    with log_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(["\n📊 Accuracy"] + lines))
    print(f"\nℹ️ Log written to {log_file}")


def save_grouped(records: List[Dict],
                root: Path,
                use_timestamp: bool = False,
                ts_format: str = "%Y%m%d-%H%M%S") -> set:
    """
    Save records grouped by source to JSON files.
    
    Args:
        records: List of records with 'source' field
        root: Root directory for saving files
        use_timestamp: Whether to append timestamp to filenames
        ts_format: Timestamp format string
        
    Returns:
        Set of benchmark names that were saved
    """
    root.mkdir(parents=True, exist_ok=True)

    # Group by source
    grouped = defaultdict(list)
    for r in records:
        grouped[r["source"]].append(r)

    # Generate timestamp suffix if needed
    ts_suffix = datetime.now().strftime(ts_format) if use_timestamp else ""

    # Write files
    for bench, items in grouped.items():
        if use_timestamp:
            out_path = root / ts_suffix / f"{bench}.json"
            out_path.parent.mkdir(exist_ok=True)
        else:
            out_path = root / f"{bench}.json"
            
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

    return set(grouped.keys())


def check_existing_templates(root: Path, benchmarks: List[str]) -> List[Dict]:
    """
    Check if templates already exist for given benchmarks and load them.
    
    Args:
        root: Root directory containing template files
        benchmarks: List of benchmark names to check
        
    Returns:
        List of existing template records, or empty list if any are missing
    """
    if not root.exists():
        return []
        
    try:
        existing_files = {f.stem for f in root.glob("*.json")}
        missing_benchmarks = set(benchmarks) - existing_files
        
        if missing_benchmarks:
            print(f"⚠️ Templates missing for: {', '.join(missing_benchmarks)}")
            return []
        
        # Load all existing templates
        print(f"✅ Loading existing templates for: {', '.join(benchmarks)}")
        templates = []
        for bench in benchmarks:
            template_file = root / f"{bench}.json"
            with template_file.open("r", encoding="utf-8") as f:
                items = json.load(f)
                templates.extend(items)
        
        return templates
    except Exception as e:
        tqdm.write(f"[WARN] Error loading existing templates: {e}")
        return []


def run_pipeline(template_model: str,
                solver_model: str,
                benchmarks: List[str],
                threads: int,
                base_url: str,
                reasoner_url: str,
                max_retry: int,
                reasoning: bool = True,
                force_template: bool = False):
    """
    Run the complete pipeline: extract templates, solve problems, and judge results.
    
    Args:
        template_model: Model ID for template extraction
        solver_model: Model ID for problem solving
        benchmarks: List of benchmark names to process
        threads: Number of concurrent threads
        base_url: API base URL for template extraction
        reasoner_url: API base URL for problem solving
        max_retry: Maximum retry attempts for template extraction
        reasoning: Whether to enable reasoning mode
        force_template: Whether to force template re-extraction
    """
    # Load dataset
    data = load_dataset(benchmarks)
    print(f"📊 Loaded {len(data)} problems from {len(benchmarks)} benchmarks")

    # Phase 1: Template extraction
    template_client = OpenAI(api_key="EMPTY", base_url=base_url)
    template_root = Path("template") / template_model.replace("/", "_")
    
    if not force_template:
        existing_templates = check_existing_templates(template_root, benchmarks)
        if existing_templates:
            templates = existing_templates
            tqdm.write(f"✅ Loaded {len(templates)} existing templates from {template_root}")
        else:
            tqdm.write(f"🔁 Extracting templates for {benchmarks}")
            templates = phase_extract(template_model, data, threads, template_client, max_retry, reasoning)
            save_grouped(templates, template_root)
            tqdm.write(f"✅ Extracted and saved {len(templates)} templates to {template_root}")
    else:
        print('⚠️ Force template extraction enabled, ignoring existing templates.')
        tqdm.write(f"🔁 Extracting templates for {benchmarks}")
        templates = phase_extract(template_model, data, threads, template_client, max_retry, reasoning)
        save_grouped(templates, template_root)
        tqdm.write(f"✅ Extracted and saved {len(templates)} templates to {template_root}")

    # Phase 2: Problem solving
    solver_client = OpenAI(api_key="EMPTY", base_url=reasoner_url)
    sample_root = Path("sample") / solver_model.replace("/", "_")
    
    samples, failed_samples = phase_solve(solver_model, templates, threads, solver_client)
    save_grouped(samples, sample_root)
    tqdm.write(f"✅ Solved {len(samples)} problems, saved to {sample_root}")
    
    if failed_samples:
        tqdm.write(f"⚠️ {len(failed_samples)} problems failed to solve")

    # Phase 3: Judging
    if samples:
        judged = phase_judge(solver_model, samples, threads, solver_client)
        save_grouped(judged, sample_root)  # Overwrite with judged results
        write_accuracy_log(judged, template_model, solver_model)
    else:
        print("❌ No samples to judge")

    print(f"\n✅ Pipeline completed! Processed benchmarks: {', '.join(benchmarks)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract templates then solve problems with the templates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--benchmark", "-b", default="AIME24",
                       help="Datasets, e.g. AIME24 or AIME24,MATH500")
    parser.add_argument("--template-model", "-tm",
                       default="/data_storage/yzc/LLaMA-Factory/saves/Qwen3-32B-Instruct/full/proposer_dpo",
                       help="Model ID for template extraction")
    parser.add_argument("--solver-model", "-sm", default="Qwen/Qwen3-32B",
                       help="Model ID for downstream solving")
    parser.add_argument("--threads", "-t", type=int, default=32,
                       help="Concurrent request pool size")
    parser.add_argument("--base-url", default="http://localhost:30000/v1",
                       help="OpenAI-compatible API base URL for template extraction")
    parser.add_argument('--reasoner-url', default="http://localhost:30001/v1",
                       help="OpenAI-compatible API base URL for problem solving")
    parser.add_argument("--max-retry", type=int, default=3,
                       help="Max retries when extracted template is empty/null")
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode for template extraction")
    parser.add_argument("--force-template", action="store_true",
                       help="Force template re-extraction even if templates exist")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmarks = [b.strip().upper() for b in args.benchmark.split(",") if b.strip()]
    
    try:
        run_pipeline(
            template_model=args.template_model,
            solver_model=args.solver_model,
            benchmarks=benchmarks,
            threads=args.threads,
            base_url=args.base_url,
            reasoner_url=args.reasoner_url,
            max_retry=args.max_retry,
            reasoning=args.reasoning,
            force_template=args.force_template
        )
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise