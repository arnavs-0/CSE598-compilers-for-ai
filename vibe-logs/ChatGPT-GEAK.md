# GEAK purpose overview

**User:** Arnav Shah (sarn248@gmail.com)  
**Created:** 9/30/2025 18:22:39  
**Updated:** 9/30/2025 18:28:22  
**Exported:** 9/30/2025 18:28:44  

## Prompt:
2507.23194v1 (1).pdf

PDF

Walk me through geak what's the purpose of this

## Response:
Got it — I walked through the paper you uploaded. Here’s a clear breakdown of **GEAK** and its purpose:

### What GEAK Is

GEAK stands for **Generating Efficient AI-centric GPU Kernels**.  
It’s an **agentic framework built by AMD** that uses large language models (LLMs) to automatically generate optimized **Triton GPU kernels** (small GPU programs). Triton is a Python-based DSL (domain-specific language) that makes writing GPU kernels easier than CUDA or HIP, but still requires expert tuning for performance.

* * *

### Why It Exists (Purpose)

1.  **Rising demand for GPU kernels**
    *   AI workloads are growing more complex and diverse.
    *   Manually writing and optimizing GPU kernels is **time-consuming and requires expertise**.
    *   Automating kernel generation can save engineers time and accelerate deployment.
2.  **Hardware-specific optimization**
    *   Most tools/benchmarks are CUDA/NVIDIA centric.
    *   AMD wants high-performance support for its **Instinct MI300X/MI250 GPUs**.
    *   GEAK helps democratize access to efficient kernels on AMD hardware.
3.  **Better than naïve LLM prompting**
    *   Plain prompting of GPT-4 or Gemini struggles with correctness (<15% pass rate).
    *   GEAK, with its **multi-agent reasoning loop (generate → evaluate → reflect → optimize)**, improves correctness (~55–63%) and speedups (up to **2.59× faster** than reference kernels).
4.  **Benchmarks & reproducibility**
    *   They release **two benchmarks**:
        *   **TritonBench-revised**: 184 kernels adapted for AMD GPUs.
        *   **ROCm Triton Benchmark**: 30 real-world kernels from AMD repos.
    *   This allows the community to measure both correctness and performance systematically.

* * *

### How GEAK Works

It’s structured as a **pipeline of agents**:

*   **Generator**: Writes initial Triton code from a task description.
*   **Evaluator**: Tests correctness and performance.
*   **Reflector**: Debugs failed kernels using error traces.
*   **Optimizer**: Improves latency and efficiency of correct kernels.

They also add techniques like:

*   **One-shot prompting** (give an example similar kernel)
*   **Knowledge injection** (hardware-specific hints, best practices)
*   **Reflexion-style feedback** (iterative debugging)
*   **Parallel & sequential scaling** (multiple runs to increase success rates)

* * *

### In One Line

GEAK’s purpose is to **automate and improve the generation of high-performance GPU kernels in Triton**, especially for AMD GPUs, by using an agentic LLM framework with benchmarking to validate both correctness and speed.

* * *

Do you want me to also make a **visual flow diagram** of the GEAK pipeline (Generator → Evaluator → Reflector → Optimizer) so it’s easier to see how the pieces fit together?

## Prompt:
what is the point of one-shot prompting and knowldege injectionn

## Response:
Good question — those two are **“boosters”** for GEAK’s LLM agent. Let’s unpack them:

* * *

### 🔹 One-Shot Prompting

*   **What it is:**  
    Instead of giving the LLM just the problem statement, GEAK also provides **one example kernel** (retrieved from a dataset of Triton code that isn’t part of the benchmark).
*   **Why it matters:**
    *   LLMs are much better at pattern matching than inventing optimized kernels from scratch.
    *   By seeing one working kernel, the LLM can mimic **style, structure, and API usage**.
    *   This improves both **call accuracy** (compiling without errors) and **execution accuracy** (passing unit tests).
*   **Analogy:** It’s like giving a student a worked-out physics example before asking them to solve a new but related problem.

* * *

### 🔹 Knowledge Injection

*   **What it is:**  
    Adding **domain-specific optimization hints** to the prompt. This can include:
    *   Best practices for Triton kernel design.
    *   Hardware details (e.g., AMD MI300X memory hierarchy, register usage).
    *   Tricks for coalesced memory access or reducing register pressure.
*   **Why it matters:**
    *   LLMs trained broadly don’t “know” the fine-grained optimization tricks for AMD GPUs.
    *   Injecting this knowledge **guides the model toward expert-like solutions**.
    *   The paper shows it significantly improves correctness and kernel quality even by itself.
*   **Analogy:** It’s like giving the student a “cheat sheet” of formulas and optimization rules before they attempt the exam.

* * *

### 🔑 Together

*   **One-shot prompting** helps the model **imitate structure**.
*   **Knowledge injection** helps the model **apply domain expertise**.

Both are complementary — one addresses _format and syntax_, the other addresses _performance-critical knowledge_.

* * *

Do you want me to show you a **before vs. after example** of a kernel where one-shot prompting and knowledge injection make the LLM’s output go from broken/naïve to working/optimized?

## Prompt:
so why does geak need this?

## Response:
Great follow-up 👌 — the reason **GEAK needs one-shot prompting and knowledge injection** comes down to the **limitations of LLMs** and the **difficulty of GPU kernel code generation**.

* * *

### 1\. Raw LLMs Struggle with Low-Level Code

*   If you just ask GPT-4 or Gemini to “write a Triton kernel for X,” it often:
    *   Fails to compile (syntax/API errors).
    *   Misses hardware-specific details (memory alignment, thread mapping).
    *   Produces correct-looking but **inefficient** code.
*   In GEAK’s baseline tests, direct prompting had **<15% correctness** on benchmarks
    2507.23194v1 (1)
    .
*   That’s way too low for real deployment.

* * *

### 2\. One-Shot Prompting Helps “Anchor” the Model

*   Triton is niche, and LLMs haven’t seen as much training data on it compared to, say, Python.
*   A single example reminds the model:
    *   “Here’s how a working kernel is structured.”
    *   “Here’s how you use Triton APIs properly.”
*   This **boosts success rate** by giving the LLM a **scaffold** to copy/adapt from.

* * *

### 3\. Knowledge Injection Fixes the Blind Spots

*   Even if an LLM writes syntactically correct Triton, it doesn’t “know” AMD GPU quirks:
    *   How to coalesce memory accesses.
    *   Avoiding register spilling.
    *   Latency/memory trade-offs specific to MI250/MI300.
*   Knowledge injection feeds the LLM optimization “tricks” and **hardware context** it otherwise lacks.

* * *

### 4\. Why GEAK Combines Them

GEAK is aiming for **expert-level performance automatically**, which means:

*   It can’t rely on brute-force LLM outputs (too unreliable).
*   It needs **one-shot prompting** to overcome structure/syntax failures.
*   It needs **knowledge injection** to overcome performance blind spots.
*   Together, they turn a general-purpose LLM into a **specialized code-generation agent** that works reliably for AMD GPUs.

* * *

⚡ **In short:**  
GEAK needs these because LLMs alone are **not yet good enough** to write correct _and_ performant GPU kernels. One-shot prompting fixes _formatting/structural weaknesses_, while knowledge injection fixes _domain knowledge gaps_. Without them, GEAK wouldn’t get anywhere close to its ~55–63% correctness and 2.6× speedups.

* * *

Do you want me to make a **simple flow diagram** showing how GEAK + these modules lift performance compared to raw LLM prompting?

## Prompt:
Walk me through how Geak works, its technical impl

## Response:
Perfect — let’s walk step by step through **how GEAK works technically**. Think of it as an **agent pipeline** wrapped around an LLM, designed to take a kernel request and iteratively improve it until it’s correct _and_ fast.

* * *

🔹 GEAK Pipeline (High-Level)
-----------------------------

The pipeline has **four modules (agents)**:

1.  **Generator** – writes candidate Triton kernel code.
2.  **Evaluator** – tests correctness and measures performance.
3.  **Reflector** – analyzes failures/errors and proposes fixes.
4.  **Optimizer** – takes correct but slow code and refines it.

Each step loops until either a correct + performant kernel is found, or a limit is reached.

* * *

🔹 Step-by-Step Technical Flow
------------------------------

### 1\. Input → Generator

*   Input can be:
    *   A **task description** (e.g., “implement softmax in Triton”), or
    *   **Reference code** (e.g., PyTorch function to convert).
*   The **Generator LLM** is prompted with:
    *   Task description.
    *   **One-shot example** (retrieved from a code database).
    *   **Knowledge injection** (hardware hints, optimization guidelines).
*   Output: Candidate Triton kernel code.

* * *

### 2\. Evaluator

*   Runs the generated kernel on actual AMD GPUs (MI250/MI300).
*   Two checks:
    1.  **Functionality test** → does it compile & pass unit tests?
    2.  **Performance test** → latency, memory usage, speedup vs. reference.

If it fails correctness → go to **Reflector**.  
If correct but slow → go to **Optimizer**.

* * *

### 3\. Reflector (Debugging & Self-Correction)

*   Takes error traces (e.g., “invalid HIP args” or runtime failure).
*   LLM analyzes:
    *   Why did it fail? (bad indexing, wrong Triton API, shape mismatch).
    *   Proposes a fix and regenerates code.
*   This process loops → GEAK calls it **Reflexion-style debugging**.
*   To avoid “infinite loops,” they cap retries (`max_perf_debug_num`).

* * *

### 4\. Optimizer (Performance Tuning)

*   For kernels that _work but are slow_:
    *   Optimizer LLM reviews past runs + performance logs.
    *   Suggests **new strategies**: better block size, coalesced loads, reduced register pressure, etc.
    *   Rewrites the kernel with these optimizations.
*   Example: Their flip kernel case study showed GEAK halved memory bandwidth usage compared to the expert’s original.

* * *

🔹 Scaling Techniques
---------------------

GEAK isn’t just one run → it **scales inference-time compute** in two ways:

1.  **Sequential Scaling**
    *   Run multiple refinement iterations.
    *   Accuracy improved from ~13% → ~44% after 19 iterations.
    *   Each loop is “try → test → reflect/optimize → retry.”
2.  **Parallel Scaling**
    *   Run multiple generations in parallel with random sampling (temperature=1).
    *   Increases diversity, raises chance one kernel is both correct & fast.
    *   Accuracy improves log-linearly with number of parallel runs.

Combined scaling = much higher robustness than single-shot LLM prompting.

* * *

🔹 Technical Modules Recap
--------------------------

*   **One-shot prompting** → helps generator mimic working Triton syntax.
*   **Knowledge injection** → ensures kernels use AMD-specific optimizations.
*   **Reflexion loop** → makes it self-correcting when kernels fail.
*   **Optimizer** → ensures final code is not just _correct_, but _fast_.

* * *

🔹 Results (Why It Works)
-------------------------

*   **Direct LLM prompting:** <15% correct, usually inefficient.
*   **GEAK with modules:**
    *   ~55% execution accuracy (TritonBench-revised).
    *   ~63% execution accuracy (ROCm benchmark).
    *   Speedups up to **2.59×** over reference kernels
        2507.23194v1 (1)
        .

* * *

⚡ **In one line:**  
GEAK works by wrapping LLM code generation in an **agentic feedback loop (generate → evaluate → reflect → optimize)**, enhanced with one-shot prompting, knowledge injection, and inference-time scaling, to reliably produce **correct and performant GPU kernels**.

* * *

Want me to **sketch a diagram** (Generator → Evaluator → Reflector → Optimizer loop, with sequential + parallel scaling shown) so you can see the technical flow visually?

## Prompt:
but I thought geak is just a test bench right?

## Response:
Ah — that’s an easy place to get confused 👍

GEAK is **not just a test bench**. It actually has **two parts**:

* * *

### 1\. **GEAK (the agent framework)**

*   This is the **core system** AMD built.
*   It’s the **multi-agent pipeline** (Generator → Evaluator → Reflector → Optimizer).
*   Purpose: **automatically generate performant Triton GPU kernels** from task descriptions.

* * *

### 2\. **GEAK Evaluation Benchmarks**

*   Alongside the agent, they released **two test suites**:
    *   **TritonBench-revised** → 184 kernels, cleaned up for AMD GPUs.
    *   **ROCm Triton Benchmark** → 30 real-world AMD kernels.
*   Purpose: give a **standardized way to measure** correctness + speedup of generated kernels.

* * *

### ⚡ So the point is:

*   **GEAK (agent)** = the _system that generates kernels_.
*   **Benchmarks** = the _datasets/tests to evaluate how well it works_.

They ship both together, but GEAK itself is more than a benchmark — it’s an **AI-driven codegen framework** that _uses_ the benchmarks to prove its performance.

* * *

Would it help if I broke down **GEAK vs. Benchmarks** in a comparison table so it’s super clear what does what?

## Prompt:
so why is it not popular

## Response:
Great question — if GEAK is promising, why isn’t it “popular” yet? There are a few reasons, most of them practical rather than technical:

* * *

### 1\. **Very New**

*   The paper was just published (July 2025, per the arXiv date
    2507.23194v1 (1)
    ).
*   Adoption takes time — especially in the **GPU developer ecosystem**, where CUDA has dominated for years.

* * *

### 2\. **Niche Audience**

*   GEAK is focused on **Triton + AMD GPUs**.
*   Most researchers and engineers are still working in **CUDA/NVIDIA ecosystems**, so the potential user base is smaller.
*   Even within AMD users, **Triton adoption is relatively new** (vs. PyTorch/CUDA kernels).

* * *

### 3\. **Complex Setup**

*   To really use GEAK, you need:
    *   **AMD hardware** (MI250, MI300).
    *   **ROCm ecosystem** installed (which has a learning curve).
    *   Ability to run **large LLMs** with inference-time scaling (not trivial for small labs).
*   That makes it less accessible than a “plug-and-play” benchmark like HumanEval.

* * *

### 4\. **Benchmarks vs. Production**

*   GEAK shows strong results in benchmarks, but **production engineers want stability, transparency, and guarantees**.
*   AI-generated kernels are still somewhat unpredictable, so adoption in production systems is slow.

* * *

### 5\. **Competition**

*   NVIDIA’s CUDA ecosystem + existing kernel libraries (cuBLAS, cuDNN, etc.) are **mature and heavily optimized**.
*   Developers often trust those more than an experimental LLM-based generator.
*   Until GEAK proves it can consistently beat hand-tuned libraries across workloads, adoption will stay limited.

* * *

### ⚡ Bottom Line

GEAK isn’t “popular” yet because it’s **new, niche, and specialized to AMD Triton kernels**, while the broader AI/ML world is still NVIDIA-centric. It’s mainly a **research prototype** right now — but AMD likely hopes it grows into a tool that lowers the barrier to writing high-performance kernels.

* * *

Do you want me to give you a **realistic outlook** (like: what would need to happen for GEAK to actually become mainstream)?

## Prompt:
but doesnt openai use triton? Why arent more people adopting it - is it just because of cuda?

## Response:
You’re spot on — **OpenAI does use Triton** (they open-sourced it in 2021, and it’s used internally to write custom GPU kernels for models like GPT-4). So why isn’t it everywhere?

Here’s the breakdown:

* * *

🔹 Why Triton is Powerful
-------------------------

*   Lets you write GPU kernels in **Python-like code** instead of raw CUDA/HIP.
*   Automatically handles some low-level details (tiling, vectorization).
*   Can sometimes **match or beat expert-written CUDA kernels** with much less code.

So in theory, Triton should be the future.

* * *

🔹 Why It’s Not More Widely Adopted (Yet)
-----------------------------------------

### 1\. **CUDA Ecosystem Lock-in**

*   CUDA has been around since 2007.
*   Entire ML stack (PyTorch, TensorFlow, cuDNN, cuBLAS, NCCL) is **deeply CUDA-integrated**.
*   Most production ML engineers know CUDA, not Triton.
*   Tooling, debugging, and docs are still **way better** on CUDA.

### 2\. **Triton Is Young**

*   OpenAI’s Triton only became practical in the last ~3–4 years.
*   Ecosystem (tutorials, libraries, benchmarks) is still catching up.
*   Outside OpenAI, only a handful of orgs (Meta, AMD, some HPC labs) are experimenting with it seriously.

### 3\. **Performance Isn’t Always Guaranteed**

*   While Triton can beat CUDA in some cases, **hand-tuned CUDA kernels** (cuBLAS, cuDNN) are extremely optimized.
*   For bread-and-butter ops (GEMMs, convolutions), Triton usually can’t beat NVIDIA’s libraries.
*   So Triton shines mainly for **niche/custom ops** where CUDA libraries don’t exist.

### 4\. **Hardware Support**

*   Triton was first built **only for NVIDIA GPUs**.
*   AMD support (via ROCm) is just emerging — that’s why AMD is pushing frameworks like **GEAK** to make Triton usable on their hardware.
*   Until AMD gains more ML market share, adoption will lag.

### 5\. **Developer Mindset**

*   Most ML researchers don’t want to think about GPU kernels at all — they just use PyTorch/TensorFlow layers.
*   Only a **small subset of performance-sensitive engineers** (OpenAI, Anthropic, Meta, AMD research teams) write custom kernels.
*   That keeps Triton adoption niche, even if it’s powerful.

* * *

⚡ TL;DR
-------

Yes — OpenAI uses Triton, and it’s very promising.  
But **CUDA’s ecosystem dominance**, **Triton’s youth**, and the fact that **most ML people don’t need custom kernels** mean adoption is still limited. Right now, Triton is more of a **specialist tool** than a mainstream default.

* * *

Would you like me to map out **where Triton _is_ actually being used in practice today** (e.g., OpenAI, PyTorch 2.0 compiler, AMD ROCm), so you can see the current adoption hotspots?

## Prompt:
what are some skepticals about the geak paper

## Response:
Nice — let’s dig into the **skeptical lens** 🕵️ on the GEAK paper. Even though it shows promising results, there are a few reasons to be cautious:

* * *

🔹 1. **Benchmark Limitations**
-------------------------------

*   **TritonBench-revised** has limited unit test coverage (often ≤6 tests per kernel).
*   The authors themselves note this could cause **false positives** — kernels may pass tests but fail in real workloads
    2507.23194v1 (1)
    .
*   Reported correctness rates (≥50%) might be inflated compared to broader, unseen scenarios.

* * *

🔹 2. **AMD-Centric**
---------------------

*   GEAK is built to highlight **AMD Instinct GPUs (MI250, MI300X)**.
*   That’s great for AMD’s ecosystem, but most AI infra is still **NVIDIA + CUDA**.
*   Skeptics might see this as more of a **vendor showcase** than a universally usable tool.

* * *

🔹 3. **Reliance on LLMs**
--------------------------

*   LLM-generated kernels are:
    *   Often **brittle** (small input changes can break them).
    *   Not guaranteed to generalize.
    *   Dependent on access to **frontier LLMs** (GPT-4.1, Gemini 2.5 Pro, etc.), which are closed-source and costly.
*   This raises questions about **reproducibility and accessibility**.

* * *

🔹 4. **Performance Claims**
----------------------------

*   They report up to **2.59× speedups**, but:
    *   These are against **reference kernels**, not necessarily expert hand-tuned CUDA/cuBLAS/cuDNN kernels.
    *   In real production, libraries are highly optimized — GEAK may not beat them consistently.
*   So skeptics might argue the performance wins are **benchmark-specific**.

* * *

🔹 5. **Debugging Trap**
------------------------

*   They admit GEAK can get stuck in “debugging loops” (repeating the same fix attempts).
*   Their solution (limit retries) is a band-aid.
*   It highlights that the system **isn’t robustly self-correcting** yet.

* * *

🔹 6. **Compute Cost**
----------------------

*   GEAK relies on **inference-time compute scaling** (multiple iterations, parallel generations).
*   This is compute-hungry — not ideal for everyday developers.
*   Critics might ask: _is it worth burning lots of GPU/LLM cycles just to generate one kernel?_

* * *

🔹 7. **Real-World Adoption**
-----------------------------

*   Open-source release is good, but:
    *   Will researchers/companies actually trust **AI-written low-level GPU code** in production?
    *   Unit test coverage and correctness guarantees aren’t at the level of hand-tuned kernels.
*   Skeptics might see it more as a **research prototype** than something deployable.

* * *

⚡ Skeptical Takeaway
--------------------

GEAK is exciting as a **proof-of-concept**, showing LLMs can automate GPU kernel writing.  
But concerns around **benchmark coverage, AMD bias, cost, reproducibility, and real-world reliability** mean it’s still early-stage.

* * *

Do you want me to put together a **pros vs. skepticisms table** so you can see the hype vs. the caveats side by side?

## Prompt:
yes

## Response:
Here’s a **Pros vs. Skepticisms table** for GEAK 👇

* * *

✅ Pros vs. 🤔 Skepticisms of GEAK
---------------------------------

| **Aspect** | **Pros (What GEAK Achieves)** | **Skepticisms / Limitations** |
| --- | --- | --- |
| **Correctness** | Boosts execution accuracy to ~55–63%, far better than <15% from direct LLM prompting
2507.23194v1 (1)

. | Benchmarks (esp. TritonBench-revised) have **low test coverage** — risk of inflated correctness. |
| **Performance** | Shows up to **2.59× speedup** over reference kernels. | Speedups are vs. “reference” kernels, not necessarily **expert-tuned CUDA/cuBLAS/cuDNN** — real-world gains may be smaller. |
| **Framework Design** | Clever **agentic loop** (Generate → Evaluate → Reflect → Optimize) ensures iterative improvement. | Reflection can fall into **debugging traps** (repeating same errors); capped retries are a workaround, not a full fix. |
| **Accessibility** | Open-sourced framework + benchmarks encourage community involvement. | Requires **AMD hardware (MI250/MI300)** + ROCm, plus access to **frontier LLMs** (GPT-4.1, Gemini) → high barrier. |
| **Innovation** | Shows that LLMs + inference-time compute scaling can write usable low-level GPU code. | Heavy reliance on **expensive, closed-source LLMs** raises reproducibility and cost concerns. |
| **AMD Ecosystem** | Provides tooling for AMD Instinct GPUs, where ecosystem is less mature than CUDA. | Very **AMD-centric**; broader ML world is still **CUDA/NVIDIA dominated**. |
| **Future Potential** | Could democratize GPU kernel optimization and reduce dependence on expert human tuning. | Still looks like a **research prototype**; trust in AI-generated kernels in **production systems** is low. |

* * *

⚡ **In short:**

*   **Pro-AMD/research angle**: GEAK shows LLMs can meaningfully automate kernel writing.
*   **Skeptical angle**: GEAK’s results may not generalize beyond benchmarks, AMD hardware, or controlled experiments.

* * *

Do you want me to also add a **“What it would take to win skeptics over”** section (i.e., concrete steps that could make GEAK more credible and widely adopted)?

## Prompt:
yes

## Response:
Got it 👍 — here’s a follow-up section:

* * *

🔑 What It Would Take to Win Skeptics Over (for GEAK)
-----------------------------------------------------

### 1\. **Stronger Benchmarks**

*   Expand beyond TritonBench-revised (low coverage).
*   Use **high-coverage test suites** (like ROCm Triton Benchmark, EvalPlus-style auto test gen).
*   Compare against **production-level baselines** (cuBLAS, cuDNN, FlashAttention).  
    👉 Would prove GEAK’s kernels aren’t just “passing narrow tests,” but are **robust + competitive**.

* * *

### 2\. **NVIDIA/CUDA Support**

*   Right now, GEAK is AMD-only.
*   Adding support for **CUDA GPUs** would massively increase adoption.  
    👉 Would let skeptics see GEAK as a **general kernel generator**, not just an AMD showcase.

* * *

### 3\. **Lower Compute Costs**

*   Inference-time scaling = expensive (many LLM calls per kernel).
*   Need lighter-weight modes (e.g., distilled models fine-tuned for Triton).  
    👉 Makes it practical for everyday researchers, not just big labs.

* * *

### 4\. **Reproducibility**

*   Reliance on **closed-source LLMs** (GPT-4.1, Gemini 2.5 Pro) hurts reproducibility.
*   Open-sourcing training/eval configs, or supporting **open LLMs (e.g., DeepSeek-Coder, Qwen-Coder)**, would build trust.

* * *

### 5\. **Production-Ready Guarantees**

*   Improve **debugging robustness** (avoid endless “reflection loops”).
*   Formal verification tools for memory safety, performance bounds.
*   Integration with frameworks (PyTorch 2.0 compiler, JAX).  
    👉 Gives engineers confidence to **deploy AI-generated kernels** in real workloads.

* * *

### 6\. **Community Adoption**

*   More tutorials, case studies, and success stories outside AMD.
*   If **academic HPC labs, startups, or OpenAI/Meta/Anthropic** publicly adopt it, that drives credibility.

* * *

⚡ **Bottom Line:**  
To win skeptics over, GEAK needs:

*   **Broader benchmarks, NVIDIA support, cheaper inference, reproducibility, and production guarantees.**  
    If it delivers on those, it could shift from “cool research demo” → “mainstream kernel generation tool.”

* * *

Do you want me to draft a **future roadmap sketch for GEAK** (like: short-term, mid-term, long-term steps AMD/GEAK team should take)? That might help put all these skeptic points into an actionable vision.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)