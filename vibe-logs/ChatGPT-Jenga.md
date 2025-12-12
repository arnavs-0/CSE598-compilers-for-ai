# Jenga memory management

## Prompt:
2503.18292v1.pdf

PDF

walk me through jenga, how does this work?

## Response:
The paper you uploaded — **“JENGA: Effective Memory Management for Serving LLM with Heterogeneity”** — presents a new memory management framework designed to make **LLM inference more efficient** on GPUs, especially for modern _heterogeneous_ models (like multimodal LLMs or those mixing self-attention and Mamba layers).

Here’s a clear walkthrough:

* * *

### 🧱 The Problem

Modern LLMs (e.g., LLaVA, Gemma-2, Jamba, Hymba) are _heterogeneous_:

*   Different layers produce **embeddings of varying sizes** (e.g., text vs. image tokens).
*   Different layers have **different token dependency patterns** — some depend on _all_ previous tokens (full attention), others only on recent ones (sliding window, Mamba).

Current systems like **PagedAttention** assume:

1.  All embeddings are the same size.
2.  Every token depends on _all previous_ tokens.

Those assumptions no longer hold → causing:

*   **Memory fragmentation** (unused GPU space)
*   **Inefficient caching** (wrong eviction or reuse decisions)

* * *

### 🧩 JENGA’s Key Idea

JENGA introduces a **two-level memory allocator** + **customizable caching policy** to handle heterogeneity efficiently.

#### 1\. **Two-Level Memory Allocation**

JENGA divides GPU memory in two layers:

*   **Bottom layer:** Large, fixed-size _pages_ (shared among layer types)
*   **Top layer:** Smaller _customized pages_ for each layer type (e.g., text, image, Mamba)

It chooses the large page size as the **Least Common Multiple (LCM)** of all embedding sizes to ensure compatibility and reduce fragmentation.

**Example:**

*   Text embeddings = 384 bytes
*   Image embeddings = 256 bytes  
    → LCM = 768 bytes → all layers can fit neatly inside this grid.

This design prevents gaps and reuses memory across different layers efficiently.

* * *

#### 2\. **Request-Aware Allocation**

To avoid internal fragmentation:

*   JENGA allocates pages belonging to the _same request_ together.
*   When a request finishes, all its pages can be freed at once → fewer stranded pages.

* * *

#### 3\. **Customizable Prefix Caching**

Different layers have different rules for what counts as a “cache hit”:

*   **Self-attention:** needs _all_ prefix tokens.
*   **Sliding window:** only needs the _last N tokens_.
*   **Mamba:** only keeps the _latest token state_.

JENGA exposes APIs (`update_last_access`, `set_prefix_length`, `get_possible_prefix`) for each layer type to define:

*   How it updates cache access times.
*   What subset of tokens to retain.
*   Which pages to evict.

This lets it **balance** and **align** eviction across layers — so one layer doesn’t evict tokens another layer still needs.

* * *

### ⚙️ Implementation

*   Implemented in **~4000 lines of Python** on top of **vLLM**.
*   Requires **no changes to CUDA kernels**.
*   Works transparently with all 90 models supported by vLLM (v0.6.4).

* * *

### 📊 Results

*   **GPU memory utilization** ↑ by up to **79.6%**
*   **Serving throughput** ↑ up to **4.92×** (1.80× on average)
*   **Latency** remains unchanged or improves at higher loads
*   Works for diverse models: LLaVA, Gemma-2, Ministral, Jamba, PyramidKV, etc.
*   Especially powerful for **vision-language** and **speculative decoding** workloads.

* * *

### 🧠 Analogy

Think of GPU memory like a big LEGO board:

*   PagedAttention gave you fixed 2×2 blocks — easy, but wasteful if you need 1×2 or 3×3.
*   **JENGA** figures out a shared grid (LCM) so all pieces — big or small — fit together neatly.
*   Plus, it’s “smart” about which pieces to remove (cache eviction) without toppling the structure.

* * *

Would you like me to draw a diagram showing how JENGA’s two-level memory allocation works compared to PagedAttention? It’s quite intuitive once visualized.

## Prompt:
Yes, compare it with vLLM as a table

2/2

## Response:
Here’s a clear comparison table between **vLLM (PagedAttention)** and **JENGA**, focusing on their design, assumptions, and performance:

* * *

### 🧠 **JENGA vs. vLLM (PagedAttention)**

| **Aspect** | **vLLM (PagedAttention)** | **JENGA** |
| --- | --- | --- |
| **Core idea** | Fixed-size paging to reduce fragmentation when storing KV caches for LLMs | Two-level allocator with customizable cache policies for heterogeneous LLMs |
| **Assumptions** | 1\. All embeddings have the same size  
2\. All tokens depend on the full prefix | Supports heterogeneous embeddings (different sizes, dependencies) |
| **Memory structure** | Single-level paging (fixed-size pages) | Two-level allocator:  
• **LCM allocator** (shared page pool)  
• **Custom allocators** per layer type |
| **Page size selection** | Fixed (usually one embedding size multiple) | **Least Common Multiple (LCM)** of all embedding sizes → ensures compatibility with minimal fragmentation |
| **Memory fragmentation** | Moderate to high for heterogeneous models | Very low (up to **79.6%** better utilization) via request-aware allocation |
| **Supported model types** | Homogeneous Transformers (e.g., Llama, GPT) | Heterogeneous LLMs:  
• Sliding window attention (Gemma-2, Ministral)  
• Mamba layers (Jamba)  
• Vision-Language models (LLaVA, InternVL)  
• Speculative decoding (multi-model) |
| **Cache management** | Global eviction / caching (same rule for all layers) | **Layer-specific prefix caching:**  
• Self-attn: full prefix  
• Sliding window: last _N_ tokens  
• Mamba: last token only |
| **Eviction policy** | LRU per request (same across all layers) | **Balanced + aligned eviction** across layer types using custom `update_last_access()` and `set_prefix_length()` |
| **Internal fragmentation control** | None | **Request-aware allocation:** keeps same-request pages grouped for efficient freeing |
| **Compatibility** | Works with vLLM engine and kernels | Fully compatible with vLLM kernels (no CUDA change) |
| **Implementation complexity** | Simple, fixed allocator | ~4K lines of Python on top of vLLM; minimal worker/kernel changes |
| **Performance gain** | Baseline (1×) | **Up to 4.92× throughput** (1.80× avg), **no latency penalty** |
| **GPU memory utilization** | Lower (wastes space with mixed-size embeddings) | **Up to +79.6% utilization improvement** |
| **Scalability to multimodal models** | Limited (requires same embedding dimensions) | Fully supports multimodal embeddings (text + vision + Mamba) |
| **Speculative decoding support** | Partial; manual page tuning required | Automatic allocation of multi-model caches with negligible overhead |
| **Kernel modification** | Requires kernel awareness of fixed-size pages | Reuses existing PagedAttention kernels unchanged |
| **Analogy** | Fixed-size filing cabinets for identical folders | Adjustable modular shelves (LCM-based) for folders of all shapes and sizes |

* * *

Would you like me to make a visual diagram showing how the **two-level allocation (LCM allocator + custom small pages)** compares to vLLM’s single-level page layout? It’s a good visual to see _why_ JENGA saves so much memory.

## Prompt:
So which one is better?

2/2

## Response:
✅ **Short answer:**  
**JENGA** is _better_ than **vLLM (PagedAttention)** — especially for **modern heterogeneous LLMs** — because it manages GPU memory more efficiently, supports new model architectures, and increases throughput significantly **without hurting latency**.

* * *

### 🧩 **Detailed reasoning**

| **Scenario** | **Winner** | **Why** |
| --- | --- | --- |
| **Classic transformer models (e.g., LLaMA, GPT-3)** | ⚖️ _Tie_ | Both perform equally well. JENGA’s extra logic doesn’t add measurable overhead for homogeneous architectures. |
| **Heterogeneous models (e.g., Gemma-2, Ministral, Jamba)** | 🏆 **JENGA** | These models mix attention types and embedding sizes. vLLM wastes memory (up to 79.6%) because it assumes uniform embeddings; JENGA handles this efficiently. |
| **Vision-Language Models (LLaVA, InternVL, Phi-3 Vision)** | 🏆 **JENGA** | vLLM allocates extra space for all token types equally; JENGA dynamically fits vision + text embeddings together and caches vision features intelligently. |
| **Speculative decoding (multiple models at once)** | 🏆 **JENGA** | Automatically allocates and shares KV cache between small and large models — no manual tuning. |
| **Peak GPU memory efficiency** | 🏆 **JENGA** | Up to **79.6%** higher utilization due to its request-aware LCM allocator. |
| **Throughput (requests/sec)** | 🏆 **JENGA** | Up to **4.92× faster** than vLLM on heterogeneous workloads (avg **1.80×**). |
| **Latency (end-to-end)** | ⚖️ _Same or slightly better_ | JENGA’s smarter batching and cache reuse reduce time-to-first-token under load. |
| **Engineering simplicity** | ⚖️ _vLLM_ | Simpler memory model and smaller codebase; good if you only serve homogeneous Transformers. |
| **Future readiness (e.g., multimodal, Mamba, sliding window)** | 🏆 **JENGA** | Explicitly designed for the new wave of hybrid and multimodal architectures. |

* * *

### ⚙️ **Summary Verdict**

| **If your workload is...** | **Use** |
| --- | --- |
| Pure text LLM (same embedding size everywhere) | ✅ vLLM (simpler, mature) |
| Mixed attention or multimodal model | 🚀 **JENGA** (much better memory efficiency and speed) |
| Speculative decoding or long-context pipelines | 🚀 **JENGA** |
| Limited GPU memory (e.g., L4s) | 🚀 **JENGA** — enables larger batches and prevents OOM |
| Need kernel simplicity and stability | ✅ vLLM |

* * *

**In essence:**  
JENGA keeps all of vLLM’s benefits but extends it to handle heterogeneous architectures — it’s the _next logical step_ for LLM serving as models move beyond uniform Transformer layers.

Would you like me to make a one-page visual summary (diagram + table) comparing their memory allocation and caching logic? It’s perfect for slides or a report.

## Prompt:
can Jenga support video llms as well? How come vLLM feels like more of the standard

## Response:
Excellent question — and you’ve hit on a key nuance that’s emerging right now in LLM infrastructure.  
Let’s unpack this carefully:

* * *

🎥 Can **JENGA** support **video LLMs**?
----------------------------------------

**Yes — in principle, JENGA _can_ support video LLMs**, but with some caveats.

### 🧠 Why it _can_:

*   **JENGA’s design is general** — it doesn’t care _what modality_ the embeddings come from.  
    It just needs to know:
    *   The **embedding sizes** for each type of input (e.g., text, image frame, audio, etc.).
    *   The **dependency pattern** between tokens (e.g., full attention, sliding window, recurrent, etc.).
*   A **video LLM** is essentially a **multimodal model**:
    *   Each video frame produces **visual embeddings** (sometimes 2D tokens per patch).
    *   The language model consumes these plus text tokens to generate text.
    *   Some architectures also store **temporal state** (like Mamba-style recurrent states).

🧩 Since JENGA already supports:

*   Vision embeddings (via its vision cache)
*   Mamba / state-space layers (via custom caching)
*   Cross-attention and multimodal fusion layers

…it can **naturally extend to video models**, which are just a larger-scale version of multimodal fusion (many vision embeddings per frame).

> **So conceptually:** JENGA’s LCM allocator + prefix-subset caching mechanism would handle video embeddings as just “another layer type” with its own size and dependency rule.

* * *

### ⚠️ But the catch:

Supporting **video LLMs at scale** also requires:

1.  **Streaming video embeddings** — these can be _huge_ (hundreds of frames × patches).  
    → JENGA’s allocator would need to manage far more frequent allocation/deallocation events.
2.  **Asynchronous I/O and offloading** — video models often move intermediate embeddings between GPU and CPU/SSD.
3.  **Temporal caching policies** — e.g., keeping only the last N frames or keyframes, not every frame’s embedding.

These are _not part of JENGA’s current implementation_ (as of the paper), but its API and allocator structure _could_ be extended to handle them — it was designed to be extensible for new layer types.

So:

> 🔧 **JENGA can support video LLMs conceptually, but you’d need to extend its caching logic for temporal embeddings.**

* * *

⚙️ Why does **vLLM** feel like “the standard”?
----------------------------------------------

That’s a great observation — and it’s true.

### 🚀 1. vLLM _is the production baseline_

*   vLLM is the **de facto standard inference engine** for serving text LLMs efficiently.
*   It introduced **PagedAttention**, which massively improved throughput (2–4× over naive inference).
*   It’s used in **Hugging Face TGI**, **Modal**, **Anyscale**, **OpenAI’s inference stack**, and many startups.

### 🧩 2. It’s stable, well-integrated, and widely supported

*   **API compatibility**: drop-in for OpenAI API or Hugging Face pipelines.
*   **Ecosystem**: Works with LoRA, quantized models, speculative decoding, FlashAttention, etc.
*   **Ease of use**: Mature Python interface; minimal CUDA tuning required.

### 🧱 3. JENGA builds _on_ vLLM — not replaces it

*   JENGA’s implementation literally _extends vLLM_ (it was implemented on top of it with ~4K lines of Python).
*   It reuses vLLM’s workers, schedulers, and CUDA kernels.
*   Think of JENGA as **vLLM++** — it’s not a competitor, it’s an _upgrade layer_ for memory management.

### 🧮 4. The “standardization” lag

*   vLLM’s API and architecture are already deeply integrated across cloud platforms.
*   JENGA is still a **research prototype (March 2025)** — proven in experiments, but not yet merged upstream.
*   It will likely **influence future vLLM releases**, rather than replace them directly.

* * *

🧭 TL;DR
--------

| **Question** | **Answer** |
| --- | --- |
| Can JENGA support video LLMs? | ✅ Yes in principle — it already handles multimodal embeddings and recurrent states; needs temporal caching extensions for full video models. |
| Why is vLLM the standard? | 📦 vLLM is production-ready, battle-tested, and integrated across the ecosystem. JENGA extends it with heterogeneous memory management but isn’t yet mainstream. |
| How they relate | 🔄 JENGA builds **on** vLLM — it’s not a fork; it’s a research evolution likely to merge into future versions. |

* * *

Would you like me to show a visual of how JENGA’s allocator could be extended to handle **video embeddings** (e.g., frames as temporal chunks with sliding eviction)? It’s a nice way to see how this could generalize.

## Prompt:
so what is a weakness of jenga in comparison to vllm

## Response:
Excellent — you’re thinking like a systems researcher now.  
While **JENGA** clearly outperforms **vLLM** on heterogeneous models, it _does_ come with a few important **trade-offs and weaknesses** relative to vanilla vLLM.

Here’s a careful breakdown 👇

* * *

⚖️ **Weaknesses of JENGA Compared to vLLM**
-------------------------------------------

| **Category** | **Weakness / Limitation** | **Why it Matters** |
| --- | --- | --- |
| 🧩 **Complexity** | **More complex allocator logic** (two-level + request-aware) | JENGA’s memory allocator (LCM allocator + custom small allocators) adds extra bookkeeping — more metadata tracking, and slightly higher control-path overhead compared to vLLM’s single-layer fixed-page model. |
| 🔧 **Implementation maturity** | **Still research-stage (as of 2025)** | JENGA was implemented in ~4K lines of Python on top of vLLM, not deeply integrated into the production vLLM codebase yet. Fewer stress tests, less exposure to real-world workloads, and limited community support. |
| ⚙️ **Engineering overhead** | **Harder to debug and tune** | The multi-level allocator can have subtle fragmentation, balancing, or eviction alignment bugs that are harder to reproduce. vLLM’s simpler allocator is easier to reason about and instrument. |
| 🧠 **Overhead on homogeneous models** | **Marginally more CPU overhead with no benefit** | For standard transformers (e.g., LLaMA, GPT-4-style models with uniform embedding sizes), JENGA’s flexibility gives no memory advantage but still incurs a small (~1–3%) control overhead. |
| 📊 **LCM allocator tradeoff** | **Possible large LCM page sizes** | For some model combinations with very different embedding sizes (e.g., text = 384 B, image = 5120 B), the LCM can become very large → internal fragmentation may _reappear_ or cause memory underutilization. |
| 🔄 **Interoperability** | **Not yet supported across all frameworks** | vLLM is already integrated with Hugging Face, Triton Inference Server, TensorRT-LLM, etc. JENGA’s extensions aren’t yet exposed in those systems. |
| 🧮 **Specialized kernels** | **Relies on vLLM’s kernels, no custom optimization yet** | While that makes it easier to adopt, it also means JENGA can’t yet exploit kernel-level optimizations tailored to its two-level layout (e.g., tensor fusion, async copy). |
| 🧰 **Limited temporal / streaming support** | **No direct support for time-sequenced caches (e.g., video, speech)** | Although its design generalizes well, JENGA doesn’t yet implement streaming/temporal eviction policies — essential for full video/audio LLMs. |
| 📦 **Memory metadata overhead** | **Slight increase in host-side metadata** | Because JENGA tracks per-request and per-layer allocations, it needs more host memory for bookkeeping (small, but measurable at massive scale). |
| 🧪 **Not yet open-source integrated** | **Unmerged with vLLM mainline** | Developers must manually integrate JENGA’s modifications to test it — no pip or Docker support yet. |

* * *

🧠 **In short**
---------------

**vLLM** is:

*   🧱 Simpler
*   🧩 More stable
*   🚀 Production-ready

**JENGA** is:

*   🔬 Smarter
*   💾 More memory-efficient
*   ⚡ Faster for complex, multimodal workloads  
    —but not yet as battle-tested.

* * *

### 📉 When JENGA _might not help_

Use **vanilla vLLM** instead if:

*   You only serve homogeneous text LLMs (e.g., GPT-J, Mistral 7B).
*   GPU memory isn’t your bottleneck.
*   You value stability, simplicity, and framework integration over maximal efficiency.

Use **JENGA** when:

*   You’re serving hybrid models (vision, Mamba, sliding-window, speculative decoding).
*   GPU memory limits your batch size or throughput.
*   You’re running research or internal deployments where performance > maintainability.

* * *

### 🔍 Summary

| **Dimension** | **vLLM** | **JENGA** |
| --- | --- | --- |
| Maturity | 🟩 Stable, widely used | 🟨 Research prototype |
| Complexity | 🟩 Simple single-level allocator | 🟥 Two-level allocator (more complex) |
| Memory efficiency | 🟨 Good | 🟩 Excellent (up to +79.6%) |
| Throughput | 🟨 High | 🟩 Much higher for heterogeneous models |
| Model compatibility | 🟨 Text-only | 🟩 Multimodal + Mamba + sliding window |
| Ecosystem support | 🟩 Full | 🟥 Limited (research only) |

* * *

If you’re deploying production inference _today_, **vLLM** is the safer choice.  
If you’re designing the _next-generation_ serving stack for multimodal or hybrid LLMs, **JENGA** represents the future direction.

* * *

Would you like me to give a concise “when to use vLLM vs JENGA” decision flowchart (like for a presentation or report)? It makes the trade-off super intuitive.

## Prompt:
so if you could fix jenga, what would you do

## Response:
That’s an excellent and very “research-minded” question — you’re thinking like a systems architect now. 👏

If I were to **evolve or “fix” JENGA** beyond its current research prototype, I’d focus on making it _production-grade_, _more adaptive_, and _kernel-aware_.  
Here’s how I’d approach it step by step:

* * *

🧱 1. **Integrate it deeply into vLLM’s core**
----------------------------------------------

Right now, JENGA is implemented as a Python-level extension on top of vLLM.  
To make it production-ready:

*   **Move the allocator into C++/CUDA** to cut Python control-path overhead.
*   **Integrate directly with vLLM’s `PagedAttention` kernel metadata flow**, so the allocator and kernels share memory layout information natively.
*   **Expose configuration knobs** (e.g., enable/disable two-level allocation) through vLLM’s existing `--memory-policy` flags.

👉 _Result:_ Lower overhead, simpler maintenance, and compatibility with future vLLM releases.

* * *

⚙️ 2. **Adaptive Page-Size Policy (Smart LCM)**
-----------------------------------------------

JENGA’s biggest potential weakness is that the **LCM of embedding sizes can explode** (e.g., 384 B vs. 5120 B → huge 5 MB pages).  
Fix:

*   Compute an **adaptive LCM/GCD hybrid** at runtime based on observed workload.
*   Use **profiling** to dynamically select between:
    *   _LCM allocator_ (for small heterogeneity)
    *   _Chunked MAX allocator_ (for large heterogeneity)
    *   _GCD sub-pages_ (for very fragmented embeddings)
*   Cache these profiles per model family.

👉 _Result:_ Keeps fragmentation low without giant page sizes.

* * *

🧮 3. **Kernel-aware page fusion**
----------------------------------

Right now, JENGA reuses vLLM’s kernels “as is.”  
Next step:

*   Modify FlashAttention/FlashInfer kernels to **read from contiguous LCM-page memory directly**.
*   Add **batched DMA copy support** so small pages can be fused into larger contiguous tensors before kernel launch.
*   Pre-record CUDA Graphs for frequent allocation patterns to cut launch overhead.

👉 _Result:_ Leverages JENGA’s better layout _and_ gets kernel-level speedups.

* * *

🧩 4. **Unified Temporal Cache Layer**
--------------------------------------

Extend prefix caching to support **temporal or streaming tokens** (for video, speech, long audio).

*   Introduce a “time dimension” to eviction policy → _temporal sliding window._
*   Allow models to register **custom eviction hooks** (e.g., drop frames older than 3 s, keep keyframes).
*   Add asynchronous GPU↔CPU spill logic for rarely accessed frames.

👉 _Result:_ Native support for **video LLMs**, speech LLMs, and streaming dialogue systems.

* * *

🔋 5. **Hybrid Memory Offload Integration**
-------------------------------------------

Combine JENGA’s fine-grained allocator with offload systems (e.g., Mooncake, CachedAttention, FlexGen).

*   Page-level GPU↔CPU↔NVMe migration.
*   Offload decisions guided by cache hit statistics and access timestamps from JENGA’s prefix-subset tracker.

👉 _Result:_ Huge models can run on limited GPU memory without manual offloading logic.

* * *

🧠 6. **Self-tuning cache policy**
----------------------------------

Turn JENGA’s per-layer eviction and prefix-length heuristics into a **learning or adaptive policy**:

*   Collect runtime telemetry (hit/miss patterns, latency).
*   Use simple online RL or bandit algorithms to pick eviction thresholds dynamically.
*   Could be extended with a “cache advisor” service like OS page replacement learning.

👉 _Result:_ Always-optimal caching decisions per model + workload.

* * *

📦 7. **Ecosystem polish**
--------------------------

*   Add integration hooks for **Triton Inference Server**, **TensorRT-LLM**, and **Hugging Face** pipelines.
*   Provide `torch.compile` / OpenAI API compatibility out-of-the-box.
*   Publish as a configurable backend: `--backend jenga`.

👉 _Result:_ Makes JENGA plug-and-play, removing the current friction of research-only usage.

* * *

### 🔮 If successful…

You’d get:

*   vLLM’s stability ✅
*   JENGA’s heterogeneity and throughput ✅
*   Temporal and multimodal readiness ✅
*   Production reliability ✅

Essentially a **“JENGA 2.0”** that becomes the next-generation standard memory manager for all future multimodal or hybrid LLMs.

* * *

Would you like me to sketch a small diagram showing how this “JENGA 2.0” architecture (adaptive LCM + temporal cache + hybrid offload) would look conceptually? It helps visualize how these fixes tie together.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)