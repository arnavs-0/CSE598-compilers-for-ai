# Walk through paper

## Prompt:
2006.06762v5.pdf

PDF

Walk me through this paper

## Response:
Got it — let’s walk through the paper **“Ansor: Generating High-Performance Tensor Programs for Deep Learning”** step by step so you understand the main ideas and techniques.

* * *

### **1\. Problem Motivation**

*   **Goal:** Speed up Deep Neural Networks (DNNs) by automatically generating optimized low-level _tensor programs_ (e.g., for matrix multiplication, convolution).
*   **Challenge:** Existing methods:
    *   Use **manual templates** → require lots of engineering and expertise.
    *   Use **sequential search** → prune search space too aggressively, missing optimal programs.
*   **Result:** Either too narrow search space or inefficient search strategies.

* * *

### **2\. Key Idea: Ansor**

Ansor introduces a **large hierarchical search space** with:

1.  **High-level sketches** → represent program structure (e.g., tiling, fusion).
2.  **Low-level annotations** → add parameters like tile size, parallelization, unrolling.

It then:

*   Samples complete programs from this space.
*   Fine-tunes them using **evolutionary search** and a **learned cost model**.
*   Uses a **task scheduler** to prioritize tuning subgraphs that most impact performance.

* * *

### **3\. Design Overview**

Ansor consists of three main components:

1.  **Program Sampler**
    *   Generates diverse programs via sketches + random annotations.
    *   Ensures large, unbiased search space coverage.
2.  **Performance Tuner**
    *   Applies **evolutionary search** (mutation + crossover) to improve programs.
    *   Uses a **learned cost model** to quickly estimate performance → fewer real measurements needed.
3.  **Task Scheduler**
    *   Allocates search time across multiple DNN subgraphs.
    *   Uses **gradient descent** to focus on bottlenecks for end-to-end speedup.

* * *

### **4\. Technical Details**

*   **Sketch Generation:**
    *   Uses _derivation rules_ (e.g., multi-level tiling, operator fusion, caching) to build candidate program structures.
    *   Sketches are abstract; no parameters fixed yet.
*   **Random Annotation:**
    *   Fills in tile sizes, parallelization, loop unrolling, etc.
    *   Ensures wide coverage of parameter space.
*   **Evolutionary Search:**
    *   **Mutation**: tweak tile sizes, parallel granularity, computation locations.
    *   **Crossover**: combine good parts from different programs.
*   **Cost Model:**
    *   A gradient-boosted decision tree trained on profiling data predicts performance.

* * *

### **5\. Task Scheduling**

*   When tuning _all_ subgraphs of a DNN:
    *   Not all subgraphs are equally important.
    *   Ansor allocates more search time to subgraphs with the biggest impact.
*   **Gradient descent** + **ε-greedy exploration** used to balance exploitation vs exploration.

* * *

### **6\. Experimental Results**

*   **Hardware:** Intel CPU, NVIDIA GPU, ARM CPU.
*   **Baselines:** PyTorch, TensorFlow, AutoTVM, Halide Auto-Scheduler, FlexTensor.
*   **Benchmarks:**
    1.  **Single operator** (e.g., conv2D, matmul)
    2.  **Subgraph** (e.g., conv + ReLU, attention blocks)
    3.  **Full DNNs** (ResNet-50, BERT, MobileNet-V2, etc.)
*   **Key Findings:**
    *   Up to **3.8× speedup** on CPU, **2.6×** on ARM, **1.7×** on GPU vs state-of-the-art.
    *   Larger search space + evolutionary fine-tuning → better programs in less time.
    *   Outperforms even **vendor libraries** (e.g., cuDNN, MKL-DNN) on some workloads.

* * *

### **7\. Ablation Studies**

*   Removing **task scheduler** → wastes time on less critical subgraphs.
*   Limiting **search space** → misses best programs.
*   Removing **fine-tuning** → slower convergence, worse final performance.

* * *

### **8\. Contributions**

1.  Large hierarchical search space for tensor programs.
2.  Evolutionary search + learned cost model for efficient exploration.
3.  Task scheduler to prioritize performance-critical subgraphs.
4.  Open-source implementation outperforming existing methods.

* * *

I can also make a **diagram or flowchart** summarizing Ansor’s workflow if you want.  
Do you want me to create that?

## Prompt:
explin me how the graphs and sketch works. what does that accomplish

## Response:
Got it — let me break down **how the graphs and sketches in Ansor work** and what they accomplish, because this is at the heart of the paper.

* * *

**1\. Starting Point: Computational Graph**
-------------------------------------------

*   A DNN can be represented as a **directed acyclic graph (DAG)**:
    *   **Nodes = Operators** (e.g., convolution, matrix multiply, ReLU).
    *   **Edges = Data dependencies** (tensors flowing between operators).

Example:

```
A → MatMul → Add → ReLU → Output
```

This is your **computation definition** — essentially, _what_ to compute, but not _how_ to implement it efficiently on hardware.

* * *

**2\. Why Do We Need Sketches?**
--------------------------------

The challenge:

*   Many different **loop orders**, **tiling strategies**, **fusion patterns**, etc., can compute the _same_ result.
*   Some choices run much faster on specific hardware than others.
*   Existing methods either:
    *   Use **fixed templates** → small search space.
    *   Use **sequential search** → prune too aggressively and miss good programs.

**Ansor solves this** by:

1.  Generating a **hierarchical search space**:
    *   **High-level sketches**: define program structure.
    *   **Low-level annotations**: fill in parameters (tile sizes, parallelization, etc.).
2.  Searching this space efficiently.

* * *

**3\. How Sketch Generation Works**
-----------------------------------

Think of **sketches** as **blueprints** for programs:

*   Start from the DAG.
*   Apply **derivation rules** recursively to build program structures:
    *   **Multi-level tiling** → break loops into smaller blocks for cache & parallelism.
    *   **Operator fusion** → combine ops like Conv + ReLU to avoid memory writes.
    *   **Caching** → store intermediate results in faster memory before writing to main memory.
    *   **Reduction factorization** → expose parallelism in reduction loops.

Each derivation rule transforms the graph to a _better structured_ program.

### Example:

Naive matmul:

```
for i, j, k: C[i,j] += A[i,k]*B[k,j]
```

After applying tiling + fusion:

```
for i0, j0, k0:  # outer tiles
  for i1, j1, k1:  # inner tiles (cache friendly)
    C_tile += A_tile * B_tile
```

*   **Sketch**: the _structure_ of loops (i0, j0, k0, i1, j1, k1).
*   **Annotations**: actual **tile sizes**, parallelization decisions, etc.

* * *

**4\. Why Not Just Pick Parameters Immediately?**
-------------------------------------------------

Two reasons:

1.  **Separating structure from parameters**:
    *   The _structure_ decides what loops/operators exist.
    *   The _parameters_ decide exact tuning knobs (e.g., tile size = 32 vs 64).
    *   This hierarchy gives _huge_ flexibility without manual templates.
2.  **Avoiding pruning mistakes**:
    *   Sequential methods prune early based on incomplete programs → miss good ones.
    *   Ansor only evaluates _complete_ programs → no premature elimination.

* * *

**5\. What This Accomplishes**
------------------------------

*   **Big Search Space**: Many structures + many parameter choices → covers programs others miss.
*   **Modular Search**: Can add new rules for new hardware (e.g., Tensor Cores).
*   **Efficiency**: Only promising _complete_ programs go through costly tuning.

Visually:

```
Graph (DAG)
   ↓
Sketch Generation (structural rules)
   ↓
Random Annotation (fill in parameters)
   ↓
Complete Programs → Evolutionary Search → Best Tensor Program
```

* * *

If you want, I can make a **diagram** showing how a DAG becomes sketches → complete programs → optimized code.

Do you want me to make that diagram?

## Prompt:
Explain to me why I would want to use this and what part of the process this replaces?

## Response:
Got it — let’s make this practical. If you’re working with deep learning models and you care about **runtime performance** across different hardware platforms, here’s why you’d want to use **Ansor** and what exactly it replaces in the typical workflow.

* * *

**1\. Typical Workflow Without Ansor**
--------------------------------------

Normally, to run a deep learning model fast on CPUs, GPUs, or accelerators, you have two main options:

1.  **Vendor-provided libraries (e.g., cuDNN, MKL-DNN):**
    *   Pros: Highly optimized for _standard operators_ (e.g., conv2d, matmul) on _specific hardware_.
    *   Cons:
        *   Limited operator support (new ops = no library support).
        *   Fixed kernels → can’t adapt to new model architectures or exotic hardware.
2.  **Manual or template-based auto-tuning (e.g., AutoTVM):**
    *   You write **templates** describing possible loop optimizations (tiling, fusion, unrolling).
    *   The system searches over parameter settings for best performance.
    *   Cons:
        *   Writing templates is **time-consuming and hardware-specific**.
        *   Templates cover **only limited search space** → may miss optimal programs.

* * *

**2\. What Ansor Replaces**
---------------------------

Ansor automates **both** the template-writing and much of the search process:

*   **You only define the computation mathematically** (e.g., matmul formula).
*   Ansor:
    1.  **Generates the search space automatically** using sketches.
    2.  **Searches** that space efficiently with evolutionary tuning + cost models.
    3.  Produces **highly optimized code** for your hardware.

So it **replaces**:

*   Manual template engineering.
*   Vendor-locked kernel reliance.
*   Sequential search methods that prune too aggressively.

You no longer need to:

*   Handcraft tiling strategies for each operator.
*   Guess loop schedules or rely on rigid templates.
*   Depend on black-box vendor kernels that might not support your operators.

* * *

**3\. Why You’d Want to Use It**
--------------------------------

You’d use Ansor because it gives you:

| **Benefit** | **Why It Matters** |
| --- | --- |
| **Automation** | No manual template design → saves huge engineering time. |
| **Cross-hardware portability** | Generates optimal kernels for CPUs, GPUs, ARM, accelerators → same model, different devices. |
| **Performance beyond libraries** | Can beat vendor kernels on unusual shapes, new ops, small batch sizes. |
| **Full-model optimization** | Task scheduler focuses on subgraphs that matter for _end-to-end_ latency. |
| **Scalability** | New hardware or operators → just define the math, Ansor handles tuning automatically. |

* * *

**4\. Where It Fits in the Pipeline**
-------------------------------------

Here’s the **before vs after** view:

| **Step** | **Without Ansor** | **With Ansor** |
| --- | --- | --- |
| Define computation (e.g., Conv2D) | Same in both | Same in both |
| Create search space | Manual templates / vendor kernels | **Automatic sketches** |
| Tune for performance | Sequential search or vendor libraries | **Evolutionary search + cost model** |
| Allocate tuning time | Equal time per subgraph or manual heuristics | **Task scheduler optimizes critical parts** |
| Generate final code | Static kernels / tuned templates | **Optimized tensor programs** |

End result:  
**You get near-optimal code with minimal manual effort, portable across hardware, and often faster than existing solutions.**

* * *

If you want, I can show you **exactly where in a PyTorch/ONNX → deployment pipeline** Ansor plugs in and what it replaces visually.

Do you want me to make that diagram?

## Prompt:
Yes, make the diagram

## Response:
![Generated image](https://chatgpt.com/backend-api/estuary/content?id=file_000000007eb0622fbf22dabf4fc7cb47&ts=489375&p=fs&cid=1&sig=dcc176022254f3596daab39005c03d0f3c831d4e9d853dabd14ca6f78c121f5d&v=0)

## Prompt:
Give me an example of a situation in which I would want to use this

## Response:
Let’s go through a concrete example so it’s clear where **Ansor** shines.

* * *

### **Situation: Deploying a New Model on an Edge Device**

Imagine you’re developing a computer vision model for **real-time object detection** on a **low-power ARM CPU** in a drone or mobile device.

*   Your model includes **custom layers**:
    *   Depthwise separable convolutions
    *   A custom attention mechanism
    *   Unusual activation functions

* * *

### **Without Ansor**

*   **Vendor libraries** (e.g., cuDNN, MKL-DNN) may not:
    *   Support your custom operators.
    *   Optimize well for **small batch sizes** or **non-standard tensor shapes**.
*   **Manual template tuning** (e.g., AutoTVM):
    *   Requires **significant engineering effort** to write templates for each custom op.
    *   You’d need to handcraft tiling/fusion strategies for ARM CPU by trial and error.
    *   Search space would still be **limited**.

Result:

*   Your custom layers likely run **sub-optimally**.
*   Deployment is bottlenecked by a few slow operators.

* * *

### **With Ansor**

1.  **Feed in your model’s computational graph** (ONNX / TensorFlow / PyTorch).
2.  Ansor automatically:
    *   Generates **sketches** for all subgraphs, including your custom ops.
    *   Randomly samples parameters (tiling, fusion, loop order).
    *   Uses **evolutionary search** + **cost model** to find fast kernels.
    *   Schedules more time on performance-critical subgraphs.
3.  Produces **optimized tensor programs** for ARM CPU without manual tuning.

Result:

*   Real-time inference speed.
*   Portable code for future hardware.
*   Minimal engineering overhead.

* * *

If you want, I can give you **another example for server GPUs** where Ansor beats vendor libraries on big models like **BERT** or **ResNet-50**.

Do you want me to give you that example too?

## Prompt:
Yes

## Response:
Alright, let’s go through a **server GPU example** where **Ansor** can outperform vendor libraries.

* * *

### **Situation: Large-Scale Inference on NVIDIA GPUs**

Suppose you’re deploying **BERT** or **ResNet-50** on a data center GPU (e.g., NVIDIA V100 or A100).

You care about:

*   **Low latency** for real-time applications (e.g., chatbots, recommendation systems).
*   **High throughput** for batch inference.
*   Ability to adapt quickly when:
    *   New GPU hardware comes out.
    *   Model architectures change.

* * *

### **Without Ansor**

*   You rely on **cuDNN** or **TensorRT** for GPU kernels.
*   These libraries:
    *   Are **heavily optimized** for standard convolution, matmul, etc.
    *   But **don’t always optimize across subgraphs** → each operator is fast, but data movement between them is not.
    *   **Don’t handle unusual tensor shapes** well (e.g., BERT’s attention layers with varying sequence lengths).
*   Manual tuning for each subgraph or custom kernel = **huge engineering cost**.

Result:

*   End-to-end performance is **limited by subgraph bottlenecks**.
*   Performance tuning **doesn’t scale** when models evolve rapidly.

* * *

### **With Ansor**

1.  **Input:** Full computational graph of BERT or ResNet-50.
2.  Ansor automatically:
    *   Partitions the model into subgraphs.
    *   Generates **sketches** for each subgraph → large search space of kernel structures.
    *   Uses **evolutionary search** to explore optimizations like:
        *   Multi-level tiling
        *   Operator fusion
        *   Caching
    *   **Task scheduler** prioritizes bottleneck layers first (e.g., large matmuls in BERT’s attention).
3.  Outputs optimized GPU kernels for **end-to-end inference speedup**.

* * *

### **Result**

*   **Up to 1.7× faster** inference vs TensorRT/cuDNN on NVIDIA GPUs (as shown in the paper).
*   Portable across GPU generations (e.g., V100 → A100 → H100).
*   No manual tuning required → works out of the box for evolving models.

* * *

If you want, I can create a **side-by-side performance comparison chart** showing where Ansor beats vendor libraries for both **edge CPUs** and **server GPUs**.

Do you want me to make that chart?

## Prompt:
Explain to me the sketches again. What does that do? Why do I need it?

## Response:
Got it — let me break **sketches** down step by step because they’re central to why Ansor works so well.

* * *

**1\. The Core Problem**
------------------------

When you want to run a model fast on CPUs, GPUs, or accelerators, you need to generate **optimized low-level code** (tensor programs) for each operator or subgraph.

But there’s a **huge search space** of possible implementations:

*   Loop orders
*   Tiling strategies (block sizes for cache/GPU threads)
*   Parallelization & vectorization choices
*   Operator fusion decisions (e.g., Conv + ReLU together)

**Naive search is too big → slow and inefficient.**  
**Manual templates → small, incomplete search space.**

So you need a way to **cover a big search space** while keeping search **efficient**.

* * *

**2\. What Sketches Do**
------------------------

A **sketch** is like a **blueprint** of a program’s structure **without fixing all details** yet.

Think of it like designing a house:

*   The **floor plan** = Sketch → rooms, walls, doors.
*   The **decor & furniture** = Annotations → paint colors, furniture size.

In Ansor:

*   Sketch = **high-level loop and dataflow structure** (tiling, fusion, caching).
*   Annotation = **low-level tuning parameters** (tile sizes, parallelism, unrolling).

* * *

**3\. Why Separate Structure & Parameters**
-------------------------------------------

Without sketches, you have two bad options:

1.  **Manual templates**:
    *   Humans hardcode structures → tiny search space → miss optimal programs.
2.  **Brute force search**:
    *   Randomly try all loop orders, tilings, etc. → impossible (too big).

By splitting into **sketch first, annotations later**:

*   Sketches **enumerate structural possibilities** efficiently with **derivation rules**.
*   Random annotations then **fill in parameters** for each sketch.
*   Evolutionary search + cost model **fine-tunes** parameters → best final program.

This avoids:

*   **Early pruning mistakes** (like sequential methods).
*   **Template writing burden** (like AutoTVM).

* * *

**4\. Why You Need Sketches**
-----------------------------

You need sketches because they:

| **Problem Without Sketches** | **Sketch Solution** |
| --- | --- |
| Hard to explore large search space efficiently | Enumerate structures first, then fill in details later |
| Manual templates take months to write | Sketch rules are automatic & reusable across operators |
| Search prunes too early, missing good programs | Complete programs only → accurate performance estimation |
| Limited optimization coverage | More tiling, fusion, caching patterns than templates allow |

* * *

**5\. Example**
---------------

Naive matrix multiplication:

```python
for i in range(N):
  for j in range(M):
    for k in range(K):
      C[i,j] += A[i,k] * B[k,j]
```

Sketch might add:

*   **Multi-level tiling**: break i,j,k into outer & inner loops for cache blocking.
*   **Fusion**: combine post-processing ops (e.g., ReLU).
*   **Caching**: store partial results in fast memory.

Annotation then chooses:

*   Exact tile sizes (e.g., i0=64, i1=4).
*   Which loops to parallelize or unroll.

* * *

**6\. Bottom Line**
-------------------

*   **Sketches = structured, rule-based program blueprints**.
*   They let Ansor **cover huge search spaces automatically** → no manual kernel design.
*   Combined with **annotations + evolutionary tuning**, they yield **high-performance code** for any hardware.

* * *

If you want, I can **draw a visual diagram** showing how a DAG → Sketch → Annotated Program → Final Optimized Kernel flows through Ansor.

Do you want me to make that diagram?

## Prompt:
So why is this not used more often?

## Response:
Excellent question — and honestly, it’s one that a lot of systems researchers talk about.

**Ansor is powerful**, but it’s not _everywhere_ yet because of a mix of **engineering complexity**, **integration challenges**, and **industry inertia**. Let’s unpack that clearly.

* * *

**1\. It Solves a Hard Problem — But Adds Complexity**
------------------------------------------------------

Ansor’s approach (automatic sketch generation + evolutionary tuning) is **computationally heavy** during the tuning phase.

*   To get optimal results, Ansor may run **thousands of program measurements** on the target hardware.
*   That means it can take **hours or days** to tune a full model if you want the best performance.
*   For big models or hardware clusters, this can be costly in developer time and compute.

So while it _saves manual engineering_, it trades that for **offline tuning time** — something not every company wants to spend resources on.

* * *

**2\. Integration Into Production Frameworks Is Nontrivial**
------------------------------------------------------------

Most production ML stacks (TensorFlow, PyTorch, ONNX Runtime) already depend on **vendor libraries** like cuDNN, MKL-DNN, and TensorRT that are:

*   Stable
*   Battle-tested
*   Maintained directly by hardware vendors

Integrating Ansor means:

*   Replacing parts of that backend (compiler, runtime, kernel selection logic)
*   Handling differences in operator semantics, memory layouts, and backend-specific codegen

That’s **non-trivial** engineering for large systems, especially since frameworks already have highly tuned kernels for mainstream workloads.

* * *

**3\. Vendor Lock-In Incentives**
---------------------------------

Hardware vendors (NVIDIA, Intel, ARM) have strong incentives to push their **proprietary libraries** (e.g., cuDNN, oneDNN) because:

*   They control the hardware.
*   They can release finely tuned, proprietary kernels that show benchmark wins.

Ansor is **hardware-agnostic and open-source**, which is great for research, but **less aligned** with vendor ecosystems that prefer proprietary optimization stacks.

* * *

**4\. Research Systems Lag Behind Productization**
--------------------------------------------------

Ansor is primarily a **research system** built on TVM (an academic + open-source compiler).  
In research:

*   It’s been very influential — its ideas shaped **TVM’s next-generation auto-scheduler** and inspired **similar systems** like Meta’s TensorComprehensions and OpenAI’s Triton.
*   But turning a research compiler into a **production compiler** that runs at scale (with debugging tools, determinism, versioning, etc.) takes years.

That process is ongoing — parts of Ansor’s design _are_ being adopted gradually.

* * *

**5\. Dynamic Models Are Hard**
-------------------------------

Ansor excels at **static computation graphs** (e.g., CNNs, BERT).  
But it struggles with **dynamic models** — ones that change control flow or tensor shapes at runtime (e.g., some NLP models or reinforcement learning policies).

That’s because:

*   Ansor needs to know the computation DAG ahead of time to generate sketches.
*   Dynamic shape or control flow breaks static assumptions → needs new research (which is happening now).

* * *

**6\. Industry Is Moving Gradually Toward It**
----------------------------------------------

Although Ansor itself isn’t a drop-in tool in PyTorch or TensorFlow yet, its **core ideas are being absorbed**:

*   **TVM’s auto-scheduler** (based on Ansor) is used in production by companies like **OctoML**.
*   **Meta’s Glow** and **OpenAI’s Triton** use similar automatic kernel generation.
*   **NVIDIA’s cuTensorNet** and **TensorRT** are adding more dynamic tuning internally — essentially converging on the same principles.

So, it’s not unused — it’s **just not directly visible** to end users because it’s still being integrated into toolchains.

* * *

### **Summary**

| **Reason** | **Impact** |
| --- | --- |
| High tuning cost | Great for static models, less ideal for quick iterations |
| Integration complexity | Hard to plug into big frameworks |
| Vendor ecosystems | Proprietary kernels dominate |
| Research–production gap | Ongoing adoption in new compilers |
| Dynamic models | Still an open research area |

* * *

In short:  
Ansor is **cutting-edge compiler technology** that **redefines how we optimize deep learning**, but it’s still in the **transition phase from research to production**.

It’s already influencing the next generation of ML compilers — and you’ll likely see its concepts in the systems you use in the next few years.

* * *

Would you like me to show you **how Ansor’s ideas evolved into modern compiler stacks** (like TVM’s auto-scheduler, Triton, and Meta’s AITemplate)? That’s a neat lineage.

## Prompt:
so why was this made if it cant compete, whats the novelty?

## Response:
That’s a sharp and very fair question — and it gets to the heart of what makes **Ansor** important as a _research breakthrough_, even if it’s not yet the “mainstream” production tool.

Let’s unpack it carefully.

* * *

**1\. The Goal Wasn’t to Compete With cuDNN — It Was to Eliminate the Manual Labor**
------------------------------------------------------------------------------------

Before Ansor (pre-2020), **high-performance deep learning compilers** like TVM, AutoTVM, and Tensor Comprehensions all relied on **human-written templates**.

*   Every operator (Conv2D, MatMul, etc.) required a **handcrafted kernel template** per hardware backend (CPU, GPU, ARM…).
*   Writing and tuning those templates was:
    *   **Extremely time-consuming** (thousands of lines per operator)
    *   **Hard to maintain** across new devices
    *   **Not generalizable** to new or custom operators

So the novelty wasn’t _“beat cuDNN today”_, it was:

> **“Can we automatically discover the kinds of high-performance tensor programs that humans or vendors spend months engineering — without any manual templates?”**

And Ansor succeeded at that.

* * *

**2\. The Key Novelty: Hierarchical, Template-Free Search Space**
-----------------------------------------------------------------

Ansor introduced a **fundamentally new way to search for performant code**.

### Before Ansor:

*   AutoTVM → required **manual templates**
*   Halide auto-scheduler → **sequentially** builds partial programs (bad at scaling)

### Ansor’s idea:

*   Build a **hierarchical search space**:
    *   **High-level sketches** capture structure (tiling, fusion, caching)
    *   **Low-level annotations** fill in tuning parameters
*   Sample **complete programs** from this space, then fine-tune them with evolutionary search.

This solved two long-standing issues:

1.  It **removed the need for human templates**.
2.  It **searched a much larger space** while remaining efficient.

That’s a massive conceptual leap in compiler automation.

* * *

**3\. Another Novelty: End-to-End Task Scheduling**
---------------------------------------------------

Ansor didn’t just tune _one operator_ — it introduced a **task scheduler** that:

*   Tunes _multiple subgraphs_ in a DNN together.
*   Uses a **gradient-descent-based algorithm** to prioritize the layers that most affect end-to-end latency.
*   Learns _where_ to spend tuning time — something no previous system did.

This moves optimization from “local tuning” (one op at a time) to **“global optimization”** across the whole model.

* * *

**4\. Empirical Results: It _Did_ Compete — Just in a Different Domain**
------------------------------------------------------------------------

Ansor wasn’t beating cuDNN on every convolution — but:

*   It **matched or outperformed AutoTVM**, **Halide**, and **FlexTensor** (the best open systems at the time).
*   It **beat vendor libraries** on:
    *   Custom operators (e.g., capsule convolution, matrix 2-norm)
    *   Unusual tensor shapes
    *   Edge hardware (ARM CPUs, embedded devices)
*   It showed **up to 3.8× CPU speedup** and **1.7× GPU speedup** over the previous state of the art in _automated systems_.

So it _did_ compete — just not as a drop-in replacement for NVIDIA’s hand-engineered cuDNN, which has teams of hardware engineers behind it.

* * *

**5\. Broader Novelty: It Shifted the Direction of the Field**
--------------------------------------------------------------

After Ansor, nearly every next-generation ML compiler incorporated its ideas:

| System | Inspired by Ansor concept | Key Similarity |
| --- | --- | --- |
| **TVM Auto-Scheduler (2021)** | Direct successor | Uses sketches + learned cost model |
| **OpenAI Triton** | Similar decoupling | Structure + parameter separation |
| **Meta AITemplate / Glow** | Subgraph-level optimization | Graph-aware scheduling |
| **MLIR (Google)** | Hierarchical IR design | Structural decomposition |

So, Ansor’s real impact was **conceptual**:  
It proved that _fully automatic, template-free, search-based compilation_ for deep learning was possible — and effective.

* * *

**6\. The Summary**
-------------------

You can think of Ansor as **the bridge between hand-tuned kernels and fully automatic ML compilers**:

| **Era** | **Approach** | **Limitation** | **Ansor’s Role** |
| --- | --- | --- | --- |
| Vendor libraries | Hand-coded kernels | Not portable, not extensible | Showed automation can reach similar performance |
| Template-based auto-tuners | Manual search spaces | Requires expertise, small space | Removed manual templates entirely |
| Sequential auto-schedulers | Beam search of partial programs | Inefficient and inaccurate | Introduced hierarchical full-program sampling |
| Post-Ansor compilers | Automated + ML-driven | Now industry trend | Built on Ansor’s principles |

* * *

### 🧠 **So the novelty:**

Ansor wasn’t made to beat cuDNN directly —  
it was made to **eliminate the human from the tuning loop** and prove that **a compiler can autonomously discover performant code**.

And that idea fundamentally changed how ML compilers are designed today.

* * *

Would you like me to show a **timeline diagram** that visually traces how Ansor’s ideas influenced later compilers (TVM auto-scheduler → Triton → AITemplate)?



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)