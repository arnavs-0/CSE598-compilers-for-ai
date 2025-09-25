# Walkthrough of paper

**User:** Arnav Shah (sarn248@gmail.com)  
**Created:** 9/24/2025 21:05:01  
**Updated:** 9/24/2025 21:10:20  
**Exported:** 9/24/2025 21:10:34  

## Prompt:
2506.12598v1.pdf

PDF

Walk me through this paper

## Response:
Got it — this paper introduces **ECLIP**, a framework for energy-efficient and practical co-location of ML inference tasks on spatially partitioned GPUs, particularly focusing on **AMD GPUs**. I’ll break it down for you step by step:

* * *

**1\. Motivation**
------------------

*   ML inference servers often run **multiple models on the same GPU** to save cost and energy.
*   GPUs like NVIDIA’s SMs or AMD’s CUs have **underutilized compute resources** when running a single model.
*   Co-locating models can help but:
    *   **Model-level partitioning** wastes GPU capacity since each model has varying compute needs across kernels.
    *   **Kernel-level partitioning** could be more efficient but existing methods have **huge overhead** or require hardware changes.

**ECLIP aims to enable fine-grained partitioning with low overhead** to improve:

*   **Throughput** (more requests per second)
*   **Energy efficiency** (more requests per watt)
*   While keeping **tail latency** under control.

* * *

**2\. Key Observations**
------------------------

Using AMD’s ROCm and CU masking API, the authors found:

1.  **High Overhead**: Calling CU masking APIs at every kernel is expensive (55µs overhead vs. 20µs kernel runtime).
2.  **Queue Limits**: Too many CU-masked streams cause **hardware queue oversubscription** → longer latency after ~7 streams.

So, the design trade-off is:

*   **Fewer reconfigurations** = less overhead
*   **More granularity** = better GPU utilization

* * *

**3\. ECLIP Framework**
-----------------------

ECLIP has **two main components**:

### **A. Runtime Scheduler**

*   Pre-allocates **7 CU-masked streams** upfront to avoid runtime API overhead.
*   Redirects each kernel to the right CU-masked stream using a **lookup table**.
*   Maintains **data dependencies** via barrier packets to ensure correct execution order.

* * *

### **B. Resource Allocation Optimizer**

*   Uses profiling to find the **minimum CUs each kernel needs** without slowdown.
*   Groups kernels to **limit partitioning switches** (switching too often kills performance).
*   Solves an **Integer Linear Programming (ILP)** problem offline to:
    *   Minimize total execution time
    *   Ensure fairness across multiple workers
    *   Limit CU reconfiguration overhead
*   Stores results in a **lookup table** for fast runtime decisions.

* * *

**4\. Experimental Setup**
--------------------------

*   Hardware: AMD MI50 GPU (60 CUs, 4 Shader Engines, 15 CUs each)
*   Models: ALBERT, DenseNet201, AlexNet, ResNet15, ResNeXt101, ShuffleNet, VGG19
*   Workload mixes: different combinations of models co-located together
*   Compared methods:
    1.  **Baseline**: no partitioning
    2.  **Model-wise**: whole-model partitioning
    3.  **Kernel-wise IOCTL**: per-kernel CU masking (slow)
    4.  **Kernel-wise Prealloc**: ECLIP’s preallocated streams without optimization
    5.  **ECLIP**: full system with optimizer + scheduler

* * *

**5\. Results**
---------------

**Throughput**

*   ECLIP improves throughput by **13% on average** (up to 21%).
*   Kernel-wise IOCTL fails due to extreme overhead.
*   Kernel-wise Prealloc suffers from too many reconfigurations.

**Tail Latency**

*   ECLIP keeps tail latency controlled by **limiting reconfigurations**.
*   Model-wise partitioning sometimes has high latency due to coarse partitioning.

**Energy Efficiency**

*   ECLIP improves energy efficiency by **25% on average** (up to 35%).
*   Gains come from better GPU utilization and fewer idle periods.

* * *

**6\. Takeaways**
-----------------

*   **ECLIP preallocates CU partitions and optimizes their use across kernels.**
*   **Offline optimization + online lookup** enables fine-grained scheduling with minimal overhead.
*   Works on **existing GPUs** (unlike some prior work needing hardware changes).
*   Achieves **13% higher throughput** and **25% better energy efficiency** on average.

* * *

If you want, I can make a **visual diagram** of ECLIP’s workflow so you can quickly grasp how the scheduler and optimizer interact.

Do you want me to create that diagram?

## Prompt:
How does compare to NVIDIA's solutions?

## Response:
Got it — the paper mainly compares ECLIP to **NVIDIA’s MPS/MIG-based solutions** and other kernel-grain approaches, so here’s how it stacks up:

* * *

**1\. NVIDIA’s MPS & MIG**
--------------------------

*   **MPS (Multi-Process Service):**
    *   Allows multiple processes to share a GPU.
    *   Uses **model-level partitioning** → each process gets a set of SMs.
    *   Granularity is **coarse** → partitions remain fixed per model.
    *   **Problem**: Leaves GPU underutilized since different kernels within a model need different SM counts.
*   **MIG (Multi-Instance GPU):**
    *   Introduced in NVIDIA A100.
    *   Partitions the GPU into isolated hardware instances.
    *   **Even coarser** granularity than MPS → fixed-size slices of GPU.
    *   **Problem**: No kernel-level flexibility → wastes resources when workloads are heterogeneous.

ECLIP avoids these issues by:

*   **Kernel-grain partitioning** on existing AMD GPUs.
*   Minimizing **repartitioning overhead** via pre-allocated streams + offline optimization.

* * *

**2\. NVIDIA Kernel-grain Solutions**
-------------------------------------

Some prior NVIDIA-focused works tried kernel-level partitioning:

| Approach | Platform | Granularity | Overheads | Hardware Changes Needed? |
| --- | --- | --- | --- | --- |
| **KRISP (HPCA ‘23)** | NVIDIA (simulated) | Kernel-level | Low (via custom HW) | Yes, requires new hardware |
| **ElasticRoom (HPDC ‘24)** | NVIDIA (real GPUs) | Kernel-level | Medium (compiler overhead) | No, but limited to GEMM ops |
| **libsmctrl (RTAS ‘23)** | NVIDIA (real GPUs) | Kernel-level | High (debug API overhead) | No, but runtime mods needed |

ECLIP vs. these:

*   **KRISP**: Very low overhead but **requires microarchitectural support** not available on current GPUs.
*   **ElasticRoom**: Limited to **matrix-multiplication kernels**, needs **runtime/compiler changes**.
*   **libsmctrl**: Works today but **overhead is too high** for fine-grain partitioning.

ECLIP:

*   **Works on real GPUs** (AMD MI50 tested)
*   Achieves **similar fine-grain benefits** without hardware changes
*   **Pre-allocates CU streams** + **offline ILP optimization** → **low overhead**

* * *

**3\. Performance Comparison**
------------------------------

From the paper’s experiments:

*   **MPS/MIG (model-wise partitioning)** improves utilization somewhat but wastes capacity for kernels with small compute needs.
*   **ECLIP**:
    *   **13% better throughput** than baseline (MPS-like)
    *   **25% better energy efficiency**
    *   **Controlled latency** with partition-switch budgeting

Kernel-wise IOCTL (similar to libsmctrl) → unusable due to **extreme overhead**.

* * *

**4\. Key Insight**
-------------------

NVIDIA’s official solutions (MPS/MIG):

*   **Simpler, coarse-grain**, but low overhead.
*   **Not flexible enough** for per-kernel partitioning.

ECLIP:

*   **Brings kernel-grain partitioning benefits** (like KRISP)
*   **Avoids runtime overhead** by using pre-allocated CU masks + offline optimization
*   **Works on existing hardware** → no new GPU features needed.

* * *

If you want, I can make a **side-by-side table** comparing **ECLIP vs MPS/MIG vs KRISP vs ElasticRoom** in terms of **granularity, overhead, hardware needs, and performance**.

Do you want me to make that table?

## Prompt:
Yes

## Response:
Here’s a **side-by-side comparison** of **ECLIP** vs **NVIDIA’s solutions** and other kernel-grain approaches:

| **Approach** | **Granularity** | **Runtime Overhead** | **Hardware Changes Needed** | **Flexibility** | **Performance / Energy Efficiency** |
| --- | --- | --- | --- | --- | --- |
| **NVIDIA MPS** | Model-level | Low | No | Fixed partitions per model | Improves utilization but leaves idle SMs → low efficiency gains |
| **NVIDIA MIG** | Instance-level (coarser) | Low | No | Rigid partition sizes | Underutilized for small models or dynamic workloads |
| **KRISP (HPCA ‘23)** | Kernel-level | Very low | **Yes** (requires new hardware) | Fine-grain per-kernel partitioning | High efficiency in simulations, but not deployable on real GPUs |
| **ElasticRoom (HPDC ‘24)** | Kernel-level (GEMM only) | Medium (compiler overhead) | No | Limited to GEMM kernels only | Better utilization but narrow applicability |
| **libsmctrl (RTAS ‘23)** | Kernel-level | **High** (debug API overhead) | No | Works today but impractical runtime mods | Latency too high for production inference workloads |
| **ECLIP (This Paper)** | **Kernel-group level** | **Low** (pre-alloc + offline ILP) | No | Works on real GPUs, model-agnostic | **13%↑ throughput, 25%↑ energy efficiency, low latency** |

### **Key Takeaways**

*   **MPS/MIG** → Simple, production-ready, but **coarse-grain** partitioning wastes capacity.
*   **KRISP** → Best fine-grain performance, but **requires hardware changes**, so not deployable.
*   **ElasticRoom/libsmctrl** → Work on real GPUs, but with **high overhead** or limited scope.
*   **ECLIP** → Combines **fine-grain benefits** with **low overhead** and **no hardware changes**.

* * *

If you want, I can also **draw a visual diagram** showing how each approach partitions GPU resources differently so you can see why ECLIP wins.

Do you want me to make that diagram?

## Prompt:
So why is this eclip important? How can it actually run on multiple models on one gpu?

## Response:
Got it — let’s break down **why ECLIP matters** and **how it enables multiple ML models to share a single GPU efficiently**.

* * *

**Why ECLIP is Important**
--------------------------

1.  **Today’s Problem**
    *   Inference servers often run **many models on one GPU** to save cost and energy.
    *   But current solutions either:
        *   **MPS/MIG (NVIDIA)** → _Too coarse_, partitions per model → wastes GPU resources.
        *   **Kernel-wise methods** → _Too fine-grain_, require new hardware or suffer **huge overhead**.
    *   Result: GPUs remain **underutilized**, wasting both **performance** and **energy**.
2.  **What ECLIP Delivers**
    *   **Kernel-group partitioning** on **existing AMD GPUs**:
        *   Splits GPU compute units (CUs) into smaller, optimized slices.
        *   Dynamically assigns these slices to kernels **from multiple models** without costly reconfigurations.
    *   Achieves:
        *   **13%↑ throughput**
        *   **25%↑ energy efficiency**
        *   **Low tail latency**
3.  **Key Insight**
    *   Full per-kernel partitioning → too much runtime overhead
    *   Full model partitioning → wastes capacity
    *   **ECLIP finds the middle ground** by grouping kernels and pre-allocating GPU resources.

* * *

**How ECLIP Actually Runs Multiple Models on One GPU**
------------------------------------------------------

Think of the GPU as a **60-lane highway** (60 CUs). Each model = cars needing lanes.

ECLIP does three things:

### **1\. Pre-allocates CU-Masked Streams**

*   Creates **7 GPU partitions (streams)** upfront, each using a different number of CUs.
*   This avoids **runtime calls** to reconfigure GPU partitions, which were previously too slow.

* * *

### **2\. Scheduler Assigns Kernels to Streams**

*   Each kernel from each model → assigned to a **pre-allocated stream** based on:
    *   How many CUs it needs (profiling data)
    *   Available GPU capacity
    *   Active workloads from other models
*   **Dependency enforcer** ensures correct order when kernels hop across streams.

* * *

### **3\. Resource Allocation Optimizer**

*   Runs **offline ILP optimization** to decide:
    *   How many CUs each kernel group gets
    *   When to switch partitions (limited to avoid overhead)
    *   Fairness so one model doesn’t hog the GPU

At runtime:

*   The scheduler simply **looks up precomputed allocations**.
*   Multiple models now share the GPU **without costly repartitioning**.

* * *

**Big Picture**
---------------

*   MPS/MIG: Entire highway → fixed slices per model → **unused lanes** if model doesn’t need them.
*   Per-kernel methods: Switch lanes every car → **traffic jams** (overhead).
*   **ECLIP**: Groups cars into convoys, assigns lanes upfront → **smooth flow, no waste**.

* * *

If you want, I can **draw a visual diagram** showing ECLIP splitting GPU compute units across multiple models in real time — it will make the scheduling logic super clear.

Do you want me to make that diagram?

## Prompt:
wouldnt this slow down running models then?

## Response:
That’s a great question — at first glance, **splitting a GPU across multiple models** sounds like it would slow them all down.  
But ECLIP is carefully designed so **it doesn’t significantly hurt performance** while boosting energy efficiency and throughput.

Here’s why:

* * *

**1\. Most ML inference workloads underutilize GPUs**
-----------------------------------------------------

*   Many ML inference models only use a **fraction of GPU compute units (CUs)** at a time.
*   Example: A small convolution kernel might only keep **20% of the GPU busy**.
*   The remaining **idle GPU resources** still consume power but do no work because GPUs can’t fully power down idle CUs mid-inference.

**ECLIP’s idea:**

> _Instead of leaving those idle resources wasted, give them to another model’s kernels._

So you’re not really slowing down one model — you’re using otherwise idle capacity.

* * *

**2\. ECLIP partitions at the _kernel-group_ level, not per-kernel**
--------------------------------------------------------------------

*   Switching GPU partitions per kernel would be slow because of:
    *   **IOCTL overhead** (55µs vs 20µs kernel runtime)
    *   **Barrier packet delays** for dependency enforcement
*   ECLIP groups multiple kernels together → fewer partition switches → **less overhead**

This keeps slowdown **much smaller** than naive kernel-wise partitioning.

* * *

**3\. Optimizer ensures fairness and minimal latency**
------------------------------------------------------

*   ECLIP runs **offline optimization** to decide:
    *   How many CUs to give each kernel group
    *   How often to switch partitions (limited to ~14 per request)
    *   How to **balance latency vs throughput vs energy**

It solves an **ILP problem** that ensures:

*   No one model monopolizes the GPU.
*   Tail latency stays **within acceptable limits**.

* * *

**4\. Results back this up**
----------------------------

From the paper’s experiments:

*   **Throughput:** ↑13% on average (up to 21%)
*   **Energy efficiency:** ↑25% on average (up to 35%)
*   **Tail latency:** Only slightly higher than baseline and **much lower** than naive kernel-wise partitioning.

So yes, **sharing introduces some interference**,  
but ECLIP **carefully schedules partitions** so that the net effect is **higher total throughput** with **acceptable latency trade-offs**.

* * *

Here’s a simple analogy:

*   **Old way:** One bus per lane → some lanes empty if buses are small.
*   **ECLIP:** Multiple small buses share lanes intelligently → more people moved per minute, even if one bus sometimes waits a bit longer.

* * *

If you want, I can make a **visual chart** comparing _baseline vs MPS vs ECLIP_ latency and throughput from the paper’s results to show the trade-offs clearly.

Do you want me to make that?

## Prompt:
Give me examples of when I would want to do this

## Response:
Got it — here are **realistic scenarios** where using something like **ECLIP** to co-locate multiple models on one GPU makes a lot of sense:

* * *

**1\. ML Inference Servers for Many Small Models**
--------------------------------------------------

*   **Scenario:**
    *   A cloud provider or internal inference server runs **many lightweight ML models** (e.g., NLP classifiers, vision models).
    *   Each model has **low GPU utilization** on its own.
*   **Why ECLIP helps:**
    *   Running each model on a separate GPU wastes compute and power.
    *   ECLIP lets you **share one GPU** across all models efficiently, boosting **requests per second** without buying more GPUs.
*   **Example:**
    *   A SaaS product offering **sentiment analysis, topic detection, and summarization APIs** — all inference-heavy, low-latency workloads.

* * *

**2\. Autonomous Vehicles & Robotics**
--------------------------------------

*   **Scenario:**
    *   A robot or self-driving car runs **multiple ML models** simultaneously:
        *   Object detection
        *   Lane detection
        *   Traffic sign recognition
        *   Pedestrian prediction
*   **Why ECLIP helps:**
    *   Each model runs periodically and **doesn’t fully saturate the GPU**.
    *   ECLIP can **co-locate models** to improve energy efficiency, crucial for **battery-powered systems**.
*   **Example:**
    *   A delivery robot with limited onboard power running **vision + speech + navigation models** on a single embedded GPU.

* * *

**3\. Edge AI Deployments**
---------------------------

*   **Scenario:**
    *   On the edge (IoT gateways, drones, retail cameras), hardware is limited.
    *   Multiple customers/models share **the same GPU**.
*   **Why ECLIP helps:**
    *   Prevents GPU idle time while keeping **latency predictable**.
    *   Reduces the number of GPUs needed → lowers **hardware and energy costs**.
*   **Example:**
    *   A smart city camera running **vehicle detection, pedestrian counting, and license plate recognition** all on one edge GPU.

* * *

**4\. Multi-Tenant Cloud ML Services**
--------------------------------------

*   **Scenario:**
    *   Different clients deploy **different ML models** on the same GPU server.
    *   Each tenant’s workload is bursty and variable.
*   **Why ECLIP helps:**
    *   Fairly allocates GPU resources **dynamically** across tenants.
    *   Guarantees **QoS (latency limits)** while **maximizing throughput**.
*   **Example:**
    *   AWS/GCP-style ML inference endpoints for multiple customers using one GPU server.

* * *

**5\. Research & Development Prototyping**
------------------------------------------

*   **Scenario:**
    *   A lab trains one big model but also runs **multiple small inference models** for experiments simultaneously.
*   **Why ECLIP helps:**
    *   Avoids over-provisioning GPUs for small models.
    *   Keeps energy costs down during large-scale experimentation.
*   **Example:**
    *   A university cluster where researchers share GPUs for **real-time ML inference tests**.

* * *

**6\. Batch + Real-Time Hybrid Workloads**
------------------------------------------

*   **Scenario:**
    *   Some workloads are **real-time inference**, others are **low-priority batch jobs**.
*   **Why ECLIP helps:**
    *   Low-priority jobs fill idle GPU cycles **without impacting real-time tasks** too much.
    *   Balances **energy efficiency** and **latency** for time-critical models.
*   **Example:**
    *   A fraud detection model running in real-time while **image embedding jobs** run in the background.

* * *

If you want, I can make a **decision flowchart** to help you see when co-locating models with ECLIP makes sense vs. when you should dedicate a GPU to one model.

Do you want me to create that flowchart?

## Prompt:
So why isn this taking off, what are the limitations and how do you think it can be alleviated?

## Response:
Ah, so you’re asking why something like **ECLIP** isn’t already standard practice, what’s holding it back, and what could be done to fix those issues.

Let’s break this down:

* * *

**1\. Why It’s Not Taking Off Yet**
-----------------------------------

Even though ECLIP shows **great performance gains**, there are practical reasons why GPU co-location at this level isn’t widely used:

### **A. Complexity of GPU Partitioning**

*   **Current ML stacks (PyTorch, TensorFlow, Triton)** assume each model owns the whole GPU.
*   Fine-grained partitioning requires changes to:
    *   GPU runtime (ROCm/CUDA)
    *   Inference server schedulers
    *   Model execution pipelines
*   Most developers prefer **simpler “one model per GPU” setups** for reliability.

* * *

### **B. Lack of Vendor Support**

*   **NVIDIA** dominates ML inference hardware.
*   Their official solutions (MPS, MIG) focus on **isolation & simplicity**, not **fine-grained sharing**.
*   AMD GPUs support CU masking, but **ML ecosystem adoption is small** compared to NVIDIA.

So ECLIP benefits haven’t reached **mainstream NVIDIA platforms** yet.

* * *

### **C. Overhead & Complexity Trade-offs**

*   Kernel-level partitioning adds:
    *   **Dependency tracking overhead** (barrier packets)
    *   **Repartitioning latency** if done too often
*   ECLIP solves this by pre-allocating partitions and using offline optimization,  
    but dynamic workloads might still break assumptions → **unpredictable latency** in production.

* * *

### **D. Difficulty with Multi-Tenant Fairness**

*   In cloud inference, multiple clients compete for GPU time.
*   **Performance isolation** is critical: one tenant can’t starve others.
*   Co-location makes **QoS guarantees harder** unless the scheduler is very sophisticated.

* * *

### **E. Hardware Power Gating Limitations**

*   Even if you partition GPUs, unused SMs/CUs often **can’t be fully power-gated** in current hardware.
*   So energy savings aren’t as big as they _could_ be until GPU vendors enable **finer-grained power gating**.

* * *

**2\. Potential Ways to Alleviate Limitations**
-----------------------------------------------

Here’s what could help make ECLIP-like systems mainstream:

| **Challenge** | **Potential Solution** |
| --- | --- |
| Lack of integration with ML stacks | Build **transparent schedulers** in PyTorch/Triton/ONNX runtime so users don’t change code. |
| Vendor ecosystem gap (NVIDIA vs AMD) | Implement **SM masking APIs for CUDA** (like libsmctrl but efficient & official). |
| Overhead & complexity in scheduling | Use **AI-driven dynamic schedulers** that adapt partitioning in real time. |
| QoS in multi-tenant settings | Integrate **SLAs & latency constraints** into partitioning optimization. |
| Hardware energy gating limits | Push GPU vendors for **finer-grain DVFS & power gating** capabilities. |

* * *

**3\. Why It Still Matters**
----------------------------

*   GPUs are getting **more powerful** (H100, MI300) → single model rarely saturates the whole chip.
*   **Energy costs** in data centers are exploding → sharing GPUs efficiently is becoming critical.
*   AI inference demand is **shifting from training to real-time inference** → _multi-model efficiency_ will matter more.

ECLIP shows a **realistic path**:

*   Works on existing hardware
*   No model code changes
*   Gains throughput & energy efficiency

But for it to “take off,” it needs **vendor support** + **tight integration** into inference runtimes.

* * *

If you want, I can give you a **future roadmap** of what ECLIP-like systems might look like if vendors like NVIDIA or AMD embraced this idea fully.

Do you want me to sketch that out?

1/2



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)