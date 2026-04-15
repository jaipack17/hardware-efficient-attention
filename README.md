# _Weaving One Thread at a Time_: Hardware-Efficient Attention for Long Context Inference

<div align="center">
<i><b>Jaikaran Singh</b> and <b>Ayush Kumar</b></i>
<br/>
<i>Indian Institute of Technology, Roorkee</i>
</div>

<br/>

As LLMs scale their context length to more than a million tokens, they run into major memory bottlenecks that significantly slow down inference. This bottleneck arises from the fact that modern GPUs only have a limited amount of VRAM and have comparatively slower memory read/write speeds than compute speeds. As a result, a naive implementation of attention in Transformers would likely run into the problem of suboptimal performance, poor load balancing in GPUs and insufficient memory due to a growing KVCache. This blog discusses methods to mitigate these issues by utilizing memory and GPU resources in an efficient and economical way. 

---

# Content

[1. Background](#1-background)
- [1.1. Attention Mechanism](#11-attention-mechanism)
- [1.2. Two Phases of LLM Inference](#12-two-phases-of-llm-inference)
- [1.3. KV Cache](#13-kv-cache)

[2. Under the Hood of a GPU](#2-under-the-hood-of-a-gpu)
- [2.1. GPU Architecture](#21-gpu-architecture)
- [2.2. Areas of Optimization](#22-areas-of-optimization)
- [2.3. KV Cache Management](#23-kv-cache-management)

[3. Hardware-Efficient Attention](#3-hardware-efficient-attention)
- [3.1. Tiling and Online Softmax](#31-tiling-and-online-softmax)
- [3.2. Optimal Work Partitioning](#32-optimal-work-partitioning)
    - [3.2.1. During Prefill Phase](#321-during-prefill-phase)
    -  [3.2.2. During Decode Phase](#322-during-decode-phase)
- [3.3. Asynchronous Execution and Pipelining](#33-asynchronous-execution-and-pipelining)
- [3.4. Reducing Memory Fragmentation](#34-reducing-memory-fragmentation)
- [3.5. Greater Arithmetic Intensity and Parallel Scalability](#35-greater-arithmetic-intensity-and-parallel-scalability)
- [3.6. Hardware Aligned Sparse Attention](#36-hardware-aligned-sparse-attention)
    - [3.6.1. About Sparsity of Attention and Sparse Attention Methods](#361-about-sparsity-of-attention-and-sparse-attention-methods)
    - [3.6.2. Why Naive Sparsity is Not Enough](#362-why-naive-sparsity-is-not-enough)
    - [3.6.3. Hardware Aligned Sparse Attention Methods](#363-hardware-aligned-sparse-attention-methods)
      
[4. Applications in Long-context Modelling](#4-applications-in-long-context-modelling)

[5. Conclusion](#5-conclusion)

---

# 1. Background
LLMs are used for a wide variety of tasks which may involve large prompts. Document summarization, question answering in a large document, codebase analysis, multi-modal inputs like videos, images, audio with large amounts of information etc. are few examples of long-context modelling scenarios. It’s of great importance to efficiently and quickly process large prompts and decode long responses while using GPU resources and memory effectively.

## 1.1 Attention Mechanism 
Attention in Transformers requires computing similarity scores between all queries and keys. Followed by a softmax operation to obtain probability distributions that are used in computing a weighted sum over the value vectors resulting in the attention output. This entire process is divided into 3 sequential steps, each of which is done parallely using GPUs.

$$
\huge{S = QK^T \in \mathbb{R}^{N \times N}}
$$

$$
\huge{P = \text{softmax} \left( \frac{S}{\sqrt{d_k}} \right) \in \mathbb{R}^{N \times N}}
$$

$$
\huge{O = PV \in \mathbb{R}^{N \times d}}
$$

In a standard implementation of this algorithm, the Q, K, and V ∈RNxd matrices reside in the GPU's off-chip High Bandwidth Memory (HBM). The process begins by transferring the Q and K matrices from the HBM to the chip to calculate the matrix S using GEMM (General Matrix Multiplication) kernels, which is then written back to the HBM. Subsequently, S is reloaded from the HBM to compute Pusing GEMM, and this result is once again stored in the off-chip memory. In the final stage, P and V are fetched from the HBM to produce the attention output O , which is then written back to the HBM for storage. Moreover P has to be saved to the HBM for the backward pass to compute gradients during training. 

The algorithm has a time complexity of $O(N^2 d)$ where N is the number of tokens and d is the head dimension. The memory complexity is $O(N^2)$ since we are using memory to store the intermediate matrices S and P in addition to the inputs and outputs of the algorithm. Since attention's computation and memory grow quadratically with sequence length, processing longer sequences becomes increasingly expensive. In a transformer, there are multiple attention layers with each layer having multiple attention heads. This means attention has to run for each head in each layer to generate a single token every time, where parallelization is possible across the number of heads dimension and batch size. Moreover, in the process described above, we are firstly materializing the NxN matrices in the global memory (which is just the order of 10s of GBs in GPUs) and repeatedly reading and writing these matrices from and to the HBM. If N was 1 million, this NxN matrix would be storing 1 trillion values, that would be just over 3,700 GBs if each value was a 32-bit floating point number! IO-operations on this amount of memory space would be slow, increasing the overall runtime of attention. To address these problems, FlashAttention was introduced in 2022 which we’ll talk more about later in the blog.

## 1.2 Two Phases of LLM Inference

Inference of most modern LLMs consists of two phases: prefill and decode. When the LLM is first given a prompt, it sees these prompt tokens for the first time. The model processes these tokens, computing the attention outputs layer by layer and at the end it generates the first output token. This phase of processing the input prompt to decode the first output token is called prefilling.

The decode phase involves generating the model response one token at a time. Each iteration of the model (one forward pass) generates a single token. Thus the decoding phase depicts the auto-regressive nature of LLMs. Each token can only look at itself and all the previous tokens when attention is computed, this is called causal masking. To avoid recomputing key and value matrices at every decoding step, they are cached in GPU memory (KVCache). As the KVCache grows with sequence length, the decoding stage becomes memory bound. The prefill phase is computationally very heavy since it requires computation of keys and values from scratch to first build up the KVCache and then compute attention for a large number of tokens at once. This makes the prefill phase compute bound. This introduces the scope of optimizing the prefill and decode phase, exploring the trade-off between compute and memory efficiency.

## 1.3 KV Cache

During the decode phase, at every step, the model needs the key and value vectors of all previous tokens to compute attention. Recomputing these from scratch at every step would be very expensive. To avoid recomputation at every step, we store or cache the key and value matrices of all previous tokens in the GPU memory. This is called KV cache.
For every token, the GPU stores two vectors i.e. key vectors and value vectors for each attention head in each layer. So the total memory occupied by the KV cache is:

$$
\huge{\text{Memory} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L_{\text{seq}} \times \text{bytes}}
$$

As you can see, memory scales linearly with sequence length. When the context is large, it becomes a serious problem. For a 30 billion parameter model with batch size 128 and sequence length 1024, the KV cache alone results in 180GB of memory usage. At 128K tokens on a single request with Llama 3 70B, it hits around 40GB. As context lengths push towards a million tokens, the KV cache dominates GPU memory, leaving little room for anything else and becoming the primary bottleneck in long-context inference. This introduces the scope of optimizing the prefill and decode phase, exploring the trade-off between compute and memory efficiency.

# 2. Under the Hood of a GPU
## 2.1 GPU Architecture

Modern GPUs have a hierarchical hardware architecture and execution models. They have multiple levels of memory (Memory hierarchy) where bandwidth and capacity are inversely related. The largest memory pool is the off-chip global memory (GMEM) also called the High-Bandwidth Memory (HBM) housing around 40-80GB VRAM. This memory is available to all Streaming Multiprocessors (around 100+ SMs in modern GPUs) but is relatively slow to read and write from.

<div align="center">
  <img width="60%" alt="image" src="https://github.com/user-attachments/assets/cfd5527e-fa12-491d-a7bf-e2647dec4d6f" />
  <br/>
  <a href="https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/"><i>Fig. GPU Memory Hierarchy of an NVidia A100 GPU.</i></a>
</div>

Each streaming multiprocessor (SM) is responsible for running a large number of parallel processing cores that compute matrix multiplications or arithmetic involving simple operations and functions. Each SM also has a small programmer-controlled shared memory (SMEM/SRAM) called the L1 Data Cache along with Registers that temporarily store data loaded from the HBM which can be used during computation where each core of the SM has direct access to this memory. This memory is really small (228KB per SM in H100 GPUs) but has significantly faster reads/writes compared to the HBM. As an intermediate stage between SRAM and global memory, GPUs also have an L2 Cache (~50-80 MB) but it cannot be directly controlled by the programmer.

The execution model of GPUs is governed by threads (cores) as the fundamental unit of computation, where each thread works parallely. A group of 32 threads (generally) that perform the same instructions on different data are called warps. Multiple warps constitute the thread blocks, and multiple thread blocks constitute a grid. An SM can run multiple threadblocks, but a threadblock can only run on a single SM. Instructions are distributed to different warps using the Warp Scheduler inside SMs. We can program how and what these threads do by writing kernel programs (code that runs on GPUs). 
That’s a lot of new terms at once, but we’ll soon see that there are numerous areas in which we can utilize this architecture to help us improve memory and compute efficiency in transformers.

## 2.2 Areas of Optimization

Firstly, to speed up attention, we can try to avoid reading, writing and storing large amounts of data in the HBM since it’s quite slow relative to compute speeds. We have to spend most of the time waiting for data to load before computation can run. Operations like softmax have no compute heaviness, they are mostly dependent upon data. If a set of operations are being performed on the same input, instead of loading the input and writing the output every time an operation has to be performed, we can load the input once, perform all operations, and then write all the outputs. This is called **kernel fusion**. 

Efficiently using the parallelism of GPUs would also be a great way to speed-up inference. Matrix-multiplication operations in GPUs are significantly (around 10 times more TFLOP/s) faster than simple arithmetic operations, thus trying to reduce the number of these non-matmul operations somehow would also lead to faster inference, since runtime would now mostly be controlled by mat-mul operations. Moreover, exploring parallelism of computation across different dimensions of data is also essential. This requires identifying those dimensions across which compute is independent in terms of whether data has to be shared along this dimension and if compute is largely dependent along this dimension. This helps in effective scheduling of instructions to different warps in the GPU, minimizing communication among different warps through the shared memory.  Decoding phase often suffers from GPU under-utilization which is a direct result of poor work partitioning across different warps, hence exploring this area is of great importance.

Latest Nvidia GPUs have new architectures like [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) and [Blackwell](https://www.nvidia.com/en-in/data-center/technologies/blackwell-architecture/) with immense capabilities like asynchronous computation, faster tensor cores and tensor memory accelerators which can be explored further to implement better kernels that partition work across GPU cores effectively (targeting high GPU utilization) and trying to overlap computation and memory access to hide the memory bottleneck.

Distributing the KVCache and Partitioning work across multiple GPUs using methods like tensor-parallelism and pipeline-parallelism would also be beneficial in overcoming the limits of a single GPU.

## 2.3 KVCache Management

Apart from improving GPU utilization, memory management and memory read/write techniques, we can also reduce how much data attention algorithms need in the first place. During decoding, the KV cache keeps growing with sequence length and needs to be accessed at every step, so it quickly becomes a major contributor to memory usage and bandwidth. To deal with this, we can either reduce the size of the KV cache using compression methods, or reduce the number of tokens accessed during attention using sparse attention methods. These ideas are usually introduced from an algorithmic perspective, but they can also be seen as ways to reduce memory movement and improve overall hardware efficiency. These observations motivate the need for hardware-aware attention methods which we explore in the following sections.

# 3. Hardware-Efficient Attention

In the past sections, we have discussed the areas where optimization is crucial to speed-up attention. We’ll now look at various methods starting with very fundamental optimizations that have been adopted in recent years to scale context lengths, speed up inference and improve long-context modelling performance, followed by new research that builds upon these methods.

## 3.1 Tiling and Online Softmax

Tiling is a technique that is used to reduce the number of memory read/writes between the GPU’s slow off-chip HBM and fast on-chip SRAM while computing attention scores. It was introduced in the [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) along with an online softmax technique that calculates softmax coefficients progressively. The motivation behind this technique is to make attention as compute-bound as possible while decreasing the memory overhead.

<div align="center">
  <h3>Stable Softmax</h3>
  <br/>
</div>

$$
\huge{\sigma(\mathbf{x}) = \left[ \frac{e^{x_1 - M}}{\sum_{j=1}^{n} e^{x_j - M}}, \dots, \frac{e^{x_n - M}}{\sum_{j=1}^{n} e^{x_j - M}} \right], \text{ where } M = \max_j x_j}
$$

We divide the Q, K, V matrices into blocks of equal size where each block contains a consecutive set of vectors (corresponding to some set of consecutive tokens in a sequence). These blocks reside on the HBM. Alongside this, on the chip memory, we allocate space for the attention output $O\in \mathbb{R}^{N \times d}$ along with N dim vectors $m$ and $l$ for storing block-wise max and normalisation factors.

<div align="center">
<img width="60%" alt="a6334619b9545c90db128d43f6af9475 (online-video-cutter com)-2" src="https://github.com/user-attachments/assets/875dc412-2e65-491e-bd80-8614f945173a" />
  <br/>
  <p><i>Fig. Visualization of FlashAttention-2 Kernel for one query block</i></p>
</div>

Then, compute attention block-wise on shared memory. For each query block ($Q_i$) loaded from HBM, we sequentially go through all key blocks computing a small chunk of the $S\in \mathbb{R}^{NxN}$ matrix on-chip. Since a comparison of a single $Q_i$ block with all $K_j$ blocks maps to the $O_i$ output block, we have to calculate blocks $m_i$ and $l_i$ for storing the row-wise maxes and row-wise normalising factors (the big sum in the softmax denominator). $m_i$ and $l_i$ are updated as $Q_i$ is compared with more K blocks, eventually obtaining the full softmax coefficients. 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/ae345cd8-e99b-4f04-b93e-0cabae12a6ad" />
  <br/>
  <p><i>Fig. Algorithm of the FlashAttention-2 kernel</i></p>
</div>

We can apply causal masking by setting all $S_{ij}$ with $j>i$ to $-\infty$ and skipping its computation. It is possible to combine $m$ and $l$ into a single vector at the end using the _logsumexp_ trick of softmax. This was a key optimization in [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691). 

The biggest advantage is that tiling eliminates the need to store, read or write $S\in \mathbb{R}^{NxN}$  in the HBM repeatedly. It only needs to store a single N dimensional vector aside from the inputs and outputs. This causes $O(N)$ memory complexity! All computations are performed on-chip in the SMs. The computation for attention scores $O$ of all query blocks are done parallely. This ensures parallelism across the sequence length dimension in addition to parallelism along batch dimension and number of heads dimension. Moreover, for the backward pass we just need the attention output $O$ and softmax coefficients to recompute matrix $P$ for calculating gradients. This requires more FLOPs but significantly less IO-operations, speeding up training as well. FlashAttention kernels have been widely used in research and real world LLMs.

## 3.2 Optimal Work Partitioning

Writing kernels allows us to choose how a big computational task can be divided and distributed to different compute units (threads) in a GPU. In CUDA, this is possible with thread indexing where each thread effectively runs the same kernel but on different data. If you parallelise at a very fine-grained level, there won’t be much of a speed-up because thousands of threads will have very little compute work individually. If you don’t parallelise at all, there would be a lot of idle threads which could have been utilized to complete the task faster. Thus, our goal is to find the optimum work partitioning strategy. 

### 3.2.1 During Prefill Phase

In [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691), each head is processed using 1 thread block, with a total of $number of heads \times batch size$ thread blocks. Additionally, different query blocks are also scheduled on different thread blocks. This ensures parallelisation across the query length dimension alongside batch and number of head dimensions. 
In CUDA, a kernel is launched on a grid of thread blocks. The GPU scheduler allows this grid to be up to 3-dimensional, where each coordinate maps to a specific piece of the output tensor. Suppose we have the $z$ coordinate of a grid indexing the batch, $y$ coordinate indexing the head, and $x$ coordinate indexing the query block. This launches a total of `gridDim.x * gridDim.y * gridDim.z` thread blocks.

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/9d207ac0-5e81-4242-949a-98cbabf38780" />
  <br/>
  <p><i>Fig. Grid and Thread Indexing</i></p>
</div>

Thread block `(3, 0, 1)` will fetch its query block and KV blocks of head 0 from the HBM. Within each threadblock, the query block is split across 4-8 warps while keeping the KV blocks accessible to all warps (internal work partitioning). The tiled attention output can be computed parallely in these warps. This helps in effectively partitioning independent work to different parallel workers, making FlashAttention-2 the fastest and most efficient exact attention algorithm for the prefill phase. 

### 3.2.2 During Decode Phase

Decoding takes significantly more runtime than prefill even when the prompt sequence is very long, thus GPUs must work efficiently at near full occupancy to yield the fastest decoding results.  The query length ($N_q$) in the decode phase is equal to 1 as discussed in [Section 2.1](#21-gpu-architecture). As a result, FlashAttention-2 runs into the problem of severe GPU under-utilization. Since parallelization is not present across the context length dimension ($N_k$ : number of keys), the GPU has to compute tiled attention for just a single query sequentially by looping over all key blocks. This means thread blocks are scheduled per batch per head, leaving the remaining thread blocks idle. To solve this, we need a partitioning scheme that is able to divide the KV blocks into chunks that can be processed in parallel for the single query. 

LLM serving frameworks like [FlashDecoding (2023)](https://princeton-nlp.github.io/flash-decoding/) and [FlashInfer (2025)](https://arxiv.org/abs/2501.01005) are built upon the same strategy of tiling in FlashAttention-2 but with a focus on decoding efficiency by parallelising work across the context length dimension using a **fixed-split** partitioning strategy. 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/ca010821-88b4-4491-9592-85f6b8393567" />
  <br/>
  <p><i>Fig 1 from <a href="https://arxiv.org/abs/2405.10480">LeanAttention (2025)</a> : Execution Schedule of FlashAttention-2, FlashDecoding/FlashInfer, and LeanAttention across a hypothetical five SM GPU executing attention of 2 heads.</i></p>
</div>

The fixed-split partitioning scheme in FlashDecoding/FlashInfer is inspired by the fixed-split GEMM decomposition scheme where the matrix multiplication of $A\in \mathbb{R}^{NxK}$ and $B\in \mathbb{R}^{KxM}$ to give $C=AB \in \mathbb{R}^{NxM}$  is batched across the K dimension of A and B. To compute a single value in C, we need to find the inner product of a row of A and a column of B. If K is large, finding this inner product will take more time, and hence we partition the matrices A and B into T parts along the inner dimension K. These partitions are then multiplied together to produce small chunks of C by scheduling them on different threadblocks. At the end, these chunks are combined together to obtain C. Due to more active threadblocks, GPU utilization is higher and C is calculated faster for a suitable value of T. 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/ef78997f-2124-4fc3-9d89-3d090fd78fcd" />
  <br/>
  <p><i>Fig. Fixed-split (split-K) Partitioning Scheme of GEMM</i></p>
</div>

In the case of Attention during decoding, $Q\in \mathbb{R}^{1xd}$, $K\in \mathbb{R}^{Nxd}$, $S=QK^T\in \mathbb{R}^{1xN}$, $P=softmax(S)$ and $O=PV \in \mathbb{R}^{1xd}$. Here computing O can be optimized with fixed-split work partitioning in addition to tiling. However, using equally sized chunks for dividing work across different thread blocks can run into problems like load imbalance if T does not divide N (quantization inefficiency). Moreover, depending on the number of SMs, there can be load imbalancing in the last wave of computation of tiles (as shown in the figure below). This leads to some idle resources which could still be used to improve performance. [LeanAttention (2025)](https://arxiv.org/abs/2405.10480) proposed a new way to distribute workload during decoding with the help of a [Stream-K](https://arxiv.org/pdf/2301.03598) style partitioning scheme, achieving near 100% GPU occupancy in a variety of attention workloads. The following figure shows the difference in the quantization efficiency of fixed-split and stream-K decomposition.


<div align="center">
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/83353dd5-0252-47f7-8895-a90163a78d9f" />
  <br/>
  <p><i>Fig 2 from LeanAttention (2025) : Tile-splitting execution schedules for 384x384x128 GEMM across a hypothetical four-SM GPU.</i></p>
</div>

Consider the same matrix multiplication $C=AB$ as earlier. Consider $C (384x384) = A (384x128) \times B (128x384)$, and let the smallest unit of mat-mul computation be of the form $(128\times4)x(4\times128)$. Here each tile size (9 tiles total) is $128 \times 128$. Thus, for computing a single tile, we need $\frac{128}{4}=32$ MAC loop iterations. For 9 tiles, $32\times9=288$ MAC loop iterations are performed. The MAC-iterations are performed in the order of tile-row (m) $\rightarrow$ tile-column (n) $\rightarrow$ tile partition (k). This way, we divide the computation of $C$ into tiles as earlier. But instead of distributing computation of each tile equally across different threadblocks, we consider tile computation as a stream of MAC-operations, and distribute these MAC-iterations over different threadblocks. Suppose T=4 number of threadblocks are available (where each threadblock runs on a single SM), we can distribute $\frac{288}{4} = 72$ MAC loop iterations to each threadblock (100% occupancy in a single wave). Thus we can compute entire tile 0, tile 1 and 8 MAC loop iterations for tile 2 on a single SM. Then the remaining 64 iterations of tile 2 on another SM and so on. Tiles can thus be divided into unequal sizes across threadblocks and their output can be synced by the SM that computed the first partition of a tile. This provides efficient load balancing for the vast majority of GEMM operations.

In LeanAttention the smallest granularity of the KV blocks is called a **LeanTile**, whose size has to be found experimentally on different hardware architectures. A single LeanTile iteration refers to the computation of local attention (un-normalized) between the small subset of tokens in this tile. In the stream-K style execution, an equal number of LeanTiles are distributed to different thread blocks in the order of batch $\rightarrow$ head $\rightarrow$ context-length linearization. This is done by first calculating the total number of output tiles ($\frac{number of queries}{tile size}), then the number of iterations per output tile (i.e. number of LeanTiles). The total number of iterations (LeanTiles) are then equally divided between all threadblocks. If LeanTiles of a head are distributed to different blocks then the partial outputs from each block are consolidated by the threadblock that processed the first LeanTile. This requires communication between thread blocks through global memory (though the overhead is negligible). This synchronisation is possible through online softmax (or softmax re-scaling operation) which was discussed in [Section 3.1](#31-tiling-and-online-softmax). This way, we are able to achieve near 100% SM occupancy and efficient quantization of the workload.

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/37f88b55-113b-4672-950d-b4a39cd503c2" />
  <br/>
  <p><i>Fig. LeanAttention (2025) Section IV</i></p>
</div>

## 3.3 Asynchronous Execution & Pipelining

NVidia introduced the [Hopper GPU Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) in 2022 with asynchronous execution features including a new Tensor Memory Accelerator (TMA) unit that can transfer large blocks of data efficiently between global memory and shared memory independently in the background while computation is being done. This paved new improvements to the existing hardware optimizations in attention algorithms by overlapping data loads/writes with computation by using TMA and warp-specialization.

<div align="center">
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/82fc430d-8583-4dc2-aa6a-504e07daa905" />
  <br/>
  <p><i><a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#asynchronous_execution">Fig. Asynchronous execution concurrency in NVIDIA Hopper</a></i></p>
</div>

The advantage of the TMA is that it frees the threads to execute other independent work, unlike in the Ampere architecture where threads were used to do tedious memory handling work. Only a single thread is required to issue asynchronous instructions (copy descriptor) of what to load, to the TMA. The TMA then does all the work of generating addresses and moving data.

<div align="center">
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/41ad6939-adab-4a6d-830a-423f58c65a45" />
  <br/>
  <p><i><a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#asynchronous_execution">Fig. Asynchronous memory copy using TMA</a></i></p>
</div>

To improve GPU resource utilization in FlashAttention-2 for the Hopper Architecture (eg. H100), [FlashAttention-3 (2024)](https://arxiv.org/abs/2407.08608) was introduced. It uses asynchronous execution at 2 levels of the attention algorithm. Recall that in FlashAttention-2, the outer-loop parallely goes over different query blocks and the inner-loop goes sequentially through the KV blocks. We would load the Query, then in iteration we would load KV blocks, perform GEMM and softmax operations ([Section 3.1](#31-tiling-and-online-softmax)). An effective way to parallelise the loading of the KV blocks for a query and computation of the attention output is to specialise different warps to either work on loading/writing data (producer warpgroups) or computing attention (consumer warpgroups). 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/5b108013-ff61-4e54-86df-9abe8c758ef7" />
  <br/>
  <p><i>Fig. Warp-specialization helps overlap compute and memory loads which speeds up iterations since memory loads of iteration j have already been done in iteration j-1.</i></p>
</div>

The producer warp groups use a circular SMEM buffer with ‘k’ stages. This means the KV blocks for at most k iterations can be loaded alongside each other. If the 
$(j mod k)^{th}$ stage of the buffer is occupied, the KV blocks of the $j^{th}$ iteration have to wait for the computation of the KV blocks in this space to be completed, and this space to be freed from the buffer. These producer warps issue instructions to the TMA for loading these queries and kv blocks asynchronously, and once these tensors are loaded, the consumer warp groups are notified for a particular iteration.

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/c63a1b67-2dba-433e-9500-d1062b070953" />
  <br/>
  <p><i>Fig. Circular SMEM Buffer with k=4 stages holding at most 4 KV Blocks at a time.</i></p>
</div>

The consumer warps wait until the required query block has been loaded into the shared memory before running iterations on the kv blocks. At every $j^{th}$ iteration, the warps wait for block $K_j$ to load, GEMM is performed to compute matrix S using the async Warp Group Matrix Multiply Add (WGMMA) instruction. Then after computing softmax outputs, the warps wait for block $V_j$ to load followed by updating attention output for the query. This is similar to tiling, but here we have different warps for loading blocks and computing outputs. 

In addition to this, we can see that in the consumer warps we have a sequential and dependent computation process where a GEMM is followed by non-mat-mul softmax operations followed by another GEMM. GEMM operations are computed way faster than softmax operations (which involve the exponential function) because of the high-speed tensor cores. This means a big portion of the time to compute attention is taken up by these softmax operations. We can try to overlap the softmax computation of one warpgroup with the GEMM of another warpgroup. FlashAttention-3 uses synchronization barriers of the CUDA API to schedule the GEMMs (GEMM1 - $PV$ of one iteration, and GEMM0 - $QK^T$ of the next iteration) of warpgroup 1 before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs. This is called **ping-pong scheduling**, best explained in the FlashAttention-3 paper’s Section 3.1.

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/c6200827-4758-4f30-a58d-817d28d04525" />
<br/>
  <p><i>Fig. FlashAttention-3 Section 3.1</i></p>
</div>

Even within each warpgroup, it is possible to overlap the softmax and GEMM operations by pipelining across different iterations using more buffers (cyclic) in registers. 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/a041a2f6-0482-4b07-8672-7a698c9195d1" />
<br/>
  <p><i>Fig. FlashAttention-3 Section 3.2</i></p>
</div>

Since this method requires additional registers for storing $S_{next}$, for higher block sizes it is possible that high register pressure leads to register spilling which hurts performance. It’s thus important to find the optimum amount of parallelism to maximize performance. A lot of these asynchronous execution techniques are used in FlashInfer (2025) to make FlashAttention-3 efficient for inference workloads. They utilize the TMA to hide the latency of KVCache fetching behind their exact attention computations. There exist some limitations to the TMA like the lack of support of non-affine memory access patterns which doesn’t allow asynchronous fetching of sparse KVCache blocks. Thus for sparse attention, FlashInfer relies on the normal Ampere architecture style transfer of data from global memory to shared memory.

## 3.4 Reducing Memory Fragmentation

Before **PagedAttention**, systems like [Orca (2022)](https://www.usenix.org/conference/osdi22/presentation/yu) and [FasterTransformer (2022)](https://github.com/NVIDIA/FasterTransformer) allocated KV cache memory statically, as a contiguous 4D tensor of shape [B,L,H,D] where L is the maximum context length the model supports. For a model like Yi-34B that supports 200K tokens, this means reserving 200K worth of KV cache per request in the batch, even if the actual request generates only 415 tokens on average. This causes severe internal fragmentation where most of the reserved memory sits unused, directly limiting how many requests can be batched simultaneously and therefore limiting throughput.

Inspired by OS virtual memory and demand paging, [vLLM (2023)](https://arxiv.org/abs/2309.06180) introduced PagedAttention. Instead of one contiguous block per request, the KV cache is split into fixed-size blocks called pages, each storing KV data for a fixed number of tokens (typically 16-32). A block table per request maps logical token positions to physical memory locations, exactly like a page table in an OS. 

<div align="center">
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/a2e00513-c7b3-4a37-8b8c-f67dc7e9dce5" />
<br/>
  <p><i>Fig. Efficient Memory Management for Large Language Model Serving with PagedAttention Section 4.2</i></p>
</div>

Memory is only allocated when a previously allocated block fills up and the model needs to continue generating, not upfront. This reduces fragmentation from 60-80% to below 4%. PagedAttention also enables memory sharing across requests. Sequences sharing a common prefix like a system prompt can share the same physical KV blocks via copy-on-write, only duplicating when a sequence diverges. PagedAttention solves physical memory fragmentation but introduces a new problem in the process. It makes the KV cache non-contiguous in virtual memory. Conventional attention kernels assume K and V are contiguous tensors, and PagedAttention breaks this assumption, requiring every attention kernel to be rewritten to handle non-contiguous block-based memory access. This creates three concrete problems: kernel rewrites add implementation complexity, a memory manager must be maintained in the serving framework to stitch together virtual memory blocks, and block table lookups add runtime overhead on both CPU and GPU. The empirical evidence is stark. vLLM's PagedAttention kernel is up to 2.8x slower than FlashAttention-2's standard kernel. FlashAttention-2's paged prefill kernel is up to 37% slower than the non-paged version. FlashInfer's paged prefill kernel is up to 42% slower. FlashAttention-3 for Hopper was not even released with PagedAttention support at all.

[vAttention (2025)](https://arxiv.org/abs/2405.04437) addresses these issues by leveraging [CUDA Virtual Memory Management (VMM) APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html) to decouple virtual and physical memory allocation. Its core insight is that the fragmentation problem lives in physical memory, not virtual memory, so there is no reason to break virtual memory contiguity to fix it. This is exactly how OS demand paging works, but GPU runtimes historically did not expose this capability. It reserves a large contiguous virtual address space for the KV cache upfront so kernels see contiguous memory and need no modification, but only commits physical memory pages on demand as tokens are generated. When a request completes, physical pages are released while the virtual address reservation stays intact for reuse.

<div align="center">
<img width="75%" alt="image" src="https://github.com/user-attachments/assets/ee61ba29-5a8a-4109-a912-f4f9ed246038" />
</div>

Two key engineering challenges had to be solved. First, CUDA VMM API calls involve a round trip to the OS kernel which adds latency. vAttention hides this by overlapping memory allocation with compute (**asynchronous execution**), opportunistically pre-allocating pages slightly ahead of when they are needed, and deferring reclamation. Second, CUDA only supports allocation at 2MB page granularity by default, which itself causes fragmentation for small requests. vAttention modifies the open-source CUDA unified virtual memory driver to add support for 64KB pages.

## 3.5 Greater Arithmetic Intensity and Parallel Scalability

Even with perfect SM occupancy, the decoding step is fundamentally memory-bound because every token generation requires loading the entire KV cache from HBM. The arithmetic intensity of standard MHA during decoding is roughly 1 FLOP per byte, far below the H100's compute roofline of ~295 FLOPs/byte, meaning GPU utilization can drop as low as 7%.  Hardware FLOPs have scaled by ~3x every two years while HBM bandwidth only grows at ~1.6x over the same period, so this gap is only widening.

Prior inference-aware attention variants tried to address this. [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA) reduces the KV cache to a single shared head across all query heads, boosting arithmetic intensity by approximately the number of heads ($h_q$) but at the cost of model quality. [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA) groups query heads to share a KV head, improving arithmetic intensity proportionally to the group size ($g_q$​) while scaling efficiently across devices, but with a moderate tensor parallelism degree each GPU still stores a sizable KV cache. [Multi-head Latent Attention](https://arxiv.org/abs/2502.07864) (MLA), introduced by DeepSeek, compresses each token's hidden state into a low-rank latent vector $c^{KV}$ of dimension $d_c=4d_h$​, caches only this vector, and absorbs the up-projection matrices $W^{UK}$​ and $W^{UV}$​ into the query and output projection matrices respectively during decoding. This means keys and values never materialize explicitly and attention is computed directly against the cached latent, effectively doubling arithmetic intensity relative to MQA. However MLA's single latent head cannot be sharded across devices, it must be replicated on every GPU, limiting parallel scalability.

[Hardware-Efficient Attention for Fast Decoding (2025)](https://arxiv.org/abs/2505.21487) directly addresses this gap by redesigning attention to do more computation per byte loaded from memory, without sacrificing model quality or parallel scalability. The paper defines group size $g_q$ as the number of query heads per distinct KV head, which largely determines arithmetic intensity. Two variants are proposed.
- Grouped-Tied Attention (GTA), ties the key and value states into a single shared projection. A single projection $W_{KV}$​ produces one tied vector used as the value. For the key, half the head dimension comes from this tied vector with no positional encoding applied, and the other half comes from a separate single-head projection where [RoPE](https://arxiv.org/abs/2104.09864) is applied and broadcast to all heads in the group. This halves the KV cache and doubles arithmetic intensity relative to GQA at the same group size from approximately $g_q​$ to $2g_q$, while preserving GQA's grouping structure for efficient multi-device sharding.

- The second, Grouped Latent Attention (GLA), extends MLA by splitting its single large latent into $m_{kv}$​ smaller heads each of dimension $2d_h/m_{kv}$​. For GLA-2 where $m_{kv}=2$, two latent heads of dimension $2d_h$​ are cached instead of one of dimension $4d_h$​. As shown in the figure below, different devices can now be shared over different latent heads without replication, fixing MLA's parallelism problem while maintaining arithmetic intensity of approximately $2g_q$​.

<div align="center">
  <img width="65%" alt="image" src="https://github.com/user-attachments/assets/9d4afb1f-26af-43bf-ab28-f42c5422f03f" />
</div>

Both variants use asynchronous software pipelining and warp specialization similar to FlashAttention-3 to overlap compute with memory transfers, and a cooperative offset calculator for paged KV, keeping tensor cores fully loaded and pushing the kernels from memory-bound towards compute-bound. GLA matches MLA quality and the optimized GLA kernel is up to 2x faster than [FlashMLA](https://github.com/deepseek-ai/FlashMLA) in speculative decoding. In online serving benchmarks, GLA reduces end-to-end latency and increases throughput by up to 2x. However, GTA's tying of keys and values is a strong constraint that may hurt tasks where keys and values benefit from capturing different aspects of the input. GLA's quality advantage over MLA narrows at smaller model scales. The speedup over FlashMLA (the optimized MLA kernel) is most pronounced when query length exceeds 1, meaning single token standard decoding sees less benefit compared to speculative decoding settings.

## 3.6 Hardware Aligned Sparse Attention

### 3.6.1 About Sparsity of Attention and Sparse Attention methods

Analysis of attention patterns in transformer models has shown that attention distribution in transformers is heavily skewed. Only a small subset of tokens actually contributes to most of the attention mass. This behaviour can be visualized via attention heatmaps across layers and heads. 

| <img src="https://github.com/user-attachments/assets/73d142e6-1e1c-4fca-9ad1-775c588aa124" width="150" /> | <img src="https://github.com/user-attachments/assets/97e221e2-2476-43b9-856c-f851aba6ccbe" width="150" /> | <img src="https://github.com/user-attachments/assets/1aee6b26-2c5f-416e-a8aa-fbbface3ae46" width="150" /> | <img src="https://github.com/user-attachments/assets/eae7f34f-d7e8-49a7-ae70-4c9ed47e6521" width="150" /> |
| :---: | :---: | :---: | :---: |

<div align="center">
  <br/>
  <p><i>Fig. Attention Heatmaps of different heads in GPT-2 for an example sequence</i></p>
</div>

As seen in the heatmaps, attention is not uniformly distributed across tokens. Instead most of the tokens have attention close to zero with only a few regions where certain tokens receive significantly higher attention. This indicates that only a few tokens contribute meaningfully to the attention computation. This observation motivated sparse attention methods where instead of calculating the attention scores for all the token pairs, we only calculate them for a subset of the relevant tokens.
- A common approach is to directly estimate token importance and select only a subset of tokens for each query. [TokenSelect (2024)](https://arxiv.org/abs/2411.02886) works at the token level and tries to directly identify important tokens instead of relying on blocks. It computes query-key scores for each head and then combines them using a soft voting scheme, where each head contributes to the importance of a token. This avoids domination by a few heads and leads to a more balanced selection. It also reuses previous selections during decoding by exploiting similarity between consecutive queries, which reduces overhead. The result is a reasonably precise and adaptive selection mechanism, but since it depends on current query-key scores, it can still miss tokens that become important later.
- [xAttention (2025)](https://arxiv.org/abs/2503.16428) operates at the block level. Instead of scoring individual tokens, it divides attention into blocks and estimates block importance using a trick i.e. summing the anti-diagonal values of each block. Blocks with higher sums are treated as more important and selected. This avoids expensive scoring mechanisms and keeps things efficient, but the selection is still approximate and depends on how well this proxy captures true importance.
-  [MoBA (2025)](https://arxiv.org/html/2502.13189v1) by DeepSeek also works with blocks but introduces a more dynamic selection process. It divides queries and KV cache into fixed-size blocks and computes an affinity score between a query and each block using inner products with pooled key vectors. Based on this, it selects the top-k blocks for each query and computes attention only within those blocks. This is inspired by mixture-of-experts style routing and allows different queries to attend to different regions. However, it works best in prefill and often requires switching back to full attention in later stages.

### 3.6.2 Why Naive Sparsity is Not Enough

The sparse attention methods reduce the number of tokens each query attends to, lowering theoretical FLOPs. But it does not automatically translate to real speedups. How the modern GPU hardware is working determines the gap between how much these sparse attention methods save on paper and how much they actually save in practice. GPUs load data from HBM in contiguous chunks. When naive sparse attention selects arbitrary individual tokens to attend to, those tokens are scattered across memory. The GPU still has to load the entire memory block containing that token even if it only needs one value from it, so you don't actually reduce memory traffic proportionally to the number of tokens you skip. Real speedup only comes when you skip entire contiguous blocks of tokens at once, because then you can skip entire memory accesses. This is why block-sparse attention is hardware-aligned but token-level sparse attention is not.

Most sparse attention methods work either only during the prefill stage (like [MInference (2024)](https://arxiv.org/abs/2407.02490)) or during the decode stage (like [H2O (2023)](https://arxiv.org/abs/2306.14048)). Therefore, these methods only apply sparsity to a single phase of the inference stage. At least one phase remains at full cost compared to full attention.  Post-training sparsity applied to a pretrained dense model forces deviations from the optimization trajectory. The model was never trained to work with sparse attention, so you lose performance. Top 20% attention by score only covers 70% of total attention weight so the “important" tokens you keep are not the same as what the model actually needs.

Most models are still trained on full causal attention, and sparse attention is usually only used during inference. This creates a mismatch between what models have actually learned and how models are actually working at test time. Together, these limitations reveal that sparsity alone is not enough. What matters is whether the sparsity pattern is structured, trainable, and aligned with how the hardware actually processes memory.

### 3.6.3 Hardware Aligned Sparse Attention Methods

The following methods address these gaps by designing sparsity with hardware constraints and training in mind from the start. 

We first look at methods that combine hierarchical sparse attention with native training support. [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (2025) (NSA) by DeepSeek does this by first doing a course-grained token compression, the key sequence is divided into blocks of length $l=32$ with a sliding stride $d=16$ so blocks slightly overlap, preventing information loss at boundaries, each block of keys is passed through a learnable MLP $\Phi$ with intra-block positional encoding to produce a single compressed key that summarizes the entire block, with an analogous compression for values. The query then attends over these compressed representations. After that, it does a fine-grained token selection based on these compressions, NSA uses the attention scores from the compressed branch to estimate which original blocks are most important to the current query. The top-n blocks ($n=16$, including 1 initial block and 2 local blocks) are selected and retained in their original uncompressed form with block size $l′=64$. The query then attends to these selected blocks in original detail. This is how fine-grained, query-specific information is recovered that compression might have lost. Critically, blockwise selection is used rather than token-level selection since GPUs have much higher throughput for contiguous block access than random index-based reads, and it enables Tensor Core utilization. The newest $w=512$ tokens are kept as a standard local attention window. This handles the immediate local context and it would otherwise dominate the other branches if not separated. The outputs of all three branches are combined via a learned gating MLP and a sigmoid activation, that takes the query as input and produces three gate scores, one for each branch. The final output is a weighted sum of the output of each branch multiplied by the gate score for each branch.

<div align="center">
  <img width="75%" alt="image" src="https://github.com/user-attachments/assets/e154f4cf-4642-44df-838a-6a2cdc801f85" />
  <br/>
  <p><i>Fig. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. Section 1.</i></p>
</div>

Standard sparse attention has low arithmetic intensity because you load large blocks of KV but skip computation, wasting memory bandwidth. NSA's kernel design uses tiling ensures all loaded KV data is actually used by loading queries by GQA groups so all query heads in a group share the same KV load. This maximizes the FLOPs per byte of memory access.

<div align="center">
  <img width="65%" height="305" alt="image" src="https://github.com/user-attachments/assets/4201a745-9b1d-4b23-8ef0-b47efaee8cfb" />
  <br/>
  <p><i>Fig. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. Section 3.4 Kernel Design.</i></p>
</div>

NSA provides efficient backward pass operators for both the compression MLP and the blockwise selection, enabling gradients to flow through the entire sparse attention mechanism. This is what makes NSA natively trainable rather than a post-training inference-only method. NSA matches or exceeds full attention on general benchmarks, [LongBench](https://arxiv.org/abs/2308.14508), and chain-of-thought reasoning. However, the three-branch design adds implementation complexity. The compression MLP adds a small amount of extra parameters and compute overhead. Performance advantage over full attention narrows at shorter sequence lengths.

Another method that does this is [Trainable Dynamic Mask Sparse Attention (2025)](https://arxiv.org/abs/2508.02124) (DMA) first identifies that long-context tasks naturally exhibit three types of sparsity. Copy tasks need positional sparsity (fixed-distance relationships), select tasks need content sparsity (semantically relevant tokens), and induce tasks need associative sparsity (query-relevant KV pairs). Rather than imposing one fixed pattern, this method learns to identify and leverage all three. Unlike NSA which uses attention scores to select blocks, DMA generates masks from the value matrix V. Dynamic weights are computed as: 

$$
\huge{\delta = e^{(\tau(v \cdot \Delta) \times A)} \quad \text{where} \quad \Delta \in \mathbb{R}^{n_h \times d_h \times n_h} \quad A \in \mathbb{R}^{n_h} \quad \delta \in \mathbb{R}^{n_h \times n}}
$$

where $\Delta$ is a learnable sampling weight matrix acting like a forget gate, $A$ is the per-head gating co-efficient, $\tau(v \cdot \Delta)$ is a non-negative activation function ensuring weights emphasize rather than suppress signals and $\delta$ contain the final scores. The final mask combines these dynamic weights with causal masking: 

$$
\huge{
\begin{aligned}
m_t &= f(\delta) \quad \text{where} \quad m_t \in \mathbb{R}^{n\_h \times n} \\\\
&= \begin{bmatrix} 
f(\sum_{j=1}^{n} \delta_{1,j}) \\\\ 
f(\sum_{j=1}^{n} \delta_{2,j}) \\\\ 
\vdots \\\\ 
f(\sum_{j=1}^{n} \delta_{n\_h,j}) 
\end{bmatrix} \quad \text{where} \quad f(\delta_{n\_h,j}) = 
\begin{cases} 
\delta_{n\_h,j} & \text{if } \delta_{n\_h,j} \in top_w(\delta_{n\_h}) \\\\ 
-\infty & \text{otherwise} 
\end{cases}
\end{aligned}
}
$$

Scores within the $top_w$ are retained to preserve the gradient flow, while all other scores are set to $-\infty$, effectively nullifying their contribution in the subsequent softmax operation. This creates a unique mask per attention head, enabling diverse patterns across heads. Attention is then computed as:

$$
\huge{
o_t = \text{softmax}\left( \frac{q_t k^\top}{\sqrt{d_h}} \circ m_t \right)v \quad \text{where} \quad p_t \in \mathbb{R}^{n\_h \times n} \quad o_t \in \mathbb{R}^{n\_h \times d\_h}
}
$$

When mask values are $-\infty$, softmax outputs exactly zero, so those computations can be completely skipped. DMA proves this is safe for both forward and backward passes as gradients are also exactly zero for masked positions, so gradient flow is fully intact. This reduces FLOPs from $O(n^2d_h)$ to $O(nwd_h)$. From 80M to 1.7B parameters, DMA consistently achieves better perplexity than MHA, Sliding Window Attention (SWA), MLA, and NSA. Up to 10x speedup at long sequences. Superior performance on needle-in-a-haystack and multi-query associative recall at 512 KV pairs. However, the mask generation from value vectors adds a forward pass dependency that complicates parallelism. Performance advantage over NSA is modest on standard benchmarks. The top-w window size is a fixed hyperparameter that may not adapt optimally across all task types.

Another idea is to unify static and dynamic sparsity for serving. [LServe (2025)](https://arxiv.org/abs/2502.14866) is a method that does it, it builds upon a key observation that when Llama-3-8B runs with 256k input tokens and 20k output tokens, prefill takes 116 seconds but decode takes 540 seconds, almost 5x longer. So you need to optimize both stages, not just one. Prior methods either tackle prefill or decode but not both in a unified framework. LServe converts half the attention heads into “streaming heads” with Λ-shaped masks, these heads only attend to initial tokens and a local sliding window, making them nearly free to compute. The other half remain full attention heads. This static conversion is done offline once and applies to both prefill and decode. For the full attention heads, LServe observes that only a constant number of KV pages (~4096) is needed to maintain long-context capability regardless of context length. It designs a hierarchical page selector that identifies important KV pages per query token based on query-centric similarity. Crucially, selection results are reused across adjacent tokens, reducing page selection overhead. The key contribution is fusing static sparsity, dynamic sparsity, and KV cache quantization into a single GPU kernel for decode. This means speedups multiply, not add. You get static sparsity savings, dynamic sparsity savings AND quantization savings in one kernel pass. In Prefilling, it achieved up to 2.9x speedup over vLLM. And in Decoding, it achieved 1.3-2.1x speedup over vLLM. It is compatible with reasoning models and matches a dense baseline on **DeepSeek-R1-Distill-Llama-8B**, tested at context lengths up to 512k tokens. However, the static head conversion requires offline profiling to identify which heads to convert. Dynamic page selection still has overhead even with 4x reduction from reuse. The constant KV page assumption may not hold for all task types.

There are methods which use Adaptive Sparse Attention via Learned Activations. [AdaSplash (2025)](https://arxiv.org/abs/2502.12082) is a method which uses this. Softmax always assigns nonzero probability to every token. For long contexts this causes attention dispersion i.e small probabilities accumulate and dilute the attention signal. $\alpha-entmax$ is a differentiable alternative that assigns exact zero probability to irrelevant tokens, making sparsity emerge naturally from the model rather than being imposed. But prior implementations of $\alpha-entmax$ didn't exploit this sparsity for actual speedup because you still need to compute the full $QK^T$ matrix before applying the transformation. $\alpha-entmax$:

<div align="center">
  <img width="60%" alt="image" src="https://github.com/user-attachments/assets/eac1f93c-ceb9-4da4-a227-3596116dd8b8" />
  <br/>
  <p><i>Fig. AdaSplash (2025) Section 2.4. Sparse Attention</i></p>
  <img width="60%" alt="image" src="https://github.com/user-attachments/assets/329c1e62-9f69-41cd-8892-6bf7129add55" />
  <br/>
  <p><i>Fig. AdaSplash (2025) Section 3.1 alpha-entmax Computation</i></p>
</div>

Prior work used bisection which converges linearly. AdaSplash combines [bisection](https://www.google.com/search?client=safari&rls=en&q=bisection&ie=UTF-8&oe=UTF-8) with [Halley's method](https://en.wikipedia.org/wiki/Halley%27s_method), when Halley's update falls within the current bisection bounds, use it (cubic convergence), otherwise fall back to bisection. This gives 7x fewer iterations in practice. AdaSplash also implements block-wise sparse attention in a custom Triton kernel. After computing attention scores for a block, it applies  $\alpha-entmax$ and skips blocks where all scores fall below the threshold. Uses recomputation in the backward pass (like FlashAttention) to avoid storing the full attention matrix. At high sparsity (>70%), AdaSplash outperforms FlashAttention-2 in runtime. At 50% sparsity it roughly matches FlashAttention-2. It has a strong performance on [RoBERTa](https://arxiv.org/abs/1907.11692), [ModernBERT](https://arxiv.org/abs/2412.13663), and GPT-2 tasks. But a limitation is that Sparsity is data-dependent and unpredictable at runtime. You can't know in advance how sparse a given input will be, making it harder to guarantee speedups. At low sparsity it's slower than FlashAttention-2. 

# 4. Applications in Long-context Modelling

The hardware-efficient optimizations discussed throughout this blog enable advanced long-context models to function efficiently. The first major application area in long-context modelling is foundational model training and the expansion of base context limits. When training language models, exposing them to large sequences traditionally leads to high memory usage, making it difficult to process entire books or high-resolution inputs. FlashAttention revolutionized this area by fundamentally optimizing how memory is accessed during computation, significantly reducing the overall memory footprint. In practice, this allows developers to train models on vastly larger context windows in a fraction of the time. The resulting models inherently comprehend longer documents better and can successfully solve complex visual and sequential tasks that were previously deemed near impossible due to strict sequence length constraints.

In real-time interactive serving, deployed models must handle highly variable requests from multiple concurrent users. Conversational histories fluctuate unpredictably, which can cause severe delays and inefficiencies. FlashInfer addresses this by dynamically balancing computational workloads across the hardware, leading to a noticeable reduction in the initial wait time for users and the delay between each generated word. It maintains this high readiness even when processing continuous, infinite streams of text or generating multiple responses simultaneously. Similarly, the Grouped Latent Attention and Grouped-Tied Attention mechanisms drastically shrink the memory required to store conversational history. This memory reduction allows servers to handle significantly more concurrent requests and deliver higher overall throughput, ensuring smooth and reliable performance in latency-sensitive applications.

The decoding phase presents a unique challenge for long-context applications, often leaving processing cores severely underutilized. LeanAttention guarantees that no part of the system sits idle, delivering a substantial acceleration in text generation. Notably, the speedup becomes increasingly profound as the conversation history grows exceptionally long. Moreover, extracting precise information and conducting complex reasoning over massive documents is of great importance. Traditional models often struggle to find relevant details buried in noise or lose track of logical steps in lengthy derivations. Native Sparse Attention proves highly effective here by learning to inherently filter out irrelevant information from the ground up during the training phase. In practical evaluations, it demonstrates flawless retrieval of deeply hidden facts within massive texts and significantly elevates the model's ability to perform deep, multi-step mathematical reasoning. Complementing this, Dynamic Mask Attention excels in associative recall and long-document tasks by adapting its focus based on the specific content it processes. These methods implement special and efficient kernels that use the techniques described in Section 3 to efficiently compute attention over the selected number of tokens without letting GPU resources sit idle and efficiently managing memory accesses of the selected tokens.

# 5. Conclusion

Scaling LLMs to process millions of tokens exposes severe memory and compute bottlenecks in modern GPUs. Overcoming physical limitations of the GPU requires deeply aligning algorithm design with hardware execution rather than relying on naive implementations. By combining fundamental techniques like tiling, optimal work partitioning with advanced innovations such as asynchronous execution, Virtual Memory Management (VMM), and hardware-aligned sparse attention, developers can maximize GPU utilization and improve memory management. Ultimately, achieving fast, economical long-context inference demands.
