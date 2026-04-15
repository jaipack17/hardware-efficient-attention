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

<ul class="toc">
  <li>
      <span class="section-number">1. </span>Background
      <ul>
          <li><span class="section-number">1.1. </span>Attention Mechanism</li>
          <li><span class="section-number">1.2. </span>Two Phases of LLM Inference</li>
          <li><span class="section-number">1.3. </span>KV Cache</li>
      </ul>
  </li>
  
  <li>
      <span class="section-number">2. </span>Under the Hood of a GPU
      <ul>
          <li><span class="section-number">2.1. </span>GPU Architecture</li>
          <li><span class="section-number">2.2. </span>Areas of Optimization</li>
          <li><span class="section-number">2.3. </span>KV Cache Management</li>
      </ul>
  </li>
  
  <li>
      <span class="section-number">3. </span>Hardware-Efficient Attention
      <ul>
          <li><span class="section-number">3.1. </span>Tiling and Online Softmax</li>
          <li>
              <span class="section-number">3.2. </span>Optimal Work Partitioning
              <ul>
                  <li><span class="section-number">3.2.1. </span>During Prefill Phase</li>
                  <li><span class="section-number">3.2.2. </span>During Decode Phase</li>
              </ul>
          </li>
          <li><span class="section-number">3.3. </span>Asynchronous Execution and Pipelining</li>
          <li><span class="section-number">3.4. </span>Reducing Memory Fragmentation</li>
          <li><span class="section-number">3.5. </span>Greater Arithmetic Intensity and Parallel Scalability</li>
          <li>
              <span class="section-number">3.6. </span>Hardware Aligned Sparse Attention
              <ul>
                  <li><span class="section-number">3.6.1. </span>About Sparsity of Attention and Sparse Attention Methods</li>
                  <li><span class="section-number">3.6.2. </span>Why Naive Sparsity is Not Enough</li>
                  <li><span class="section-number">3.6.3. </span>Hardware Aligned Sparse Attention Methods</li>
              </ul>
          </li>
      </ul>
  </li>
  
  <li>
      <span class="section-number">4. </span>Application in Long-context Modelling
  </li>
  
  <li>
      <span class="section-number">5. </span>Conclusion
  </li>
</ul>

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

# Under the Hood of a GPU
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

<img width="1279" height="410" alt="image" src="https://github.com/user-attachments/assets/5b108013-ff61-4e54-86df-9abe8c758ef7" />
