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
