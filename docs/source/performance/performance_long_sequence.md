# Long Sequence Performance

## LLAMA2-7B (FP8)

- The table below shows the pre-training performance of the LLAMA2-7B with CP (context parallelism) and compares it against the results without CP at various input sequence lengths. The detailed model-parallel configurations and the achieved performance are shown in the training results with CP. In non-CP training runs, we use the most performant model- and data-parallel configurations without CP given the memory capacity constraint of the H100 GPU system.

  - Container: [NeMo24.03.01.framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-H100


<table>
  <thead>
    <tr>
      <th rowspan="2" class="top-border">SeqLen (K)</th>
      <th rowspan="2" class="top-border"># of GPUs</th>
      <th rowspan="1" class="top-border">Without CP</th>
      <th colspan="5" class="top-border">With CP</th>
      <th rowspan="2" class="top-border">Speedup with CP/without CP</th>
    </tr>
    <tr>
      <th>TFLOPS / GPU</th>
      <th>TP</th>
      <th>PP</th>
      <th>DP</th>
      <th>CP</th>
      <th>TFLOPS / GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>768</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>768</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>8</td>
      <td>730</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>730</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>16</td>
      <td>660</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>660</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>32</td>
      <td>595</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>610</td>
      <td class="speedup">1.03</td>
    </tr>
    <tr>
      <td>64</td>
      <td>64</td>
      <td>534</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>574</td>
      <td class="speedup">1.07</td>
    </tr>
    <tr>
      <td>128</td>
      <td>128</td>
      <td>424</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>555</td>
      <td class="speedup">1.31</td>
    </tr>
    <tr>
      <td>256</td>
      <td>256</td>
      <td>392</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>549</td>
      <td class="speedup">1.40</td>
    </tr>
    <tr>
      <td>512</td>
      <td>512</td>
      <td>104</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>549</td>
      <td class="speedup">5.28</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>1024</td>
      <td>26.5</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>32</td>
      <td>536</td>
      <td class="speedup">20.23</td>
    </tr>
  </tbody>
</table>


### Speedup of LLAMA2 7B training with CP over without CP
![cp_speedup_figure](https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/tutorial_cp_speedup_figure.png)