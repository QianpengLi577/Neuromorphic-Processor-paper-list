# Neuromorphic-Processor-paper-list

My name is Qianpeng Li, a master in Institute of Automation, Chinese Academy of Sciences. I am interesting in Neuromorphic Processor, spiking neural network accelerator and machine learning. In this repository I will share with you some useful, interesting and magic papers.

## Table of Contents

- [Time Driven](#time-driven)
- [Great design](#great-design)
- [Event Driven](#event-driven)
- [Interesting Methods](#interesting-methods)
- [The meeting was not held](#the-meeting-was-not-held)
- [学位论文](#学位论文)

## Time Driven

- **to be update soon**

## Great design
I will share some wonderful design such as TrueNorth, Loihi, Darwin and soon on
- **Loihi: A Neuromorphic Manycore Processor with On-Chip Learning(Loihi)**
- **Darwin: A neuromorphic hardware co-processor based on spiking neural networks(Darwin)**
- **Cambricon-X: An Accelerator for Sparse Neural Networks(Cambricon-X)**
- **Towards artificial general intelligence with hybrid Tianjic chip architecture(TianJiC)**
- **TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip(TrueNorth)**
- **FINN: A Framework for Fast, Scalable Binarized Neural Network Inference(FINN)**
- **MorphIC: A 65-nm 738k-Synapse/mm2 Quad-Core Binary-Weight Digital Neuromorphic Processor With Stochastic Spike-Driven Online Learning(MorphIC)**
- **A 0.086-mm2 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS(ODIN)**
- **ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales(Reckon)**
- **NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation(NeuroSync)**
- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing(NeuroEngine)**

## Event Driven

- **FlexLearn: Fast and Highly Efficient Brain Simulations Using Flexible On-Chip Learning**
  - This paper conbine up to 17 representative learning rules to adjust the synaptic weights, and design and compact the specialized datapaths to maximize parallelism.
- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing**
  - NeuroEngine applies a simpler datapath(update and predict neuron state using 3 stages pipeline), multi-queue scheduler and lazy update to minimize its neuron computation and event scheduling.
- **NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation**
  - NeuroSync uses checkpoints, rollback and recover to handle mis-speculations. By applying providing trace on demand to minimize trace compution. Besides, the author defers post learning to reduce the number of incorrect weight updates.
- **ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales**
  - ReckOn is a ultra low power, high speed SNN chip. This chip accelerates spiking RNN by update e-prop(a method to train spiking RNN), combines advantage of LIF and ALIF neuron model and applies updates stochastically to 8b synaptic weights results

## Interesting Methods

### How to design a event-driven SNN acclelrator

* **Neuromorphic LIF Row-by-Row Multiconvolution Processor for FPGA**
  * The reading weight method used by the author is similar to **Efficient Hardware Acceleration of Sparsely Active Convolutional Spiking Neural Networks**, except that the accelerator is based on rows
* **FPGA-Based Implementation of an Event-Driven Spiking Multi-Kernel Convolution Architecture**
  * Divergent search for affected neuron information
* **ASIE: An Asynchronous SNN Inference Engine for AER Events Processing**
  * For convolutional neural networks, the affected neuron information is found in a divergent way

### Sparsity

- **Sparse Compressed Spiking Neural Network**
  - Bit mask is used to indicate which weights are not zero, which reduces the number of sliding windows and reduces the storage, which is used to accelerate the sparse convolutional neural network
- **A 1.13μJ/classification Spiking Neural Network Accelerator with a Single-spike Neuron Model and Sparse Weights**
  - Mask vectors are generated through some base templates to represent the sparse connections of neural networks
- **Optimized Compression for Implementing Convolutional Neural Networks on FPGA**
  - Two pruning methods, reverse pruning and peak pruning, are proposed to prune the sparse matrix. In terms of weight storage, convolution kernel sparse matrix storage adopts non-zero value + row index + column index, with a total of 16bit (considering that 11 * 11 is enough for ordinary people). For the fully connected layer sparse matrix, the interval between non-zero value and non-zero value is 16bit in total for storage. If the interval exceeds 2 ^ 8, supplement 0
- **A_Neuromorphic_Processing_System_With_Spike-Driven_SNN_Processor_for_Wearable_ECG_Classification**
  - Hierarchical Memory Access
  - G-STBP, Improved STBP for liquid pool

### Sparse representation / index

- **Cambricon-X: An Accelerator for Sparse Neural Networks**

  - Direct index and distributed index are realized
- **SMASH: Co-designing Software Compression and Hardware-Accelerated Indexing for Efficient Sparse Matrix Operations**

  - Hierarchical bitmap is proposed
- **SparTANN: Sparse Training Accelerator for Neural Networks with Threshold-based Sparsification**
  - Threshold method is used to control sparsity
  - CSR sparse representation is used
- **Other methods**

  - **Direct index**: 1 bit mark whether there is connection √
  - **Step by step index**: mark the distance from the previous non-zero weight √
  - **CCO/COO**: given row and column addresses, convolution may be friendly
  - **CSR**: how many rows (incremental) and columns are stored, and decoding may be troublesome
  - **CSC**: similar to CSR, rows and columns are interchangeable
  - ELLPACK: two peer matrices, one for storing columns and one for storing data
  - List of lists: multiple linked lists. The linked list is saved by row. One row includes non-zero values and corresponding columns
  - Diagonal storage: a matrix with the same number of columns. Store diagonal elements from the lower left corner and record the offset of the diagonal
  - RLC: contains the current data and the number of data repetitions
  - highlight is most popular

### Calculation Method

- **Speeding-up neuromorphic computation for neural networks: Structure optimization approach**
  - Through the calculation scheme of dendrites and axons, the problem that one layer must be calculated completely before the next layer can be calculated in the forward calculation of layers is prevented.
- **Efficient Design of Spiking Neural Network With STDP Learning Based on Fast CORDIC**
  - The network size is 784 * 100. Set 14 * 10pe array. Calculate the 14 * 10 network every time and cycle 560 times to get the final network. This article is actually similar to axon or dendrite calculation

### Algorithm Improvement

- **A 65-nm Neuromorphic Image Classification Processor With Energy-Efficient Training Through Direct Spike-Only Feedback**
  - The SD algorithm is modified and the neuron model is modified. In addition, the calculation of weight update is reduced by setting (batch size = 1) whether to update (rarely affecting performance)
- **ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales**
  - Modify the e-prop training rules of SRNN and only use the qualification trace at the current time. Use 8-bit fixed-point number during training and update it randomly.

### Memory Allocation

- **Efficient Hardware Acceleration of Sparsely Active Convolutional Spiking Neural Networks**
  - Aer: row col valid invalid containing pulses. Each channel has an aer queue. Nine small SRAM (3 * 3) memory membrane potentials are used, which can be accessed concurrently with less delay. Nine different memory columns are read each time. Membrane voltage memory mempot time-sharing multiplexing divides the calculation process into layers - > channel. All time steps of one layer need to be calculated each time. Each layer is allocated according to channel, and the output pulse event is put into AEQ

### Modern Architecture Technology

- **NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation**
  - Checkpoint, rollback and recovery are used for error correction, collective trace update and delayed post learning
- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing**
  - The neuron voltage calculation is divided into three-stage pipeline, and the event scheduling is carried out by using bitmap and FIFO. Whether to update the neuron voltage is predicted according to the input weighted pulse value.

### Single Core or Multi-Core

- **Spiking Neural Network Integrated Circuits: A Review of Trends and Future Directions**
  - When the synapse of single nucleus exceeds 10K, simulated nucleus should be used
  - Single core design
    - Network topology: crossbarm, locally competitive algorithm, multi layer.However, these network topologies cannot be reconfigured
    - Storage architecture: in memory computing, RRAM, standard SRAM
    - Learning algorithm: STDP, SDSP, top-down gradient based approach
  - Multi core design
    - Design principles: scalability, tradeoff between flexibility and reliability, programmability and compatibility
    - Implementation platform: analog computing or digital computing
    - Performance comparison: appropriate benchmark is required

### Encoding model

- **Deep Spiking Neural Network: Energy Efficiency Through Time based Coding**
  - A coding method called time switch coding (TSC) is designed
  - Compared with TTFS, TSC has fewer synaptic operations to reduce energy

### Event code representation

- **S2N2: A FPGA Accelerator for Streaming Spiking Neural Networks**
  - The author uses binary tensor to reduce memory utilization
  - Fixed per layer propagation delays are used
  - S2N2 supports LIF neuron model based on FINN

### Low power

- **A 0.086-mm2 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS**
  - Synaptic model, supporting SDSP
  - The neuron model supports LIF and 20 this izh model
  - Scheduling scheme to convert burst into monopulse
  - It supports 256 neurons and 64K synapses

### Event scheduling

- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing**
  - The neuron voltage calculation is divided into three-stage pipeline, and events are scheduled using bitmap and FIFO. Whether to update the neuron voltage is predicted according to the input weighted pulse value.
- **Hardware Implementation of Spiking Neural Networks on FPGA**
  - Multiple event queues are used, and each queue represents all events after n time

## The meeting was not held

- **DAC2022**

  - **Unicorn: A Multicore Neuromorphic Processor with Flexible Fan-In and Unconstrained Fan-Out for Neurons	Date: 2022.07.14    Time: 11:15-11:37**
    - for flexible fan-in, combine mult NUs(neuron units) to generate a bigger fan-in neuron
    - for unconstrained fan-out, find TRT(target routing table) to any core/neuron you want 
  - **SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture**
    - SATO using Sparsity of activation and process the input of all time steps in parallel
    - using Bucket-Sort Based Dispatcher to tradeoff the workload of PE
  - **A Time-to-first-spike Coding and Conversion Aware Training for Energy-Efficient Deep Spiking Neural Network Processor Design		Date: 2022.07.12    Time: 16:14-16:42**
    - A ANN to SNN transformation method called cat is designed
    - A new TTFS coding rule is designed (I don't quite understand this method)
    - I think this method fully using the time information in SNN
  - **SoftSNN: Low-Cost Fault Tolerance for Spiking Neural Network Accelerators under Soft Errors(gotten)		Date：2022.07.12   Time: 13:52 -14:15**

    - This article uses two operations to mitigate soft error
    - First: when the weight mutation exceeds max, it is modified to the number of advance payments
    - The second: when the neuron cannot reset, it will output 0 through spike signal
- **MICRO2022-0701**

  - Revised paper on July 1
- **ISCA2022-0622**

  - The meeting ended on June 22
  - **dont have interesting papers for me**
- **ICCAD2022-0721**

  - July 21 notification of acceptance
- **VLSI2022-0617**

  - The meeting ended on June 17
  - **dont have interesting papers for me**


## 学位论文

- **面向稀疏神经网络的片上系统设计与实现**
  - The author proposes a sparse weight representation method of dynamic ell. The author changes the column coordinates in the sparse representation of ell to step index

- **面向深度神经网络的数据流架构软硬件协同优化研究**
  - A FD-CSR sparse representation method suitable for fine-grained data flow is proposed, and the column coordinates are represented by bitmap. This is mainly to adapt to the author's hardware architecture and to realize storage alignment

- **面向达尔文II类脑计算芯片的仿真训练平台**
  - The fan out relationship, weight and delay of the source neuron are recorded in the form of a linked list. The delay management unit of Darwin I is used. The route adopts the relative origin, and the dynamic virtual connection reduces the number of packets to a certain extent
