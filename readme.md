# Neuromorphic-Processor-paper-list

My name is Qianpeng Li, a master in Institute of Automation, Chinese Academy of Sciences. I am interesting in Neuromorphic Processor, spiking neural network accelerator and machine learning. In this repository I will share with you some useful, interesting and magic papers.

## Table of Contents
 - [Time Driven](#time-driven)
 - [Event Driven](#event-driven)
 - [Interesting Methods](#interesting-methods)

## Time Driven

- **to be update soon**

## Event Driven

- **FlexLearn: Fast and Highly Efficient Brain Simulations Using Flexible On-Chip Learning**
  - This paper conbine up to 17 representative learning rules to adjust the synaptic weights, and design and compact the specialized datapaths to maximize parallelism.
- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing**
  - NeuroEngine applies a simpler datapath(update and predict neuron state using 3 stages pipeline), multi-queue scheduler and lazy update to minimize its neuron computation and event scheduling.
- **NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation**
  -  NeuroSync uses checkpoints, rollback and recover to handle mis-speculations. By applying providing trace on demand to minimize trace compution. Besides, the author defers post learning to reduce the number of incorrect weight updates.
- **ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales**
  - ReckOn is a ultra low power, high speed SNN chip. This chip accelerates spiking RNN by update e-prop(a method to train spiking RNN), combines advantage of LIF and ALIF neuron model and applies updates stochastically to 8b synaptic weights results

## Interesting Methods

### Sparsity
- **Sparse Compressed Spiking Neural Network**
  - Bit mask is used to indicate which weights are not zero, which reduces the number of sliding windows and reduces the storage, which is used to accelerate the sparse convolutional neural network
- **A 1.13μJ/classification Spiking Neural Network Accelerator with a Single-spike Neuron Model and Sparse Weights**
  - Mask vectors are generated through some base templates to represent the sparse connections of neural networks
- **Optimized Compression for Implementing Convolutional Neural Networks on FPGA**
  - Two pruning methods, reverse pruning and peak pruning, are proposed to prune the sparse matrix. In terms of weight storage, convolution kernel sparse matrix storage adopts non-zero value + row index + column index, with a total of 16bit (considering that 11 * 11 is enough for ordinary people). For the fully connected layer sparse matrix, the interval between non-zero value and non-zero value is 16bit in total for storage. If the interval exceeds 2 ^ 8, supplement 0

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
