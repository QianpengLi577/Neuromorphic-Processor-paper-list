# Neuromorphic-Processor-paper-list

My name is Qianpeng Li, a master in Institute of Automation, Chinese Academy of Sciences. I am interesting in Neuromorphic Processor, spiking neural network accelerator and machine learning. In this repository I will share with you some useful, interesting and magic papers.

## Table of Contents
 - [Time Driven](#time-driven)
 - [Event Driven](#event-driven)

## Time Driven

- **to be update soon**

## Event Driven

- **FlexLearn: Fast and Highly Efficient Brain Simulations Using Flexible On-Chip Learning**
  This paper conbine up to 17 representative learning rules to adjust the synaptic weights, and design and compact the specialized datapaths to maximize parallelism.
- **NeuroEngine: A Hardware-Based Event-Driven Simulation System for Advanced Brain-Inspired Computing**
  NeuroEngine applies a simpler datapath(update and predict neuron state using 3 stages pipeline), multi-queue scheduler and lazy update to minimize its neuron computation and event scheduling.
- **NeuroSync: A Scalable and Accurate Brain Simulator Using Safe and Efficient Speculation**
  NeuroSync uses checkpoints, rollback and recover to handle mis-speculations. By applying providing trace on demand to minimize trace compution. Besides, the author defers post learning to reduce the number of incorrect weight updates.
- **ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales**
  ReckOn is a ultra low power, high speed SNN chip. This chip accelerates spiking RNN by update e-prop(a method to train spiking RNN), combines advantage of LIF and ALIF neuron model and applies updates stochastically to 8b synaptic weights results
