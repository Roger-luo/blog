---
title: The quantum Hamiltonian ground state problem & machine learning
tags: 
    - quantum physics
    - machine learning
mathjax: true
---

I was trying and is trying to solve this problem with modern machine learning techniques from last year. And I had quite a lot failure, also this problem looks quite similar to **reinforcement learning**. However, models and techniques in **Computer Vision** or **Go** don't work here. But does this mean **Tensor Networks** and **DFT** methods are the only choice? What can we learn from modern machine learning techniques?

## Problem Setup

In quantum physics, using a Hamiltonian to describe the system is quite common, and there are quite a lot properties related to the ground state of the Hamiltonian.

A very simple model of a Hamiltonian will looks like

$$
H = -J\sum_{< ij \>} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i
$$

which is called the **Traverse Field Ising Model** (TFIM), where $J$ and $h$ are just (real-valued) constants. The Hamiltonian is, in fact, a large matrix composed of many small matrix by kronecker product. I'm not going to write about quantum physics here, because I'll try to explain the problem more like a machine learning task.

And the ground state of this Hamiltonian is defined as the eigen vector with smallest eigen value of $H$, we denote it as $\Psi$. Or to express this mathematically, the task is

To find a $\Psi$, 

$$
H\Psi = E_0 \Psi
$$

that for any $\phi_i \in \mathcal{H}$ satisfy

$$
H\phi_i = E_i \phi_i
$$

we have

$$
E_0 \leq E_i, \forall E_i
$$

### How to construct the Hamiltonian (matrix)?

In fact, the **Hamiltonian** is an operator for a $n$-body quantum system, usually we only study the $k$-local Hamiltonians, which means the Hamiltonian is composed by many local operators on $k$-bodies only. This is because we believe, in general, the interaction is limited in local region, rather than global interaction.

An example of $2$-local Hamiltonian will be the Heisenberg model:

$$
H = \sum_{< ij \>} \sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j + \sigma^z_i \sigma^z_j
$$

Here, the $\sigma$ represents Pauli matrices, they looks like:

$$
\sigma^x = \begin{pmatrix}
0 & 1 \\\\
1 & 0
\end{pmatrix}
\quad
\sigma^y = \begin{pmatrix}
0 & -i\\\\
i &  0
\end{pmatrix}
\quad
\sigma^z = \begin{pmatrix}
1 &  0\\\\
0 & -1
\end{pmatrix}
$$

each index $i, j$ means the id of the site/qubit/etc. on a lattice, $\sum_{< ij \>}$ means to sum up $i, j$ as nearest neighbors:

1. on chain lattice: $j = i+1$
2. on square lattice: this means sum over the following nearest neighbors on a square lattice

![nearest-neighbors-on-square]()


### How hard is this problem

Unlike **image classification** or **image generation**, the problem of finding the ground state of a many-body
Hamiltonian was well studied by complexity theorists.[^toby2013] There is even a field called **Hamiltonian Complexity** for
various problem related to Hamiltonian.[^sevag2016]


Generally, for arbitrary **2-local Hamiltonian**, the complexity to find its ground state is

[^toby2013]: [Complexity classification of local Hamiltonian problems](https://arxiv.org/abs/1311.3161)
[^sevag2016]: [Quantum Hamiltonian Complexity](https://arxiv.org/abs/1401.3916)

## Methology

### Density Functional Theory

### Variational Monte Carlo

To sum over the Hilbert space directly can be hard, but with Monte Carlo, we can sum it approximately.

$$
\begin{aligned}
E &= \sum_{ss'} \Psi_s^* H_{ss'} \Psi_{s'} \\\\
  &= \sum_{s} |\Psi_s|^2 \frac{\sum_{s'} H_{ss'} \Psi_{s'}}{\Psi_s} \\\\
  &= \sum_{s} |\Psi_s|^2 E_{local}
\end{aligned}
$$

However, general VMC suffers from the sign problem. Prior knowledge is needed to reduce the complexity.

### Tensor Networks (Models)

## A machine learning perspective of the local Hamiltonian problem

In machine learning, we usually use a **cost function** to describe how well our machine learns the problem. For example, we use **KL divergence**
as the loss function in **image classification**, **W-Loss** for measuring how similar the picture **Generator** generates. In physics, this is just
the energy, like in clasical statistical physics, the **KL divergence** of **Boltzmann machine** is just the free energy, mimizing the energy means
acheiving maximum likelihood.

### Restricted Boltzmann Machine? A Feedforward Neural Network

Let's review the very first paper that drives people in this field to machine learning techniques in 2016.

### Convolutional Neural Network? PEPS's competitor? Probably no.

### Prior

The prior knowledge is important in reducing the complexity of learning. [^shai2014]

[^shai2014]: [On the Computational Complexity of Deep Learning](http://lear.inrialpes.fr/workshop/osl2015/slides/osl2015_shalev_shwartz.pdf)


## Beyond Quantum State Tomography

Quantum State Tomography (QST) is a kind of inverse problem of what we were discussing about. They both require a good (sparse) model for quantum states,
however, quantum state tomography uses data (measured from experiments) to add constrains to the training, where for ground states, the constrain is
the model itself.

Therefore, is it possible to adjust the model based on the prior knowledge we learnt from data already?

