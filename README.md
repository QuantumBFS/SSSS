# Deep Learning and Quantum Programming: A Spring School

Song Shan lake Spring School, with lectures, code challenge and happy fatty night.

Song Shan Lake, Dong Guan, China, from 5th May to 10th May, 2019.

## Table of Contents
1. Deep Learning
    * [`lecture_notes.pdf`](https://github.com/QuantumBFS/SSSS/blob/master/1_deep_learning/lecture_notes.pdf) and [`slides/`](https://github.com/QuantumBFS/SSSS/tree/master/1_deep_learning/slides)
    * Demo codes
        * Poor man's computation graph: [`computation_graph.py`](https://github.com/QuantumBFS/SSSS/blob/master/1_deep_learning/computation_graph.py)
        * Variational free energy with flow model: [`realnvp/`](https://github.com/QuantumBFS/SSSS/tree/master/1_deep_learning/realnvp)
        * Hamiltonian inverse design with reverse mode AD: [`schrodinger.py`](https://github.com/QuantumBFS/SSSS/blob/master/1_deep_learning/schrodinger.py)
        * Solving the fastest descent problem with NeuralODE [`brachistochrone/`](https://github.com/QuantumBFS/SSSS/tree/master/1_deep_learning/brachistochrone)
2. Tensor Networks
   * [`Slides on tensor networks`](https://github.com/QuantumBFS/SSSS/blob/master/2_tensor_network/Tutorial_tensor_network.pdf)
   * [`Slides on contraction methods for infinite tensor networks`](https://github.com/QuantumBFS/SSSS/blob/master/2_tensor_network/tensor_contraction_methods.pdf)
   * [`Tutorial and demo codes on computing $2$-D Ising model partition function using tensor networks`](https://github.com/QuantumBFS/SSSS/blob/master/2_tensor_network/tensor_contraction_simple.ipynb)
   * [`Tutorial and demo codes on the MPS Born machine`](https://github.com/QuantumBFS/SSSS/blob/master/2_tensor_network/mps_tutorial.ipynb)
3. Julia language
4. Variational Quantum Computing
    * Lecture Note: [quantum_lecture_note.pdf](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/quantum_lecture_note.pdf)
    * Slides: [google slides](https://docs.google.com/presentation/d/1jUTpa8pB3jEOWDW1U0rDTDQ-kpri8j8S4y77GQCo3iM/edit?usp=sharing)
    * Notebooks
        * The solution to the graph embeding problem: [graph_embeding.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/graph_embeding.ipynb)
        * Quantum circuit computing with Yao.jl: [QC-with-Yao.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/QC-with-Yao.ipynb)
        * Landscape of a quantum circuit: [variational_quantum_circuit.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/variational_quantum_circuit.ipynb)
        * Variational quantum eigensolver: [variational_quantum_circuit.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/variational_quantum_circuit.ipynb)
        * Matrix Product state inspired variational quantum eigensolver [VQE_action.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/VQE_action.ipynb)
        * Quantum circuit born machine: [qcbm_gaussian.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/qcbm_gaussian.ipynb)
        * Gradient vanishing problem: [variational_quantum_circuit.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/variational_quantum_circuit.ipynb) and [VQE_action.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/VQE_action.ipynb)
        * Mapping a quantum circuit to tensor networks: [qc_tensor_mapping.ipynb](https://github.com/QuantumBFS/SSSS/blob/master/4_quantum/qc_tensor_mapping.ipynb)

Welcome for pull requests and issues!

     
## Challenge

[Song-Shan-Hu Sping School Coding Challenge](Challenge.md)

## Preparation

### Quick start

- [quick start for git](http://rogerdudler.github.io/git-guide/)
- [quick start for command line interface](https://www.makeuseof.com/tag/a-quick-guide-to-get-started-with-the-linux-command-line/)

### Installation
- [how to install ubuntu](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop)
- [install annaconda](https://www.anaconda.com/distribution/)
- [install PyTorch](https://pytorch.org/)
- [install Julia](https://julialang.org)
- [intall Yao.jl](https://github.com/QuantumBFS/Yao.jl#installation)

### Julia
- [Julia语言的中文教程](https://github.com/Roger-luo/TutorialZH.jl)
- [快速入门 Julia 语言](https://www.bilibili.com/video/av28248187?from=search&seid=5171149583764025744)
- [Julia入门指引](https://discourse.juliacn.com/t/topic/159)


## Usage

You can open this repo as a Julia package if you have julia installed:

1. open your Julia REPL, press `]`
2. type the following

```julia
(1.0) pkg> add https://github.com/QuantumBFS/SSSS.git
```

3. press backspace
4. type the following

```julia
julia> using SSSS

julia> notebooks()
```

## License

The code is released under MIT License. The rest part is released under [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<p align="center">
<img align="middle" src="_assets/SongShanHu2019.jpeg" width="500" alt="poster"/>
</p>
