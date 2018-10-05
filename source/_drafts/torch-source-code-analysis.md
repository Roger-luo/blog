---
title: A brief guide to PyTorch backend (part I)
date: 2018-10-03 12:56:38
tags: python, pytorch, machine-learning
mathjax: true
---

## Background
So I'm currently trying to add complex number to PyTorch. Complex number is quite important for physicists, and we use it quite frequently
in this novel field, **(Quantum) Physics & Machine Learning**. (Well, even just Quantum Physics, we need complex numbers).

I started using complex number in my neural network while working on a project learning a quantum many-body Hamiltonian's ground state in 2016.
However, support of complex number is quite limited, and there was no complex number support in frameworks like **PyTorch**, **MXNet**, etc. at that
time.

So I used Julia to write my first version of the research code with gradients calculated manually. However, things become harder, when we start to try deeper networks (see another blog: Failure on using deep neural network to learn a ground state). At that time I didn't try to add complex number to [Knet.jl](https://github.com/denizyuret/Knet.jl) and [Flux.jl](https://github.com/FluxML/Flux.jl) and they do not have complex number support either. But it looks easier to have complex number supported in [Flux.jl](https://github.com/FluxML/Flux.jl) now for me (I only need to change the AD rules).

Then I found [@PhilippPelz](https://github.com/PhilippPelz) implemented a complex version of PyTorch himself (with CUDA support), however, it didn't provide everything I need, (and epecially I was not able to compile it against CPU due to an `reinterpret_cast` issue).

So I re-factored [@PhilippPelz](https://github.com/PhilippPelz)'s implementation (actually for a few times) and had a very first complex version of **PyTorch 0.2**, it was mixed with some research code, so I didn't put it on Github publicly, but in a private gitlab repo. I seperated the GPU backend as [legacy-THCZ](https://github.com/Roger-luo/legacy-THCZ)
and made a [PR for CPU backend with SIMDs later in Jan, 2018](https://github.com/pytorch/pytorch/pull/4899) as a plugin-like backend to ATen.

However, it is too large (about 10k loc), it was not merged. Then I was back to Julia again, since if I cannot use other people's code without much effort, Python become not that attractive to me.

Well, now, after **one year and a half**, it is possible to support complex numbers in PyTorch. I started working on it from September as an extension in [pytorch-complex](https://github.com/Roger-luo/pytorch-complex) to accelerate [QuCumber](https://github.com/PIQuIL/QuCumber). This won't be an issue if QuCumber's [workaround](https://github.com/PIQuIL/QuCumber/blob/d9b87dbb96fef52f0f25dd8dab8b04fb8a6ffe12/qucumber/utils/cplx.py) wasn't that slow (about 20x slower than Julia on CPU for `scalar mul`). 

And if PyTorch supports complex number, we will be able to use some mature models directly in some complex-valued scenario. 

## 1.Tensor as a data structure

Although, `TH` will be abandoned and ported to `ATen/native`, but I still think it is a good way to learn what is a `Tensor` programatically.

`Tensor` (AKA multi-dimensional array, not mathematician or physicist's tensor) is a data structure that provides a multi-dimensional view to a storage. However, in Pytorch, this was mixed with `AD`'s `Variable` later. But we will just talk about `Tensor` as a view of `Storage` first.

In `TH`, a series of `THStorage` struct is implemented for different types (e.g `float` (`float32`), `double` (`float64`), `int64_t`, `int32_t`, etc.).

Which looks like:

```c
typedef struct THStorage
{
 real *data;
 ptrdiff_t size;
 int refcount;
 char flag;
 THAllocator *allocator;
 void *allocatorContext;
 struct THStorage *view;
} THStorage;
```

You cannot find this anymore since we have a generic type `at::StorageImpl` now. But the method is stays the same, let me explain a little bit of this legacy C code:

- `real` here means the data type (e.g float32, float64, etc.)
- `size` means the size of this storage (as a contiguous memory)
- `refcount` this is actually a way of GC (garbage collection) called [reference count](https://en.wikipedia.org/wiki/Reference_counting), CPython and `std::shared_ptr` in C++ use this method as well.
- `allocator` this is a pointer to CPU allocator, like what you do for `new` in C++.
- `view` this stores a `view` of the storage as the name says.

In later `PyTorch` version, they implemented `c10` that provides another reference count utility called `c10::intrusive_ptr`, this will do what have here for `THStorage` for any other type that inherit from `c10::intrusive_ptr_target`, which has better performance than `std::shared_ptr` since it does reference counting intrusively.

Then, let's check what is a `Tensor` exactly, in fact, despite of AD (automatic differenciation), it is just a (numpy/MATLAB-styled) multi-dimensional array. Although,
the concept of `Tensor` was accepted widely with tensorflow in recent years, physicists have been using it (multi-dimensional array) for condensed matter physics, etc. many years ago, libraries like [iTensor](http://itensor.org), TensorToolkit (MATLAB) was built for those purpose. e.g a quantum state of $n$-body is a rank-$n$ tensor.

Or you probably have used it before in `C` as:

```c
double tensor[R1][R2][R3];
```

Or FORTRAN:

```fortran
REAL, DIMENSION(0:R1, 0:R2, 0:R3) :: A
```

this will allocate a memory of size $R1 \times R2 \times R3$, but the way to view this memory is to use multiple indices, e.g `A[1][2][3]`, in `C` this is actually indexing a pointer multiple times.
But most people probably prefers MATLAB-style: `A[1, 2, 3]`, which is not possible in `C++`, since `C++`'s `operator []` is a single argument operator (because it is compatible with `C`!).

But generally, `Tensor` will just store the shape information and the location & size of the `Storage`, it is a way to view the storage.

In previous `C` version of PyTorch, tensor is actually such a mutable structure:

```c
typedef struct THTensor
{
 long *size;
 long *stride;
 int nDimension;

 THStorage *storage;
 ptrdiff_t storageOffset;
 int refcount;

 char flag;

} THTensor;
```

This is called **strided array**, un-like C-styled multi-dimensional array, we allocate the array on a contiguous memory and use **stride** to distinguish elements of different dimension. This is inherited from `numpy`.

Apparently, it is quite fast when you trying to **broadcast** something to every elements of the tensor, since the memory address of each elements are contiguous.
