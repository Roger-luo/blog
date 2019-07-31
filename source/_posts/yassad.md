---
title: Implement Your Own Source To Source AD in ONE day!
date: 2019-07-27 14:49:33
tags: automatic differentiation
mathjax: true
---

I wrote a blog post about how to implement your own (operator overloading based) automatic differentiation (AD) in one day (actually 3 hrs) last year. AD looks like magic sometimes, but I'm going to talk about some black magic this time: the source
to source automatic differentiation. I wrote this during JuliaCon 2019 hackthon with help from [Mike Innes](https://github.com/MikeInnes).
It turns out that writing a blog post takes longer than writing a source to source AD ;-). This is basically just simple version of Zygote.

I wrap this thing as a very simple package here, if you want to look at more detailed implementation: [YASSAD.jl](https://github.com/Roger-luo/YASSAD.jl).

If you have used operator overloading based AD like [PyTorch](https://github.com/pytorch/pytorch), [Flux/Tracker](https://github.com/FluxML/Tracker.jl), [AutoGrad](https://github.com/HIPS/autograd), you may find they have some limitations:

- A `Tensor` type or `Variable` type provided by the package has to be used for tracing the function calls
- They cannot handle control flows in general, even in some cases, some workarounds can be taken

However, programming without control flow is not programming! And it is usually very annoying to rewrite a lot code with tracked types. If we want to have a framework for **Differentiable Programming** as what people like **Yan LeCun** has been proposing, we need to solve these two problems above.

In fact, these problems are quite straight forward to solve in source to source automatic differentiation, since we basically know everything happens. I will implement a very simple source to source AD without handling control flows, you can also check the complete implementation as [Zygote.jl](https://github.com/FluxML/Zygote.jl).

But before we start, let's review some basic knowledge.

## Basics

### The compilation process of Julia language

I will briefly introduce how Julia program is compiled and run in this section:

1. all the code are just strings
2. the Julia parser will parse the strings first to get an Abstract Syntax Tree (AST)
3. some of the nodes in this AST are macros, macros are like compiled time functions on expressions, the compiler will expand the macros. Then we get an expanded version of AST, which do not have any macros. You can inspect the results with `@macroexpand`.
4. Now, we will lower the AST, get rid of syntax sugars and represent them in Static Single Assignment Form (SSA), you can get it with `@code_lowered`, and you can modify this process with Julia `macro`s.
5. When function call happens, we use the function signature to dispatch the function to a certain method, and start doing type inference. You can modify this process with `@generated` functions, and check the results with `@code_typed`.
6. The compiler will then generate the llvm IR. You can inspect them with `@code_llvm`
7. After we have llvm IR, Julia will use llvm to generate native code to actually exectute this function.
8. By executing the function, we will meet another function call, so we go back to step 5

I steal a diagram from JuliaCon 2018 to demonstrate this process:

![](/images/julia-compile-diagram.png)

As you can see. Julia is not a static compiled language, and it uses function as boundary of compilation.

### SSA Form IR

A complete introduction of SSA can be [a book](http://ssabook.gforge.inria.fr/latest/book.pdf). But to implement your own source
to source AD only require three simple concept:

- all the variable will only be assigned once
- most variable comes from function calls
- all the control flows become branches

If you have read my last post, I believe you have understand what is computation graph, but now let's look at this diagram again: what is this computation graph exactly?

![comput-graph](/images/comput-graph-forward.gif)

While doing the automatic differentiation, we represent the process of computation as a diagram. Each node is an operator with a intermediate value. And each operator also have an **adjoint operator** which will be used in backward pass. Which means each variable
in each node will only be assigned once. This is just a simple version of SSA Form right?

The gradient can be then considered as an adjoint program of the original program. And the only thing we need to do is to generate the adjoint program. In fact, this is often called Wengert list, tape or graph as described in [Zygote's paper: Don't Unroll Adjoint](https://arxiv.org/pdf/1810.07951.pdf). Thus we can directly use the SSA form as our computational graph. Moreover, since in Julia the SSA form IR is lowered, it also means we only need to defined a few primitive routines instead of defining a lot operators.

Since the backward pass is just an adjoint of the original program, we can just write it as a closure

```julia
function forward(::typeof(your_function), xs...)
    # function declaration
    output = # function output
    output, function (Δ)
        # a closure
    end
end
```

The advantage of defining this as closure is that we can let the compiler itself handle shared variable between the adjoint program
and the original program instead of managing it ourselves (like what we do in my last post). We call these closures **pullback**s.

So given a function like the following

```julia
function foo(x)
    a = bar(x)
    b = baz(x)
    return b
end
```

If we do this manually, we only need to define a `forward` function

```julia
function forward(::typeof(foo), x)
      x1, back1 = forward(baz, x)
      x2, back2 = forward(bar, x1)
      return x2, function (Δ)
         dx1 = back2(Δ)
         dx2 = back1(dx1)
         return dx2
      end
end
```

In general, an adjoint program without control flow is just applying these pullbacks generated by their **forward** function in reversed order. But how do we do this automatically? Someone may say: let's use macros! Err, we can do that. But our goal is to differentiate arbitrary function defined by someone else, so things can be composable. This is not what we want. Instead, we can tweak the IR, the **generated function**s in Julia can not only return a modified AST from type information, it can also return the IR.

The generated function can be declared with a `@generated` macro

```julia
@generated function foo(a, b, c)
    return :(1 + 1)
end
```

It looks like a function as well, but the difference is that inside the function, the value of each function argument `a`, `b`, `c`
is their type since we do not have their values during compile time.

![](/images/julia-generated-compile-diagram.png)

In order to manipulate the IR, we need some tools. Fortunately, there are some in [IRTools](https://github.com/MikeInnes/IRTools.jl), we will use this package to generate the IR code.

First, we can use `@code_ir` to get the `IR` object processed by `IRTools`. Its type is `IR`. The difference between the one you get from `@code_lowered` is that this will not store the argument name, all the variables are represented by numbers, and there are some useful function implemented for this type.

```julia
julia> @code_ir foo(1.0)
1: (%1, %2)
  %3 = (Main.baz)(%2)
  %4 = (Main.bar)(%3)
  return %4
```

In this form, each line of code is binded to a variable, we call the right hand statement, and left hand variable. You use a dict-like interface to use this object, e.g

```julia
julia> using IRTools: var

julia> ir[var(3)]
IRTools.Statement(:((Main.baz)(%2)), Any, 1)
```

It will return a statement object, which stores the expression of this statement, the inferred type (since we are using the IR before type inference, this is `Any`). For simplicity, we will not use typed IR in this post (since in principal, their implementations are similar). The last number is the line number.

What is the first number `1` in the whole block? It means code block, in SSA form we use this to represent branches, e.g

```julia
julia> function foo(x)
           if x > 1
               bar(x)
           else
               baz(x)
           end
       end
foo (generic function with 1 method)

julia> @code_ir foo(1.0)
1: (%1, %2)
  %3 = %2 > 1
  br 3 unless %3
2:
  %4 = (Main.bar)(%2)
  return %4
3:
  %5 = (Main.baz)(%2)
  return %5
```

`ifelse` is just branch statement in lowered SSA form, and in fact, `for` loops are similar. Julia's for loop is just a syntax sugar of `iterate` function. As long as we can differentiate through `br`, we will be able to differentiate through control flows.

```julia
julia> function foo(x)
           for x in 1:10
               bar(x)
           end
           baz(x)
       end
foo (generic function with 1 method)

julia> @code_ir foo(1.0)
1: (%1, %2)
  %3 = 1:10
  %4 = (Base.iterate)(%3)
  %5 = %4 === nothing
  %6 = (Base.not_int)(%5)
  br 3 unless %6
  br 2 (%4)
2: (%7)
  %8 = (Core.getfield)(%7, 1)
  %9 = (Core.getfield)(%7, 2)
  %10 = (Main.bar)(%8)
  %11 = (Base.iterate)(%3, %9)
  %12 = %11 === nothing
  %13 = (Base.not_int)(%12)
  br 3 unless %13
  br 2 (%11)
3:
  %14 = (Main.baz)(%2)
  return %14
```

So how do we get the IR? In order to get the IR, we need to know which method is dispatched for this generic function. Each generic
function in Julia has a method table, you can use the type signature of the function call to get this method, e.g when you call `foo(1.0)`, Julia will generate `Tuple{typeof(foo), Float64}` to call the related method. We can get the meta information of this method by providing the `IRTools.meta` function with this type signature

```julia
julia> IRTools.IR(m)
1: (%1, %2)
  %3 = (Main.baz)(%2)
  %4 = (Main.bar)(%3)
  return %4
```

And we can manipulate this IR with functions like `push!`:

```julia
julia> push!(ir, :(1+1))
%5

julia> ir
1: (%1, %2)
  %3 = (Main.baz)(%2)
  %4 = (Main.bar)(%3)
  %5 = 1 + 1
  return %4
```

`IRTools` will add the variable name for you automatically here. Similarly, we can use `insert!` to insert a statement before the 4th variable:

```julia
julia> using IRTools: var

julia> insert!(ir, var(4), :(1+1))
%5

julia> ir
1: (%1, %2)
  %3 = (Main.baz)(%2)
  %5 = 1 + 1
  %4 = (Main.bar)(%3)
  return %4
```

Or we can insert a statement after the 4th variable:

```julia
julia> using IRTools: insertafter!

julia> insertafter!(ir, var(4), :(2+2))
%6

julia> ir
1: (%1, %2)
  %3 = (Main.baz)(%2)
  %5 = 1 + 1
  %4 = (Main.bar)(%3)
  %6 = 2 + 2
  return %4
```

With these tools, we can now do the transformation of forward pass. Our goal is to replace each function call with the function call to `forward` function and then collect all the pullbacks returned by `forward` function to generate a closure. But wait! I didn't mention closure, what is the closure in SSA IR? Let's consider this later, and implement the transformation of forward part first.

Let's take a statement and have a look

```julia
julia> dump(ir[var(3)])
IRTools.Statement
  expr: Expr
    head: Symbol call
    args: Array{Any}((2,))
      1: GlobalRef
        mod: Module Main
        name: Symbol baz
      2: IRTools.Variable
        id: Int64 2
  type: Any
  line: Int64 1
```

In fact, we only need to check whether the signature of its expression is `call`. We can use the `Pipe` object in `IRTools` to do the transformation, the transformation results are stored in its member `to`.

```julia
julia> IRTools.Pipe(ir).to
1: (%1, %2)
```

## Implementation

### Forward Transformation

We name this function as `register` since it has similar functionality as our old `register` function in my last post. The only difference is: you don't need to write this `register` function manually for each operator now! We are going to do this automatically. 

**Warning**: since I'm doing this demo in REPL, I use `Main` module directly, if you put the code in your own module, replace it with your module name.

```julia
function register(ir)
    pr = Pipe(ir)
    argument!(pr, at = 1)
    for (v, st) in pr
        ex = st.expr
        if Meta.isexpr(ex, :call)
            yJ = insert!(pr, v, stmt(xcall(Main, :forward, ex.args...), line = ir[v].line))
            pr[v] = xgetindex(yJ, 1)
        end
    end
    finish(pr)
end
```

I'll explain what I do here: first since we are generating the IR for the `forward` function, we have an extra argument now

```julia
forward(f, args...)
```

Thus, I added one argument at the beginning of this function's IR.

Then, we need to iterate through all the variables and statements, if the statement is a function call then we replace it with the call
to `forward` function. Remember to keep the line number here, since we still want some error message. Since the returned value of `forward` is a tuple of actually forward evaluation and the pullback, to get the correct result we need to index this tuple, and replace
the original variable with the new one. The `xgetindex` here is a convenient function that generates the expression of `getindex`

```julia
xgetindex(x, i...) = xcall(Base, :getindex, x, i...)
```

Let's see what we get

```julia
julia> register(ir)
1: (%3, %1, %2)
  %4 = (Main.forward)(Main.baz, %2)
  %5 = (Base.getindex)(%4, 1)
  %6 = (Main.forward)(Main.bar, %5)
  %7 = (Base.getindex)(%6, 1)
  return %7
```

Nice! We change the function call to forward now!

Now, it's time to consider the closure problem. Yes, in this lowered form, we don't have closures. But we can instead store them in a callable object!

```julia
struct Pullback{S, T}
    data::T
end

Pullback{S}(data::T) where {S, T} = Pullback{S, T}(data)
```

This object will also store the function signature, so when we call pullback, we can look up the IR of the original call to generate the IR of this pullback. The member `data` here will store a `Tuple` of all pullbacks with the order of their `forward` call. In order to construct the `Pullback` we need the signature of our function call, so we need to revise our implementation as following.

```julia
function register(ir, F)
    pr = Pipe(ir)
    pbs = Variable[]
    argument!(pr, at = 1)
    for (v, st) in pr
        ex = st.expr
        if Meta.isexpr(ex, :call)
            yJ = insert!(pr, v, stmt(xcall(Main, :forward, ex.args...), line = ir[v].line))
            pr[v] = xgetindex(yJ, 1)
            J = insertafter!(pr, v, stmt(xgetindex(yJ, 2), line = ir[v].line))
            push!(pbs, substitute(pr, J))
        end
    end
    pr = finish(pr)
    v = push!(pr, xtuple(pbs...))
    pbv = push!(pr, Expr(:call, Pullback{F}, v))
    return pr
end
```

In order to store the pullbacks, we need to get the pullback from the tuple returned by `forward` and allocate a list to record all pullbacks.

Here `xtuple` is similar to `xgetindex`, it is used to generate the expression of constructing a tuple.

```julia
xtuple(xs...) = xcall(Core, :tuple, xs...)
```

Let's pack the pullback and the original returned value as a tuple together, and return it!

```julia
function register(ir, F)
    pr = Pipe(ir)
    pbs = Variable[]
    argument!(pr, at = 1)
    for (v, st) in pr
        ex = st.expr
        if Meta.isexpr(ex, :call)
            yJ = insert!(pr, v, stmt(xcall(Main, :forward, ex.args...), line = ir[v].line))
            pr[v] = xgetindex(yJ, 1)
            J = insertafter!(pr, v, stmt(xgetindex(yJ, 2), line = ir[v].line))
            push!(pbs, substitute(pr, J))
        end
    end
    pr = finish(pr)
    v = push!(pr, xtuple(pbs...))
    pbv = push!(pr, Expr(:call, Pullback{F}, v))
    ret = pr.blocks[end].branches[end].args[1]
    ret = push!(pr, xtuple(ret, pbv))
    pr.blocks[end].branches[end].args[1] = ret
    return pr, pbs
end
```

The `return` statement is actually a simple branch, it is the last branch of the last statement of the last code block.

OK, let's see what we get now

```julia
julia> register(ir, Tuple{typeof(foo), Float64})
1: (%3, %1, %2)
  %4 = (Main.forward)(Main.baz, %2)
  %5 = (Base.getindex)(%4, 1)
  %6 = (Base.getindex)(%4, 2)
  %7 = (Main.forward)(Main.bar, %5)
  %8 = (Base.getindex)(%7, 1)
  %9 = (Base.getindex)(%7, 2)
  %10 = (Core.tuple)(%9, %6)
  %11 = (Pullback{Tuple{typeof(foo),Float64},T} where T)(%10)
  %12 = (Core.tuple)(%8, %11)
  return %12
```

Now let's implement the `forward` function

```julia
@generated function forward(f, xs...)
    T = Tuple{f, xs...}
    m = IRTools.meta(T)
    m === nothing && return
end
```

We will get the meta first, if the meta is `nothing`, it means this method doesn't exist, so we just stop here. If we have the meta, then
we can get the `IR` from it and put it to `register`

```julia
@generated function forward(f, xs...)
    T = Tuple{f, xs...}
    m = IRTools.meta(T)
    m === nothing && return
    frw = register(IR(m), T)
end
```

However, the object `frw` has type `IR` instead of `CodeInfo`, to generate the `CodeInfo` for Julia compiler, we need to put argument names back with

```julia
argnames!(m, Symbol("#self#"), :f, :xs)
```

And since the second argument of our `forward` function is a vararg, we need to tag it to let our compiler know, so the compiler will not feed the first function call with a `Tuple`.

```julia
frw = varargs!(m, frw, 2)
```

In the end, our forward function will looks like

```julia
@generated function forward(f, xs...)
    T = Tuple{f, xs...}
    m = IRTools.meta(T)
    m === nothing && return
    frw = register(IR(m), T)
    argnames!(m, Symbol("#self#"), :f, :xs)
    frw = varargs!(m, frw, 2)
    return IRTools.update!(m, frw)
end
```

Let's see what we got now

```julia
julia> @code_ir forward(foo, 1.0)
1: (%1, %2, %3)
  %4 = (Base.getfield)(%3, 1)
  %5 = (Main.forward)(Main.baz, %4)
  %6 = (Base.getindex)(%5, 1)
  %7 = (Base.getindex)(%5, 2)
  %8 = (Main.forward)(Main.bar, %6)
  %9 = (Base.getindex)(%8, 1)
  %10 = (Base.getindex)(%8, 2)
  %11 = (Core.tuple)(%10, %7)
  %12 = (Main.Pullback{Tuple{typeof(foo),Float64},T} where T)(%11)
  %13 = (Core.tuple)(%9, %12)
  return %13
```

If you try to actually run this, there will be some error unfortunately

```julia
julia> forward(foo, 1.0)
ERROR: MethodError: no method matching getindex(::Nothing, ::Int64)
Stacktrace:
 [1] * at ./float.jl:399 [inlined]
 [2] forward(::typeof(*), ::Float64, ::Float64) at /Users/roger/.julia/dev/YASSAD/src/compiler.jl:0
 [3] baz at ./REPL[4]:1 [inlined]
 [4] forward(::typeof(baz), ::Float64) at /Users/roger/.julia/dev/YASSAD/src/compiler.jl:0
 [5] foo at ./REPL[2]:1 [inlined]
 [6] forward(::typeof(foo), ::Float64) at /Users/roger/.julia/dev/YASSAD/src/compiler.jl:0
 [7] top-level scope at none:0
```

This is because the `forward` will be recursively called, which also means we only need to define the inner most (primitive) operators by overloading the `forward` functions, e.g we can overload the `*` operator in this case

```julia
julia> forward(::typeof(*), a::Real, b::Real) = a * b, Δ->(Δ*b, a*Δ)

julia> forward(foo, 1.0)
(1.0, YASSAD.Pullback{.....}
```

### Backward Transformation

But this pullback is not callable yet. Let's generate the IR for pullback. Similarly, we can define

```julia
@generated function (::Pullback{S})(delta) where S
    m = IRTools.meta(S)
    m === nothing && return
    ir = IR(m)
    _, pbs = register(ir, S)
    back = adjoint(ir, pbs)
    argnames!(m, Symbol("#self#"), :delta)
    return IRTools.update!(m, back)
end
```

Because the backward pass is called separately, we don't have the forward IR anymore, unfortunately we need to call `register` again here, but no worries, this will only happen once during compile time. Before we generate the IR for adjoint program, we also need to know which variable has pullback, thus instead of using a list, we need a dict to store this, and return it to pullback. Therefore, we need to revise our `register` as following

```julia
function register(ir, F)
    pr = Pipe(ir)
    pbs = Dict{Variable, Variable}()
    argument!(pr, at = 1)
    for (v, st) in pr
        ex = st.expr
        if Meta.isexpr(ex, :call)
            yJ = insert!(pr, v, stmt(xcall(Main, :forward, ex.args...), line = ir[v].line))
            pr[v] = xgetindex(yJ, 1)
            J = insertafter!(pr, v, stmt(xgetindex(yJ, 2), line = ir[v].line))
            pbs[v] = substitute(pr, J)
        end
    end
    pr = finish(pr)
    v = push!(pr, xtuple(values(pbs)...))
    pbv = push!(pr, Expr(:call, Pullback{F}, v))
    ret = pr.blocks[end].branches[end].args[1]
    ret = push!(pr, xtuple(ret, pbv))
    pr.blocks[end].branches[end].args[1] = ret
    return pr, pbs
end
```

since the adjoint program has the reversed order with the original IR, we will not use `Pipe` here, we can create an empty `IR` object,
and add two argument to it here, one is the `Pullback` object itself, the other is the input gradient of the backward pass (pullback).

```julia
adj = empty(ir)
self = argument!(adj)
delta = argument!(adj)
```

First, let's get our pullbacks. The `getfield` function I call here is the lowered form of syntax sugar `.` for getting members, this is equivalent to `self.data`.

```julia
pullbacks = pushfirst!(adj, xcall(:getfield, self, QuoteNode(:data)))
```

Then let's iterate the all the variables in reversed order

```julia
vars = keys(ir)
for k in length(vars):-1:1
    v = vars[k]
    ex = ir[v].expr
    if haskey(pbs, v)
        pbv = insertafter!(adj, pullbacks, xcall(:getindex, pullbacks, k))
        g = push!(adj, Expr(:call, pbv, v))
    end
end
```

if this variable exists in our dict of pullbacks, we get it and call it with this variable. However, there is a problem of this implementation, if one variable has multiple gradient, we need to accumulate them together, thus we need to record these variables'
gradietns as well.

```julia
grads = Dict()
```

Then we can implement two method of `grad`:

```julia
grad(x, x̄) = push!(get!(grads, x, []), x̄)
```

Store the gradient `x̄` in the list of `x` in `grads`.

```julia
grad(x) = xaccum(adj, get(grads, x, [])...)
```

Return the accumulated variable of all gradients.

The `xaccum` is the same as previous `xgetindex`, but the builtin `accumulate` function in Julia is defined on arrays, we need one to accumulate variant variables, let's do it ourselves

```julia
xaccum(ir) = nothing
xaccum(ir, x) = x
xaccum(ir, xs...) = push!(ir, xcall(YASSAD, :accum, xs...))
accum() = nothing
accum(x) = x
accum(x, y) =
  x == nothing ? y :
  y == nothing ? x :
  x + y

accum(x, y, zs...) = accum(accum(x, y), zs...)

accum(x::Tuple, y::Tuple) = accum.(x, y)
accum(x::AbstractArray, y::AbstractArray) = accum.(x, y)
```

In the end, the pullback will return each input variable's gradient of the original program. Which means it always has
the same number of gradients as input variables. But our `forward` function has one extra variable which is the function,
we will return its gradient as well, in most cases, it is `nothing`, but if it is a closure, or a callable object, it may
not be `nothing`.

So, in the end, our `adjoint` function looks like

```julia
function adjoint(ir, pbs)
    adj = empty(ir)
    self = argument!(adj)
    delta = argument!(adj)
    pullbacks = pushfirst!(adj, xcall(:getfield, self, QuoteNode(:data)))

    grads = Dict()
    grad(x, x̄) = push!(get!(grads, x, []), x̄)
    grad(x) = xaccum(adj, get(grads, x, [])...)
    grad(last(keys(ir)), delta)

    vars = keys(ir)
    for k in length(vars):-1:1
        v = vars[k]
        ex = ir[v].expr
        if haskey(pbs, v)
            pbv = insertafter!(adj, pullbacks, xcall(:getindex, pullbacks, k))
            g = push!(adj, Expr(:call, pbv, grad(v)))

            for (i, x) in enumerate(ex.args)
                x isa Variable || continue
                grad(x, push!(adj, xgetindex(g, i)))
            end
        end
    end
    gs = [grad(x) for x in arguments(ir)]
    Δ = push!(adj, xtuple(gs...))
    return!(adj, Δ)
    return adj
end
```

## Conclusion

Let's try this with matrix multiplication + matrix trace, which is the same with what we do in our last post!

Look! we can use the builtin types directly!

```julia
using LinearAlgebra

function forward(::typeof(*), A::Matrix, B::Matrix)
    A * B, function (Δ::Matrix)
        Base.@_inline_meta
        (nothing, Δ * B', A' * Δ)
    end
end

function forward(::typeof(tr), A::Matrix)
    tr(A), function (Δ::Real)
        Base.@_inline_meta
        (nothing, Δ * Matrix(I, size(A)))
    end
end

julia> using LinearAlgebra, BenchmarkTools

julia> mul_tr(A::Matrix, B::Matrix) = tr(A * B)
mul_tr (generic function with 1 method)

julia> A, B = rand(30, 30), rand(30, 30);

julia> mul_tr(A, B)
216.7247235502547

julia> z, back = forward(mul_tr, A, B)；

julia> julia> back(1);
```

The performance is similar to the manual implementation as well (in fact it should be the same)

The manual version is:

```julia
julia> @benchmark bench_tr_mul_base($(rand(30, 30)), $(rand(30, 30)))
BenchmarkTools.Trial: 
  memory estimate:  28.78 KiB
  allocs estimate:  5
  --------------
  minimum time:     10.696 μs (0.00% GC)
  median time:      13.204 μs (0.00% GC)
  mean time:        24.075 μs (43.31% GC)
  maximum time:     62.964 ms (99.97% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

the generated version:

```julia
julia> @benchmark tr_mul($A, $B)
BenchmarkTools.Trial: 
  memory estimate:  36.17 KiB
  allocs estimate:  14
  --------------
  minimum time:     12.921 μs (0.00% GC)
  median time:      15.659 μs (0.00% GC)
  mean time:        27.304 μs (40.97% GC)
  maximum time:     60.141 ms (99.94% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

Now we have implemented a very simple source to source automatic differentiation, but we didn't handle control flow here. A more
complete implementation can be find in `Zygote.jl/compiler`, it can differentiate through almost everything, including: self defined types, control flows, foreign function calls (e.g you can differentiate `PyTorch` functions!), and `in-place` function (mutation support). This also includes part of our quantum algorithm design framework [Yao.jl](https://github.com/QuantumBFS/Yao.jl) with some custom primitives.

Our implementation here only costs 132 lines of code in Julia. Even the complete implementation's compiler only costs 495 lines of code. It is possible to finish in one or a few days!
