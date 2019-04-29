export AbstractOptProblem, AbstractRuntime, loss, ellength, rand_individual, AbstractRuntime, rosenbrock, GeneralOptProblem
"""
    AbstractOptProblem

optimization problem, required interfaces are `ellength`, `loss`.
optional interfaces are `rand_individual`.
"""
abstract type AbstractOptProblem end

"""
    ellength(prob::AbstractOptProblem) -> Int

Length of an individual as the input of an optimization problem.
"""
function ellength end

"""
    loss(prob::AbstractOptProblem, x) -> Real

Loss function for an optimization problem.
"""
function loss end

struct GeneralOptProblem{FT} <: AbstractOptProblem
	lossfunc::FT
	N::Int
end
ellength(opt::GeneralOptProblem) = opt.N
loss(opt::GeneralOptProblem, x) = opt.lossfunc(x)

"""randomly generate an individual"""
rand_individual(prob::AbstractOptProblem) = randn(prob |> ellength)

"""
    AbstractRuntime

runtime information for optimization, required interfaces are `best`.
"""
abstract type AbstractRuntime end

"""
    best(prob::AbstractRuntime) -> (individual, cost)

The best suited individual, and its cost.
"""
function best end

rosenbrock(x::Vector{Float64}) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
