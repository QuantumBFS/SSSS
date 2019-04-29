include("Core.jl")
include("CMAESCore.jl")
export CMAESIter

# Covariance Matrix Adaptation Evolution Strategy
# ==================
#
# Implementation: (μ/μ_I,λ)-CMA-ES
#
# μ is the number of parents
# λ is the number of offspring.
#

############################ New Interface ############################
# * less input parameters, like number of iterations, tol, verbose.
# * clean `call_back` support, with full access to runtime information.

struct CMAESIter{PT<:AbstractOptProblem, T}
	cr::CMAESRuntime{T}
	prob::PT
end
function CMAESIter(prob::AbstractOptProblem, individual; num_parents::Integer, num_offsprings::Integer, σ::Real=1)
	cr = CMAESRuntime(individual, num_parents=num_parents, num_offsprings=num_offsprings, N=ellength(prob), σ=σ)
	CMAESIter(cr, prob)
end

function CMAESIter(lossfunc::Function, individual; num_parents::Integer, num_offsprings::Integer, σ::Real=1)
	prob = GeneralOptProblem(lossfunc, length(individual))
	CMAESIter(prob, individual, num_parents=num_parents, num_offsprings=num_offsprings, σ=σ)
end

function Base.iterate(ci::CMAESIter, state=1)
	cmaes_step!(ci.cr, ci.prob, τ=init_τ(ci.prob), τ_c=init_τ_c(ci.prob), τ_σ=init_τ_σ(ci.prob)), state+1
end

####################### Problem definition ###############################
function populate!(fitoff::AbstractVector, offspring::AbstractVector, prob::AbstractOptProblem)
    for i in 1:length(fitoff)
        fitoff[i] = loss(prob, offspring[i]) # Evaluate fitness
    end
end
