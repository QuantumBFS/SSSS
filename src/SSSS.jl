module SSSS

using IJulia, Pkg

export notebooks

# function tutorial()
#     cmd = IJulia.find_jupyter_subcommand("notebook")
#     push!(cmd.exec, joinpath(@__DIR__, "..", "notebooks", "tutorial.ipynb"))
#     return IJulia.launch(cmd, joinpath(@__DIR__, "..", "notebooks"), false)
# end

function notebooks()
    return IJulia.notebook(dir=joinpath(@__DIR__, ".."))
end

REQUIRE = [
    "GR",
    "PyCall",
    "IJulia",
    "Revise",
    "Interact",
    "Plots",
    "Latexify",
    "FFTW",
    "Flux",
    "BitBasis",
    "KrylovKit",
    "https://github.com/QuantumBFS/QuAlgorithmZoo.jl.git",
    "Zygote#master",
    "Yao#master",
    "YaoBlocks#master",
    "YaoArrayRegister#master",
]

function __init__()
    for each in REQUIRE
        if each in keys(Pkg.installed())
            continue
        else
            Pkg.add(each)
        end
    end
end

end # module
