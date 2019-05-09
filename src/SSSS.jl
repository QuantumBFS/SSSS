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

# REQUIRE = [
#     "GR",
#     "PyCall",
#     "IJulia",
#     "Revise",
#     "Plots",
#     "Latexify",
#     "FFTW",
#     PackageSpec(name="Flux", rev="master"),
#     "BitBasis",
#     "KrylovKit",
#     PackageSpec(url="https://github.com/QuantumBFS/QuAlgorithmZoo.jl.git", rev="master"),
#     PackageSpec(name="IRTools", rev="master"),
#     PackageSpec(name="NNlib", rev="master"),
#     PackageSpec(name="Zygote", rev="master"),
#     PackageSpec(name="Yao", rev="master"),
#     PackageSpec(name="YaoBlocks", rev="master"),
#     PackageSpec(name="YaoArrayRegister", rev="master"),
# ]

# function __init__()
#     for each in REQUIRE
#         if each in keys(Pkg.installed())
#             continue
#         else
#             Pkg.add(each)
#         end
#     end
# end

end # module
