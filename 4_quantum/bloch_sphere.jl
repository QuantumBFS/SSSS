using Makie, Yao, Colors

function bloch_arrow!(scene, x::AbstractBlock{1})
    r = ArrayReg(bit"0")
    bloch_arrow!(scene, r; color=:red)
    rand_color = RGB(rand(0:255)/255, rand(0:255)/255, rand(0:255)/255)
    # rand_color = rand(colormap("Blues", 100)[50:end])
    bloch_arrow!(scene, apply!(r, x); color=rand_color)
    return scene
end

function bloch_arrow!(scene, r::ArrayReg; kwargs...)
    @assert nqubits(r) == 1 "invalid quantum state, expect only one qubit"
    st = statevec(r)
    global_phase = exp(im * angle(st[1]))
    st = st ./ global_phase
    θ = acos(real(st[1])) # st[1] is real
    ϕ = iszero(θ) ? zero(θ) : angle(st[2] / sin(θ))
    bloch_arrow!(scene, θ, ϕ; kwargs...)
    return scene
end


function bloch_sphere!(scene)
    wireframe!(scene, Sphere(Point3f0(0), 1f0), color=:cyan3, linewidth=0.1, thickness = 0.6f0, transparency = true)
    return scene
end

function bloch_arrow!(scene, θ, ϕ; color=:red)
    x = sin(2θ) * cos(ϕ)
    y = sin(2θ) * sin(ϕ)
    z = cos(2θ)

    lines!(scene, Float64[0.0, x], Float64[0.0, y], Float64[0.0, z], linewidth=5.0, color=color)
    return scene
end

scene = Scene(resolution=(1000, 1000))
bloch_sphere!(scene)
rotate_cam!(scene, 0.5, 0.0, 0.0)
scene.center = false
bloch_arrow!(scene, ArrayReg(bit"0"))

for _ in 1:10
    sleep(0.01)
    bloch_arrow!(scene, Rx(rand()))
end
