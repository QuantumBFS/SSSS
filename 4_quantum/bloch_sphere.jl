using Makie, GLMakie, Observables, Colors, Yao
AbstractPlotting.inline!(true)

function bloch_arrow(θ, ϕ)
    x = sin(2θ) * cos(ϕ)
    y = sin(2θ) * sin(ϕ)
    z = cos(2θ)
    return (x, y, z)
end

function bloch_arrow(r::ArrayReg)
    @assert nqubits(r) == 1 "invalid quantum state, expect only one qubit"
    st = statevec(r)
    global_phase = exp(im * angle(st[1]))
    st = st ./ global_phase
    θ = acos(real(st[1])) # st[1] is real
    ϕ = iszero(θ) ? zero(θ) : angle(st[2] / sin(θ))
    return bloch_arrow(θ, ϕ)
end

function plot_arrow!(scene, r::ArrayReg; kwargs...)
    x, y, z = bloch_arrow(r)
    lines!(scene, [0.0, x], [0.0, y], [0.0, z]; kwargs...)
    return scene
end

function plot_block!(scene, x::AbstractBlock; linewidth=5.0, kwargs...)
    r = ArrayReg(bit"0")
    plot_arrow!(scene, r; color=:red, linewidth=linewidth, kwargs...)
    rand_color = RGB(rand(0:255)/255, rand(0:255)/255, rand(0:255)/255)
    plot_arrow!(scene, apply!(r, x); color=rand_color ,linewidth=linewidth, kwargs...)
    return scene
end

function bloch_sphere!(scene)
    wireframe!(scene, Sphere(Point3f0(0), 1f0), color=:cyan3, linewidth=0.1, thickness = 0.6f0, transparency = true)
    return scene
end


function plotapply(x::AbstractBlock)
    AbstractPlotting.inline!(false)
    scene = Scene(resolution=(1000, 1000), center=false)
    bloch_sphere!(scene)
    plot_block!(scene, x)
    rotate_cam!(scene, 0.5, 0.0, 0.0)
    return scene
end

function animate_rot()
    scene = Scene(resolution=(1000, 1000))
    bloch_sphere!(scene)
    scene.center = false

    time = Node(0.0)
    scene = lift()
    bloch_arrow!(scene, x)
    rotate_cam!(scene, 0.5, 0.0, 0.0)
end

# Pauli Gates
function animate_rot(blk::AbstractBlock)
    AbstractPlotting.inline!(false)
    screen = GLMakie.Screen()
    GLMakie.GLFW.set_visibility!(GLMakie.to_native(screen), AbstractPlotting.use_display[])

    AbstractPlotting.inline!(true)
    scene = Scene(resolution=(1000, 1000), center=false)


    bloch_sphere!(scene)

    r = ArrayReg(bit"0")
    plot_arrow!(scene, r; color=:red, linewidth=5.0)
    time = Node(0.0)

    lifted = lift(time) do t
        x, y, z = bloch_arrow(apply!(r, rot(blk, t)))
        [0.0, x], [0.0, y], [0.0, z]
    end

    lines!(scene, lifted; color=:blue, linewidth=5.0)
    rotate_cam!(scene, 0.1, 0.0, 0.0)

    for _ in 1:100
        sleep(0.1)
        push!(time, to_value(time) + 0.001)
        display(screen, scene)
    end
    return scene
end

# 1. plot pauli gates
plotapply(X)

# 2. plot rotate animation
scene = animate_rot(chain(X, H, X))

# 3. make video
scene = Scene(resolution=(1000, 1000), center=false)
bloch_sphere!(scene)
r = ArrayReg(bit"0")
plot_arrow!(scene, r; color=:red, linewidth=5.0)

T = Node(0.0)
lifted = lift(T) do t
    x, y, z = bloch_arrow(apply!(r, rot(X, t)))
    [0.0, x], [0.0, y], [0.0, z]
end

lines!(scene, lifted; color=:blue, linewidth=5.0)
rotate_cam!(scene, 0.1, 0.0, 0.0)

record(scene, "bloch.gif", 1:100) do i
    push!(T, 0.001 * i)
end
