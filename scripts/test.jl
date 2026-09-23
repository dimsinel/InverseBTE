using DrWatson
@quickactivate "InverseBTE"

 
import DifferentialEquations as DE
f(u, p, t) = 1.01 * u^(1/3)
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = DE.ODEProblem(f, u0, tspan)
sol = DE.solve(prob, DE.Tsit5(), reltol = 1e-8, abstol = 1e-8)

import Plots
Plots.plot(sol, linewidth = 5, title = "Solution to the linear ODE with a thick line",    xaxis = "Time (t)", yaxis = "u(t) (in μm)", label = "My Thick Line!") # legend=false
Plots.plot!(sol.t, t -> 0.5 * exp(1.01t), lw = 3, ls = :dash, label = "True Solution!")


function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = DE.ODEProblem(lorenz!, u0, tspan)
sol = DE.solve(prob)


Plots.plot(sol, idxs = (1, 2, 3))


using NeuralPDE


linear(u, p, t) = cos(t * 2 * pi)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0, tspan)


using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)
chain = Chain(Dense(1, 5, σ), Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64

using OptimizationOptimisers

opt = Adam(0.1)
alg = NNODE(chain, opt, init_params = ps)

sol = solve(prob, alg, verbose = true, maxiters = 2000, saveat = 0.01)



