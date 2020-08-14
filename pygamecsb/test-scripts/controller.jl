using Ipopt
using JuMP
using PyCall
using PyPlot


xstart = [1000, 500]
#xgoal = [0, 0]

m = Model(optimizer_with_attributes(Ipopt.Optimizer,"max_iter" => 1000))

Np = 20
Nx = 5
Nc = 3 #x,y,ψ
Nu = 2
β = 0.85

#Dynamics
@variable(m, x[1:Np, 1:Nx])
@variable(m, u[1:(Np-1), 1:Nu])

@constraint(m, pos[k=1:(Np-1), v=1:2],		x[k+1, v] == x[k, v] + x[k, v + Nc])
@NLconstraint(m, velx[k=1:(Np-2), v=1], x[k+1, v + Nc] == β*(x[k, v+Nc] + u[k,1]*cos(x[k,3])))

#Initial condition
x0 = zeros(Nx)
u0 = zeros(Nu)
@constraint(m, [v=1:Nx], x[1, v] == x0[v])
@constraint(m, [v=1:Nu], u[1, v] == u0[v])

@objective(m, Min, sum([(x[end,i] - xstart[i]).^2 for i = 1:2]))

@time optimize!(m)

xv = value.(x)

plot(xv[:,1], xv[:,2], "x")
