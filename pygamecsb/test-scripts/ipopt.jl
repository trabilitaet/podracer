using Ipopt, JuMP

Np = 5
r0 = [10,10]
v0 = [0,0]
phi0 = 0
r1 = [100,100]


m = Model(with_optimizer(Ipopt.Optimizer))

@variable(m,rx[1:Np], start = 0)
@variable(m,ry[1:Np], start = 0)
@variable(m,phi[1:Np], start = 0)
@variable(m,vx[1:Np], start = 0)
@variable(m,vy[1:Np], start = 0)
@variable(m,0 <= a[1:Np] <= 100, start = 0)
@variable(m,-pi/10 <= w[1:Np] <= pi/10, start = 0)


@NLobjective(m, Min, sum((r1[1]-rx[j])^2+(r1[2]-rx[j])^2 for j=1:Np))

@NLconstraint m begin
    rx[2]-rx[1]-vx[1] == 0
    rx[3]-rx[2]-vx[2] == 0
    rx[4]-rx[3]-vx[3] == 0
    rx[5]-rx[4]-vx[4] == 0
    ry[2]-ry[1]-vy[1] == 0
    ry[3]-ry[2]-vy[2] == 0
    ry[4]-ry[3]-vy[3] == 0
    ry[5]-ry[4]-vy[4] == 0
	phi[2]-phi[1]-w[1] == 0    
	phi[3]-phi[2]-w[2] == 0    
	phi[4]-phi[3]-w[3] == 0    
	phi[5]-phi[4]-w[4] == 0
	vx[2]-0.85*vx[1]-0.85*a[1]*cos(phi[1]) == 0
	vx[3]-0.85*vx[2]-0.85*a[2]*cos(phi[2]) == 0
	vx[4]-0.85*vx[3]-0.85*a[3]*cos(phi[3]) == 0
	vx[5]-0.85*vx[4]-0.85*a[4]*cos(phi[4]) == 0
	vy[2]-0.85*vy[1]-0.85*a[1]*sin(phi[1]) == 0
	vy[3]-0.85*vy[2]-0.85*a[2]*sin(phi[2]) == 0
	vy[4]-0.85*vy[3]-0.85*a[3]*sin(phi[3]) == 0
	vy[5]-0.85*vy[4]-0.85*a[4]*sin(phi[4]) == 0
	rx[1] == r0[1]
	phi[1] == phi0
	ry[1] == r0[2]
	vx[1] == v0[1]
	vy[1] == v0[2]
end

JuMP.optimize!(m)
println("done")