# Jonathan
'''
def ActorCritic_Learn(pi, V_phi, n, eta, M): \\maybe met bootstrapping
    initialize theta, phi
    while not converged:
        grad = 0
        for m in range(M)
            sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
            for t in range(T)
                Q(s,a) = sum(r_t+k + V_phi(s_t+n)^2) van k=0 tot n-1
        phi = phi - eta * rho som(Q(s,a)-V_phi(s_t)^2)
        phi = phi - eta * rho som(Q(s,a)-V_phi(s_t)^2)
    return pi
'''
