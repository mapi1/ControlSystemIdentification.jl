G1 =  tf([0.3,0, 1], [1, -0.5, 0], 1)
G1 =  tf([0.3, 1], [1, -0.5], 1)
u = randn(N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
na, nb = 1,1
Gest = arx(d, na, nb, direct = true, inputdelay = 1)
Gest ≈ G1

direct = true
yx, A = getARXregressor(y, u, na, nb, direct = direct, inputdelay = 1)

w = A\yx
a, b = params2poly(w, na ,nb, direct = direct, inputdelay = 1)
Gest = tf(b, a, 1)

Ad = [A u[2:end] u[2:end]]
yx
Ad\yx

T = 1000
G = tf(1, [1, 2 * 0.1 * 1, 1])
G = c2d(G, 1)
u = randn(T)
y = lsim(G, u, 1:T)[1][:]
d = iddata(y)
model = arma_ssa(d, 2, 2, L = 200)

@test numvec(model)[1] ≈ numvec(G)[1] atol = 0.7
@test denvec(model)[1] ≈ denvec(G)[1] atol = 0.2
@test freqresptest(G, model) < 2

N = 2000
t = 1:N
u = randn(N)
G = tf(0.8, [1, -0.9], 1)
y = lsim(G, u, t)[1][:]
e = randn(N)
yn = y + e

na, nb, nc = 1, 1, 1
d = iddata(yn, u, 1)
find_na(y, 6)
find_nanb(d, 6, 6)
Gls = arx(d, na, nb)
Gtls = arx(d, na, nb, estimator = tls)
Gwtls = arx(d, na, nb, estimator = wtls_estimator(y, na, nb))
Gplr, Gn = plr(d, na, nb, nc, initial_order = 10)

# @show Gplr, Gn

@test freqresptest(G, Gls) < 1.5
@test freqresptest(G, Gtls) < 1
@test freqresptest(G, Gwtls) < 0.1
@test freqresptest(G, Gplr) < 0.1

d = iddata(y, u, 1)
Gls = arx(d, na, nb)
Gtls = arx(d, na, nb, estimator = tls)
Gwtls = arx(d, na, nb, estimator = wtls_estimator(y, na, nb))
Gplr, Gn = plr(d, na, nb, nc, initial_order = 10)

@test freqresptest(G, Gls) < sqrt(eps())
@test freqresptest(G, Gtls) < sqrt(eps())
@test freqresptest(G, Gwtls) < sqrt(eps())
@test freqresptest(G, Gplr) < sqrt(eps())