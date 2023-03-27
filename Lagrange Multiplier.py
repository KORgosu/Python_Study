import autograd.numpy as np
from autograd import grad
import numpy as np
from scipy.optimize import minimize
def objective(X):
    x, y, z = X
    return 3*x + 2*y**3 - 3*z # f(X)
def eq1(X):
    x, y, z = X
    return x - 2*y + z -1 # g(X)
def eq2(X):
    x, y, z = X
    return x**2 + y**2 -4 # h(X)
def F(L):
    'Augmented Lagrange function'
    x, y, z, _lambda, _mu = L
    return objective([x, y, z]) - _lambda * eq1([x, y, z]) - _mu * eq2([x ,y, z])
# Gradients of the Lagrange function
dfdL = grad(F, 0)
# Find L that returns all zeros in this function.
def obj(L):
    x, y, z, _lambda, _mu = L
    dFdx, dFdy, dFdz, dFdlam, dFdmu = dfdL(L)
    return [dFdx, dFdy, dFdz, eq1([x, y, z]), eq2([x, y, z])]

from scipy.optimize import fsolve
x1, y1, z1, _lam, _mu = fsolve(obj, [0.0, 0.0, 0.0, 1.0, 1.0])
x2, y2, z2, _lam, _mu = fsolve(obj, [0.0, 0.0, 0.0, -1.0, -1.0])
print(f'라그랑즈 곱수를 이용한 최적점 하나는 {x1, y1, z1}')
print(f'라그랑즈 곱수를 이용한 최적점 다른 하나는 {x2, y2, z2}')