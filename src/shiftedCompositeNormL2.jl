export ShiftedCompositeNormL2

@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b)

Returns the shift of a function c composed with the ``\ell_{2}`` norm (see CompositeNormL2.jl).
Here, c is linearized i.e, ``c(x+s) \approx c(x) + J(x)s``. 
```math
f(s) = λ \|c(x) + J(x)s\|_2
```
where ``\lambda > 0``. c! and J! should be functions
```math
\begin{aligned}
&c(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^m \\
&J(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^{m\times n}
\end{aligned}
```
such that J is the Jacobian of c. A and b should respectively be a matrix and a vector which can respectively store the values of J and c.
"""
mutable struct ShiftedCompositeNormL2{
  R <: Real,
  V0 <: Function,
  V1 <: Function,
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R},
} <: ShiftedCompositeProximableFunction
  h::NormL2{R}
  c!::V0
  J!::V1
  A::V2
  b::V3
  g::V3
  res::V3
  sol::V3
  dsol::V3
  function ShiftedCompositeNormL2(
    λ::R,
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
  ) where {R <: Real}
    g = similar(b)
    res = similar(b)
    sol = similar(b)
    dsol = similar(b)
    if length(b) != size(A,1)
      error("ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b)}(NormL2(λ),c!,J!,A,b,g,res,sol,dsol)
  end
end


shifted(λ::R, c!::Function, J!::Function, A::AbstractMatrix{R}, b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} = begin
  c!(b,xk)
  J!(A,xk)
  ShiftedCompositeNormL2(λ,c!,J!,A,b)
end

shifted(
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3<: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h.lambda, ψ.c!, ψ.J!, A, b)
end
 
shifted(
  ψ::CompositeNormL2{R, V0, V1, V2, V3},
  xk::AbstractVector{R}
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h.lambda, ψ.c!, ψ.J!, A, b)
end

fun_name(ψ::ShiftedCompositeNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}}

  mul!(ψ.g, ψ.A, q)
  ψ.g .+= ψ.b

  #ψ.res .= ψ.g
  spmat = qrm_spmat_init(ψ.A; sym=false)
  spfct = qrm_spfct_init(spmat)
  qrm_analyse!(spmat, spfct; transp='t')
  qrm_set(spfct, "qrm_keeph", 0)
  qrm_factorize!(spmat, spfct, transp='t')

  qrm_solve!(spfct, ψ.g, y, transp='t')
  qrm_solve!(spfct, y, ψ.sol, transp='n')

  """
  # 1 step of iterative refinement

  mul!(y, ψ.A', ψ.sol)
  mul!(ψ.dsol, ψ.A, y)

  ψ.res .-= ψ.dsol

  if norm(ψ.res) > eps(R)^0.75
    qrm_solve!(spfct, ψ.res, y, transp='t')
    qrm_solve!(spfct, y, ψ.dsol, transp='n')
    ψ.sol .+= ψ.dsol
  end  
  """

  ψ.sol .*= -1

  # Scalar Root finding
  α = 0.0
  while norm(ψ.sol) >= σ*ψ.h.lambda

    α += (norm(ψ.sol)/σ*ψ.h.lambda - 1.0)*(norm(y)/norm(ψ.sol))^2
    ψ.sol .*= -1
    
  end

  mul!(y, ψ.A', ψ.sol)
  y .+= q
  return y
end