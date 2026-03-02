# Box null regularizer
export NullRegularizerBox

@doc raw"""
    NullRegularizerBox(l, u)
    NullRegularizerBox(::Type{T}, n)


Returns the box null regularizer, i.e., the function that is identically zero on a box and +∞ outside of it.
```math
h(x) = \chi(x | [l, u])
```

### Arguments
- `l`: The lower bound of the box.
- `u`: The upper bound of the box.

In the second constructor, the bounds are vectors of type T and size n set to -Inf and +Inf, respectively. 
"""
mutable struct NullRegularizerBox{T <: Real, V <: AbstractVector{T}} 
    l::V
    u::V
end

NullRegularizerBox(::Type{T}, n::Integer) where {T <: Real} = NullRegularizerBox(fill(T(-Inf), n), fill(T(Inf), n))

function (h::NullRegularizerBox{T})(y) where {T <: Real}
  ϵ = √eps(eltype(y))
  @inbounds for i in eachindex(y)
    if !(h.l[i] - ϵ ≤ y[i] ≤ h.u[i] + ϵ)
      return T(Inf)
    end
  end
  return zero(T)
end

fun_name(ψ::NullRegularizerBox{T}) where {T <: Real} = "box null regularizer"
fun_expr(ψ::NullRegularizerBox{T}) where {T <: Real} = "t ↦ χ(t | [l, u])"
fun_params(ψ::NullRegularizerBox{T}) where {T <: Real} = 
  "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function Base.show(io::IO, ψ::NullRegularizerBox{T}) where {T <: Real}
  println(io, "description : ", fun_name(ψ))
  println(io, "expression  : ", fun_expr(ψ))
  println(io, "parameters  : ", fun_params(ψ))
end

function prox!(
  y::AbstractVector{T},
  h::NullRegularizerBox{T},
  q::AbstractVector{T},
  ν::T
) where {T <: Real}
  @assert ν > zero(T)
  @inbounds for i in eachindex(y)
    y[i] = prox_zero(q[i], h.l[i], h.u[i])
  end
  return y
end

function iprox!(
  y::AbstractVector{T},
  h::NullRegularizerBox{T},
  g::AbstractVector{T},
  d::AbstractVector{T},
) where {T <: Real}
  @inbounds for i in eachindex(y)
    y[i] = iprox_zero(d[i], g[i], h.l[i], h.u[i])
  end
  return y
end