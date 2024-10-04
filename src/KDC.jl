# using Functors
# include("Component.jl")

struct KDC <: AbstractKnowledgeDrivenComponent
    params::ComponentParams
end

@functor KDC
