# using Functors
# include("Component.jl")

struct KDC <: AbstractKnowledgeDrivenComponent
    params::ComponentParams
end

function Base.show(io::IO, kdc::AbstractKnowledgeDrivenComponent)
    println(io, "\nKnowledge-Driven Component")
    display(kdc.params)
end

@functor KDC
