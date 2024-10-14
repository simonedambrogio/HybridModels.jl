# using Functors
# include("Component.jl")

struct KnowledgeDrivenComponent <: AbstractKnowledgeDrivenComponent
    params::ComponentParams
end

function Base.show(io::IO, kdc::KnowledgeDrivenComponent)
    println(io, "\nKnowledge-Driven Component")
    show(io, kdc.params)
end

@functor KnowledgeDrivenComponent
