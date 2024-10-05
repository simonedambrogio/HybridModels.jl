using Functors
# include("Component.jl");

struct DDC <: AbstractDataDrivenComponent
    params::ComponentParams
end

function Base.show(io::IO, kdc::AbstractKnowledgeDrivenComponent)
    println(io, "\nData-Driven Component")
    display(kdc.params)
end

function DDC(θ)
    DDC(
        ComponentParams(
            typeof(θ)[θ], 
            Symbol[:θ], 
            Function[identity]
        )
    )
end

@functor DDC
