using Functors
# include("Component.jl");

struct DDC <: AbstractDataDrivenComponent
    params::ComponentParams
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
