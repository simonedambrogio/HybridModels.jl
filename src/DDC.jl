using Functors
# include("Component.jl");

struct DataDrivenComponent <: AbstractDataDrivenComponent
    params::ComponentParams
end

function Base.show(io::IO, ddc::DataDrivenComponent)
    println(io, "\nData-Driven Component")
    show(io, ddc.params)
end

function DataDrivenComponent(θ)
    DataDrivenComponent(
        ComponentParams(
            typeof(θ)[θ], 
            Symbol[:θ], 
            Function[identity]
        )
    )
end

@functor DataDrivenComponent
