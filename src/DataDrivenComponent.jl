using Functors
# include("Component.jl");

struct DataDrivenComponent <: AbstractDataDrivenComponent
    params::ComponentParams
    
    # Inner constructor
    DataDrivenComponent(params::ComponentParams) = new(params)
end

function Base.show(io::IO, ddc::DataDrivenComponent)
    println(io, "\nData-Driven Component")
    if typeof(ddc.params.params) <: AbstractVector{<:AbstractFloat}
        show(io, ddc.params)
    else
        println(io, "")
    end
end

# Outer constructor
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
