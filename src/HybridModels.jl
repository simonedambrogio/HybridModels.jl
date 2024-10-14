# __precompile__(false)

module HybridModels
    
    import Functors
    import MacroTools
    include("AbstractTypes.jl")
    include("Component.jl")
    include("KnowledgeDrivenComponent.jl")
    include("DataDrivenComponent.jl")
    include("HybridModel.jl")
    include("@kdc.jl")
    include("@ddc.jl")
    include("@hybridmodel.jl")
    
    export HybridModel, KnowledgeDrivenComponent, DataDrivenComponent, @hybridmodel, @kdc, @ddc, predict!
end

