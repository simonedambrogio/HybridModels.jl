# __precompile__(false)

module HybridModels
    
    import Functors
    import MacroTools
    include("AbstractTypes.jl")
    include("Component.jl")
    include("KDC.jl")
    include("DDC.jl")
    include("HybridModel.jl")
    include("@kdc.jl")
    include("@ddc.jl")
    include("@hybridmodel.jl")
    
    export HybridModel, KDC, DDC, @hybridmodel, @kdc, @ddc, predict!
end

