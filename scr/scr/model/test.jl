path2root = dirname(Base.active_project());
include( joinpath(path2root, "scr", "Modelling", "utils.jl") );
using Functors, StatsFuns, MacroTools, JLD2, HybridModels;
include("𝐷.jl");
X, y = load_object("scr/Modelling/analysis/output/1/test_data.jld2")[1];

begin

    # utils --------------------------------
    using DataFrames
    f⁻(x::T, k::T, t::T) where T = x * exp(-k*t);

    # --- Left --- #
    function n0L(X::𝐷)
        (2f0 .+ X.nL) .* X.initial_visit;
    end;
    function n1_0L(X::𝐷, ω)
        (2f0 .+ (ω .* X.nL) .+ ((1f0.-ω) .* X.nB)) .* X.gR .* X.first_visit;
    end;
    function n1_1L(X::𝐷)
        (2f0 .+ X.nL) .* X.gL .* X.first_visit;
    end;
    function n2_0L(X::𝐷, λ₂)
        (2f0 .+ f⁻.(X.nL, λ₂, X.t)) .* X.gR .* X.after_first_visit;
    end;
    function n2_1L(X::𝐷)
        (2f0 .+ X.nL) .* X.gL .* X.after_first_visit;
    end;

    # r
    function r0L(X::𝐷)
        (2f0 .+ X.nL) .* 0.5f0  .* X.initial_visit;
    end;
    function r1_0L(X::𝐷, λ₀, nL::Vector{T}) where T
        (nL .* (0.5f0 .+ (X.μR .- 0.5f0) .* λ₀)) .* X.gR .* X.first_visit
    end;
    function r1_1L(X::𝐷)
        (1f0 .+ X.rL) .* X.gL .* X.first_visit
    end;
    function r2_0L(X::𝐷, λ₂)
        (1f0 .+ f⁻.(X.rL,λ₂,X.t)) .* X.gR .* X.after_first_visit
    end;
    function r2_1L(X::𝐷)
        (1f0 .+ X.rL) .* X.gL .* X.after_first_visit
    end;

    # --- Right --- #
    # N
    function n0R(X::𝐷)
        (2f0 .+ X.nR) .* X.initial_visit;
    end;
    function n1_0R(X::𝐷, ω)
        (2f0 .+ (ω .* X.nR) .+ (1f0.-ω) .* X.nB) .* X.gL .* X.first_visit
    end;
    function n1_1R(X::𝐷)
        (2f0 .+ X.nR) .* X.gR .* X.first_visit;
    end;
    function n2_0R(X::𝐷, λ₂)
        (2f0 .+ f⁻.(X.nR, λ₂, X.t)) .* X.gL .* X.after_first_visit;
    end;
    function n2_1R(X::𝐷)
        (2f0 .+ X.nR) .* X.gR .* X.after_first_visit;
    end;

    # r
    function r0R(X::𝐷)
        (2f0 .+ X.nR) .* 0.5f0  .* X.initial_visit;
    end;
    function r1_0R(X::𝐷, λ₀, nR::Vector{T}) where T
        (nR .* (0.5f0 .+ (X.μL .- 0.5f0) .* λ₀)) .* X.gL .* X.first_visit
    end;
    function r1_1R(X::𝐷)
        (1f0 .+ X.rR) .* X.gR .* X.first_visit
    end;
    function r2_0R(X::𝐷, λ₂)
        (1f0 .+ f⁻.(X.rR,λ₂,X.t)) .* X.gL .* X.after_first_visit
    end;
    function r2_1R(X::𝐷)
        (1f0 .+ X.rR) .* X.gR .* X.after_first_visit
    end;


end;


macro hybridmodel(expr)
    @capture(expr, function func_name_(X_)
        kdc_block_
        ddc_block_
        body__
    end) || error("Invalid @hybridmodel syntax")

    @capture(kdc_block, @kdc(kdc_params__)) || error("Invalid @kdc syntax")
    @capture(ddc_block, @ddc(ddc_params__)) || error("Invalid @ddc syntax")

    # Process KDC parameters
    kdc_params_dict = Dict{Symbol, Any}()
    for param in kdc_params
        if @capture(param, key_ = value_)
            kdc_params_dict[key] = value
        else
            error("Invalid syntax in @kdc block. Use 'parameter = value' format.")
        end
    end

    kdc_param_names = collect(keys(kdc_params_dict))
    kdc_param_values = [kdc_params_dict[name] for name in kdc_param_names]

    # Process DDC parameters
    ddc_params_dict = Dict{Symbol, Any}()
    for param in ddc_params
        if @capture(param, key_ = value_)
            ddc_params_dict[key] = value
        else
            error("Invalid syntax in @ddc block. Use 'parameter = value' format.")
        end
    end

    ddc_param_names = collect(keys(ddc_params_dict))
    ddc_param_values = [ddc_params_dict[name] for name in ddc_param_names]

    # Create expressions to access parameters from KDC and DDC instances
    kdc_param_access = [:($(name) = m.kdc.params[m.kdc.names .== $(QuoteNode(name))][1]) for name in kdc_param_names]
    ddc_param_access = [:($(name) = m.ddc.params[m.ddc.names .== $(QuoteNode(name))][1]) for name in ddc_param_names]

    hybrid_model = quote
        function (m::Agent)(X)
            $(kdc_param_access...)
            $(ddc_param_access...)
            $(body...)
        end
    end

    result = quote
        $hybrid_model

        kdc_params = KDCParams{Float32}(
            Float32[$(kdc_param_values...)],
            $kdc_param_names  # Changed this line
        )
        ddc_params = DDCParams(
            [$(ddc_param_values...)],
            $ddc_param_names  # Changed this line
        )
        $(func_name) = Agent(kdc_params, ddc_params)

        $(func_name)
    end

    return esc(result)
end


function ucb(x::AbstractMatrix{T}, β::T) where T
    # Check if x has exactly two rows
    if size(x, 1) != 2
        throw(ArgumentError("Input matrix x must have exactly 2 rows, got $(size(x, 1))"))
    end
    
    # Compute Value of Information
    N  = x[1,:]' + x[2,:]'
    Na = x[1,:]'
    return sqrt.( log.(N) ./ Na ) .* β
end;

function mymodel(X)
    # --- Transform and Extract Parameters --- #
    λ₀ = 0.99f0
    ω = 0.60f0
    κ₁ = 0.25f0
    λ₂ = 0.01f0
    τ = 0.08f0 
    β = 1.2f0

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;


@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0 ω = 0.60f0 κ₁ = 0.25f0 λ₂ = 0.01f0 τ = 0.08f0 
    @ddc β = 1.2f0

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;

mymodel(X)


@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc begin
        λ₀ =  (randn(Float32)) |> abs
        ω  =  (randn(Float32)) |> abs
        κ₁ = -(rand(Float32) + 1)
        λ₂ = -(rand(Float32) + 4)
        τ  = -(rand(Float32) - 1)
    end

    @ddc β = 1.2f0

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    voiL = ucb([nL nR]', β)'
    voiR = ucb([nR nL]', β)'

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL .- coiL) (voiR .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;


using Random, RobustNeuralNetworks
input_dim = 2;
rng = Xoshiro();
ny = 1
nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
γ = 5  



@hybridmodel function mymodel(X)
    # --- Transform and Extract Parameters --- #
    @kdc λ₀ = 0.99f0 ω = 0.60f0 κ₁ = 0.25f0 λ₂ = 0.01f0 τ = 0.08f0 
    @ddc θ = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng)

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = n0L(X) .+ n1_0L(X, ω)      .+ n1_1L(X) .+ n2_0L(X, λ₂) .+ n2_1L(X)
    rL = r0L(X) .+ r1_0L(X, λ₀, nL) .+ r1_1L(X) .+ r2_0L(X, λ₂) .+ r2_1L(X)

    nR = n0R(X) .+ n1_0R(X, ω)      .+ n1_1R(X) .+ n2_0R(X, λ₂) .+ n2_1R(X)
    rR = r0R(X) .+ r1_0R(X, λ₀, nR) .+ r1_1R(X) .+ r2_0R(X, λ₂) .+ r2_1R(X)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Value of Information --- #    
    nn = LBDN(θ)
    voiL = nn([nL nR]')
    voiR = nn([nR nL]')

    # --- Cost of Information --- #    
    coiL = (1f0 .- X.gL) .* κ₁
    coiR = (1f0 .- X.gR) .* κ₁

    return [(voiL' .- coiL) (voiR' .- coiR) (ρL .- ρR) (ρR .- ρL)]' ./ τ
end;

mymodel(X)

mymodel.kdc

typeof(mymodel)

# train!(mymodel)