# Construct a neural network using Flux
using Random, RobustNeuralNetworks, ProgressBars, JSON;
# input_dim = 2;
rng = Xoshiro();
ny = 1
nh = fill(32,4)   # 4 hidden layers, each with 32 neurons
γ = 5             # Start with a Lipschitz bound of 5

println("Network Info: ")
println("\t $(length(nh)) hidden layers, each with $(nh[1]) neurons\n")

function (m::AbstractAgent)(X, idxinput::BitVector)

    # --- Tranform and Extract Parameters --- #
    c = transformpars(m);
    
    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = @inline n0L(m, X, c) .+ n1_0L(m, X, c) .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = @inline r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = @inline n0R(m, X, c) .+ n1_0R(m, X, c) .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = @inline r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Bonus of Sampling --- #    
    nn = LBDN(m.θ)
    # input = Matrix([(nL.-50)./50 (nR.-50)./50 (X.nB.-50)./50 ρL ρR X.gL X.first_visit]')[idxinput, :]
    input = Matrix([nL./100 nR./100 X.nB./100 ρL ρR X.gL X.first_visit]')[idxinput, :]
    bonusL = nn(input)'
    
    # input = Matrix([(nR.-50)./50 (nL.-50)./50 (X.nB.-50)./50 ρR ρL X.gR X.first_visit]')[idxinput, :]
    input = Matrix([nR./100 nL./100 X.nB./100 ρR ρL X.gR X.first_visit]')[idxinput, :]
    bonusR = nn(input)'

    # --- Cost of Sampling --- #    
    costL = (1 .- X.gL) .* c.κ₁
    costR = (1 .- X.gR) .* c.κ₁

    [(bonusL .- costL) (bonusR .- costR) (ρL .- ρR) (ρR .- ρL)]' ./ c.τ
end;

initializepars(input_dim::Signed) = DenseLBDNParams{Float32}(input_dim, nh, ny, γ; nl=Flux.tanh, learn_γ=true, rng);

# var2remove = "none"
function extract(m::AbstractAgent, X, idxinput::BitVector; type=:long)

    # --- Tranform and Extract Parameters --- #
    c = transformpars(m);

    # --- Compute α and β to establish the shape of the beta posterior distribution --- #
    nL = @inline n0L(m, X, c) .+ n1_0L(m, X, c)     .+ n1_1L(m, X, c) .+ n2_0L(m, X, c) .+ n2_1L(m, X, c)
    rL = @inline r0L(m, X, c) .+ r1_0L(m, X, c, nL) .+ r1_1L(m, X, c) .+ r2_0L(m, X, c) .+ r2_1L(m, X, c)

    nR = @inline n0R(m, X, c) .+ n1_0R(m, X, c)     .+ n1_1R(m, X, c) .+ n2_0R(m, X, c) .+ n2_1R(m, X, c)
    rR = @inline r0R(m, X, c) .+ r1_0R(m, X, c, nR) .+ r1_1R(m, X, c) .+ r2_0R(m, X, c) .+ r2_1R(m, X, c)

    # --- Value of Select --- #    
    ρL = rL ./ nL
    ρR = rR ./ nR

    # --- Bonus of Sampling --- #    
    σL = sqrt.( (rL .* (nL .- rL)) ./ ((nL .^ 2) .* (nL .+ 1)) ) .* 12;
    σR = sqrt.( (rR .* (nR .- rR)) ./ ((nR .^ 2) .* (nR .+ 1)) ) .* 12;

    nn = LBDN(m.θ);
    input = Matrix([nL./100 nR./100 X.nB./100 ρL ρR X.gL X.first_visit]')[idxinput, :]
    bonusL = nn(input)'
    
    input = Matrix([nR./100 nL./100 X.nB./100 ρR ρL X.gR X.first_visit]')[idxinput, :]
    bonusR = nn(input)'

    # --- Cost of Sampling --- #    
    costL = (1 .- X.gL) .* c.κ₁
    costR = (1 .- X.gR) .* c.κ₁

    QsamplingL = (bonusL .- costL)
    QsamplingR = (bonusR .- costR)

    if type==:long
        (;
            voi = vcat( bonusL[:], bonusR[:] ),
            sampling = vcat( QsamplingL[:], QsamplingR[:] ), #vcat( Vector{Float32}(QsamplingL), Vector{Float32}(QsamplingR)),
            σA = vcat( Vector{Float32}(σL), Vector{Float32}(σR)),
            σB = vcat( Vector{Float32}(σR), Vector{Float32}(σL)),
            nA = vcat( Vector{Float32}(nL), Vector{Float32}(nR)),
            nB = vcat( Vector{Float32}(nR), Vector{Float32}(nL)),
            nC = vcat( Vector{Float32}(X.nB), Vector{Float32}(X.nB)),
            ρA = vcat( Vector{Float32}(ρL), Vector{Float32}(ρR)),
            ρB = vcat( Vector{Float32}(ρR), Vector{Float32}(ρL)), 
            gA = vcat( Vector{Float32}(X.gL), Vector{Float32}(X.gR)) .|> Bool, 
            fv = vcat( Vector{Float32}(X.first_visit), Vector{Float32}(X.first_visit) ) .|> Bool
        )
    elseif type==:wide

        (;
            voiL       = bonusL[:], 
            voiR       = bonusR[:],
            voiLucb    = σL[:], 
            voiRucb    = σR[:],
            coiL       = costL[:], 
            coiR       = costR[:], 
            QsamplingL = QsamplingL[:], 
            QsamplingR = QsamplingR[:],
            nL         = nL,
            nR         = nR,
            ρL         = ρL,
            ρR         = ρR,
            gL         = X.gL .|> Bool, 
            gR         = X.gR .|> Bool, 
            fv         = X.first_visit .|> Bool
        )

    end

end;
