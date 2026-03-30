include(joinpath(@__DIR__, "main.jl"))

function main_banknote(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main(dataset_names = ["banknote"], time_limit = time_limit, depths = depths)
end
