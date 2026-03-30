include(joinpath(@__DIR__, "main_iterative_algorithm.jl"))

function main_iterative_banknote(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main_iterative(dataset_names = ["banknote"], time_limit = time_limit, depths = depths)
end
