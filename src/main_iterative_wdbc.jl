include(joinpath(@__DIR__, "main_iterative_algorithm.jl"))

function main_iterative_wdbc(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main_iterative(dataset_names = ["wdbc"], time_limit = time_limit, depths = depths)
end
