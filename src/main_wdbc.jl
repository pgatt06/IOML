include(joinpath(@__DIR__, "main.jl"))

function main_wdbc(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main(dataset_names = ["wdbc"], time_limit = time_limit, depths = depths)
end
