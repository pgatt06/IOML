include(joinpath(@__DIR__, "main_merge.jl"))

function main_merge_wbc(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main_merge(dataset_names = ["wdbc"], time_limit = time_limit, depths = depths)
end

function main_merge_wdbc(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main_merge_wbc(time_limit = time_limit, depths = depths)
end
