include(joinpath(@__DIR__, "main_merge.jl"))

function main_merge_banknote(; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    main_merge(dataset_names = ["banknote"], time_limit = time_limit, depths = depths)
end
