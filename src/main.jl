include(joinpath(@__DIR__, "building_tree.jl"))
include(joinpath(@__DIR__, "utilities.jl"))

function print_exact_result(label::String, tree, resolution_time, gap, dataset)
    print(label, "\t")
    print(round(resolution_time, digits = 1), "s\t")
    print("gap ", round(gap, digits = 1), "%\t")

    if tree === nothing
        println("no feasible solution")
        return
    end

    train_errors = prediction_errors(tree, dataset.X_train, dataset.Y_train, dataset.classes)
    test_errors = prediction_errors(tree, dataset.X_test, dataset.Y_test, dataset.classes)
    println("train/test errors ", train_errors, "/", test_errors)
end

function run_exact_dataset(dataset_name::String; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    dataset = prepare_dataset(dataset_name)
    print_dataset_header(dataset, time_limit)

    for depth in depths
        println("  D = ", depth)

        # This is the exact univariate OCT model from CM1: each internal node selects one feature and one threshold.
        tree, objective, resolution_time, gap = build_tree(
            dataset.X_train,
            dataset.Y_train,
            depth,
            dataset.classes;
            multivariate = false,
            time_limit = time_limit,
        )
        print_exact_result("    Univariate...\t", tree, resolution_time, gap, dataset)

        # This variant keeps the same tree structure but allows a linear split at each internal node.
        tree, objective, resolution_time, gap = build_tree(
            dataset.X_train,
            dataset.Y_train,
            depth,
            dataset.classes;
            multivariate = true,
            time_limit = time_limit,
        )
        print_exact_result("    Multivariate...\t", tree, resolution_time, gap, dataset)
        println()
    end
end

function main(; dataset_names = DEFAULT_DATASETS, time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    for dataset_name in dataset_names
        run_exact_dataset(dataset_name; time_limit = time_limit, depths = depths)
    end
end
