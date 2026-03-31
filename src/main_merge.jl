include(joinpath(@__DIR__, "building_tree.jl"))
include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "merge.jl"))

function testMerge(
    X_train,
    Y_train,
    X_test,
    Y_test,
    depth,
    classes;
    time_limit::Int = -1,
    isMultivariate::Bool = false,
    gammas = MERGE_GAMMAS,
)
    println("      Gamma\tClusters\tGap\tTrain/Test\tTime")

    for gamma in gammas
        # Gamma controls how aggressively same-class samples are merged before solving the clustered tree model.
        clusters = simpleMerge(X_train, Y_train, gamma)
        tree, objective, resolution_time, gap = build_tree(
            clusters,
            depth,
            classes;
            multivariate = isMultivariate,
            time_limit = time_limit,
        )

        print("      ", round(100 * gamma, digits = 0), "%\t", length(clusters), "\t\t")
        print(round(gap, digits = 1), "%\t")

        if tree === nothing
            println("-/-\t\t", round(resolution_time, digits = 1), "s")
            continue
        end

        train_errors = prediction_errors(tree, X_train, Y_train, classes)
        test_errors = prediction_errors(tree, X_test, Y_test, classes)
        println(train_errors, "/", test_errors, "\t\t", round(resolution_time, digits = 1), "s")
    end

    println()
end

function run_merge_dataset(dataset_name::String; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    dataset = prepare_dataset(dataset_name)
    print_dataset_header(dataset, time_limit)

    for depth in depths
        println("  D = ", depth)
        println("    Univariate")
        testMerge(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isMultivariate = false,
        )

        println("    Multivariate")
        testMerge(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isMultivariate = true,
        )
    end
end

function main_merge(; dataset_names = DEFAULT_DATASETS, time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    for dataset_name in dataset_names
        run_merge_dataset(dataset_name; time_limit = time_limit, depths = depths)
    end
end
