include(joinpath(@__DIR__, "main_merge.jl"))
include(joinpath(@__DIR__, "shift.jl"))

function testIterative(
    X_train,
    Y_train,
    X_test,
    Y_test,
    depth,
    classes;
    time_limit::Int = -1,
    isExact::Bool = false,
    shiftSeparations::Bool = false,
    gammas = ITERATIVE_GAMMAS,
)
    println("      Gamma\tClusters\tGap\tTrain/Test\tTime\tIterations")

    for gamma in gammas
        clusters = simpleMerge(X_train, Y_train, gamma)
        # The iterative algorithm starts from clustered data and reopens only the clusters cut by the current tree.
        tree, objective, resolution_time, gap, iteration_count = iteratively_build_tree(
            clusters,
            depth,
            X_train,
            Y_train,
            classes;
            multivariate = false,
            time_limit = time_limit,
            isExact = isExact,
            shiftSeparations = shiftSeparations,
        )

        gap_text = gap == -1 ? "n/a" : string(round(gap, digits = 1), "%")
        print("      ", round(100 * gamma, digits = 0), "%\t", length(clusters), "\t\t", gap_text, "\t")

        if tree === nothing
            println("-/-\t\t", round(resolution_time, digits = 1), "s\t", iteration_count)
            continue
        end

        train_errors = prediction_errors(tree, X_train, Y_train, classes)
        test_errors = prediction_errors(tree, X_test, Y_test, classes)
        println(train_errors, "/", test_errors, "\t\t", round(resolution_time, digits = 1), "s\t", iteration_count)
    end

    println()
end

function run_iterative_dataset(dataset_name::String; time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    dataset = prepare_dataset(dataset_name)
    print_dataset_header(dataset, time_limit)

    for depth in depths
        println("  D = ", depth)

        # FU solves the clustered formulation once on the merged data.
        println("    FU")
        testMerge(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isMultivariate = false,
            gammas = ITERATIVE_GAMMAS,
        )

        # FhS uses cluster barycenters during the iterations, as in the heuristic studied in the course project.
        println("    FhS")
        testIterative(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isExact = false,
            shiftSeparations = false,
        )

        # This post-processing step only shifts thresholds after optimization; it does not change the selected tree topology.
        println("    FhS with shifts")
        testIterative(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isExact = false,
            shiftSeparations = true,
        )

        # FeS replaces barycenters by exact samples inside each cluster, which is more faithful but also more expensive.
        println("    FeS")
        testIterative(
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            dataset.Y_test,
            depth,
            dataset.classes;
            time_limit = time_limit,
            isExact = true,
            shiftSeparations = false,
        )
    end
end

function main_iterative(; dataset_names = DEFAULT_DATASETS, time_limit::Int = 180, depths = DEFAULT_DEPTHS)
    for dataset_name in dataset_names
        run_iterative_dataset(dataset_name; time_limit = time_limit, depths = depths)
    end
end
