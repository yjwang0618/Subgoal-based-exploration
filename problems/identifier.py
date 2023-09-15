import besd_gw
import besd_ky
import besd_pd
import besd_mc
import besd_gwItem
import besd_ll


def identify_problem(argv, bucket):
    benchmark_name = argv[0]
    prob_name = argv[-1]
    decode = benchmark_name.split("_")
    print(decode)
    print('benchmark_name = ', benchmark_name)
    print('prob_name = ', prob_name)
    if decode[0] == "besd":
        which_problem = int(argv[1])
        replication_no = int(argv[2])
        print('which_problem = ', which_problem)
        print('replication_no = ', replication_no)

        if decode[1] == "gw": problem_class = besd_gw.class_collection[benchmark_name]

        elif decode[1] == "ky": problem_class = besd_ky.class_collection[benchmark_name]

        elif decode[1] == "pd": problem_class = besd_pd.class_collection[benchmark_name]

        elif decode[1] == "mc": problem_class = besd_mc.class_collection[benchmark_name]

        elif decode[1] == "it": problem_class = besd_gwItem.class_collection[benchmark_name]

        elif decode[1] == "ll": problem_class = besd_ll.class_collection[benchmark_name]

        else: raise ValueError("func name not recognized")

        problem = problem_class(replication_no, which_problem, bucket, prob_name)

    else:
        raise ValueError("task name not recognized")
    return problem