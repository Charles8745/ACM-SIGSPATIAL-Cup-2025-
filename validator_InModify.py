import sys

task_specs = {
    "a": {
        "uid_range": (147001, 150000),
        "d_min": 61,
        "d_max": 75,
        "t_min": 0,
        "t_max": 47,
        "coord_min": 1,
        "coord_max": 200,
    },
    "b": {
        "uid_range": (27001, 30000),
        "d_min": 61,
        "d_max": 75,
        "t_min": 0,
        "t_max": 47,
        "coord_min": 1,
        "coord_max": 200,
    },
    "c": {
        "uid_range": (22001, 25000),
        "d_min": 61,
        "d_max": 75,
        "t_min": 0,
        "t_max": 47,
        "coord_min": 1,
        "coord_max": 200,
    },
    "d": {
        "uid_range": (17001, 20000),
        "d_min": 61,
        "d_max": 75,
        "t_min": 0,
        "t_max": 47,
        "coord_min": 1,
        "coord_max": 200,
    },
    "test": {
        "uid_range": (50, 60),
        "d_min": 60,
        "d_max": 74,
        "t_min": 0,
        "t_max": 47,
        "coord_min": 1,
        "coord_max": 200,
    },
}

def error(message):
    print(message)
    sys.exit(1)

def load_dataset(fpath, specs):
    uid_dict = dict()
    for l in open(fpath):
        if l.startswith("uid"):
            continue
        l = l.rstrip()
        uid_str, d_str, t_str, x_str, y_str = l.split(",")
        uid = int(uid_str)
        if uid < specs["uid_range"][0]:
            continue
        d = int(d_str)
        t = int(t_str)
        x = int(x_str)
        y = int(y_str)

        if uid not in uid_dict.keys():
            uid_dict[uid] = list()
        if d >= specs["d_min"]:
            uid_dict[uid].append((d, t, x, y))

    return uid_dict

def check_consistency(pred_seq, ans_seq, uid):
    # check the consistency between a trajectory in prediction and its counterpart in reference
    error_prefix = "Error occurring regarding uid {}: ".format(uid)

    # the trajectory length
    if len(pred_seq) != len(ans_seq):
        error(
            error_prefix + \
            "The length doesn't match between the generated and reference trajectories.")

    # consistency of day and time
    for idx, (pred_step, ans_step) in enumerate(zip(pred_seq, ans_seq)):
        pred_d, pred_t = pred_step[:2]
        ans_d, ans_t = ans_step[:2]
        if not (pred_d == ans_d and pred_t == ans_t):
            error(
                error_prefix + \
                "Day and time are not the same; "
                "(d, t) = ({}, {}) for generated while (d, t) = ({}, {}) for reference at step {} of the trajectory.".format(
                    pred_d, pred_t, ans_d, ans_t, idx))

def main(city_name, raw_data_path, test_data_input):
    # parse the arguments
    # if len(sys.argv) != 4:
    #     error(
    #         "Usage: \n"
    #         "    python3 validator.py task_id dataset_file_path submission_file_path\n"
    #         "        where task_id is a, b, c, or d")

    # task_id = sys.argv[1].lower()
    # dataset_fpath = sys.argv[2]
    # generated_fpath = sys.argv[3]

    task_id = city_name
    dataset_fpath = raw_data_path
    generated_fpath = test_data_input

    if task_id not in task_specs.keys():
        error("Invalid task_id: {}".format(task_id))

    # retrieve the corresponding task specifications
    specs = task_specs[task_id]

    uid_range = specs["uid_range"]
    d_min = specs["d_min"]
    d_max = specs["d_max"]
    t_min = specs["t_min"]
    t_max = specs["t_max"]
    coord_min = specs["coord_min"]
    coord_max = specs["coord_max"]

    # prepare the reference set of uid's
    uid_set_ref = set()
    uid_min, uid_max = specs["uid_range"]
    for uid in range(uid_min, uid_max + 1):
        uid_set_ref.add(uid)

    # now start the actual test...
    uid_set = set()
    pred_uid_dict = dict()
    ans_uid_dict = dict()

    print("Loading the submission file...")
    for i, l in enumerate(open(generated_fpath)):
        if i == 0 and l.startswith("uid,"):
            # skip the header line
            continue
        cols = l.rstrip().split(",")

        error_prefix = "Error at line index {}: ".format(i)

        # the number of columns
        if len(cols) != 5:
            error(
                error_prefix + \
                "The number of columns must be 5")

        # each column must be numeric
        for c in cols:
            if not c.isnumeric():
                error(
                    error_prefix + \
                    "Each column must be numeric")

        # convert the columns
        uid_str, d_str, t_str, x_str, y_str = cols
        uid = int(uid_str)
        d = int(d_str)
        t = int(t_str)
        x = int(x_str)
        y = int(y_str)

        # remember the uid
        uid_set.add(uid)

        # range check
        if d < d_min or d > d_max:
            error(
                error_prefix + \
                "d={} is out of range (It must be within the prediction target period, which is from {} to {}.)".format(d, d_min, d_max))
        if t < t_min or t > t_max:
            error(
                error_prefix + \
                "t={} is out of range".format(t))
        if x < coord_min or x > coord_max:
            error(
                error_prefix + \
                "x={} is out of range".format(x))
        if y < coord_min or y > coord_max:
            error(
                error_prefix + \
                "y={} is out of range".format(y))

        if uid not in pred_uid_dict.keys():
            pred_uid_dict[uid] = list()
        pred_uid_dict[uid].append((d, t, x, y))
    print("")

    # uid check
    print("Checking the set of uid's...")
    print("# of uid's: {}".format(len(uid_set)))
    if uid_set != uid_set_ref:
        error(
            "The set of uid's doesn't match that of reference; "
            "there seem to be extra or lacking uid's")
    print("")

    # comparison between the submission file and the dataset
    print("Now loading the dataset file and comparing the submission data to it...")
    ans_uid_dict = load_dataset(dataset_fpath, specs)
    for uid in range(specs["uid_range"][0], specs["uid_range"][1] + 1):
        pred_seq = pred_uid_dict[uid]
        ans_seq = ans_uid_dict[uid]
        check_consistency(pred_seq, ans_seq, uid)
    print("")


    print("Validation finished without errors!")

if __name__ == "__main__":
    main()
