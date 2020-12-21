from util import *
import numpy as np
from sgdml.predict import GDMLPredict


def load_sgdml_model(self, path):
    a = np.load(path)
    training_indices = a["idxs_train"]
    m = GDMLPredict(a)
    return m, training_indices


def load_npz_prepredicted(self, path):
    m = np.load(path)
    training_indices = []
    return m, training_indices


def sgdml_all_default(train_indices, args):
    from sgdml.cli import create, train, validate, select, test
    from sgdml.utils import ui, io

    ui.print_step_title("STEP 1", "Cross-validation task creation")
    task_dir = create(**args)
    dataset = args["dataset"][1]

    if (train_indices is not None) and not (type(train_indices) == int):
        #  CHANGE TRAINING INDICES
        #  AND RELATED ARRAYS
        R_train = dataset["R"][train_indices]
        F_train = dataset["F"][train_indices]
        E_train = dataset["E"][train_indices]

        for file in os.listdir(task_dir):
            if file.endswith(".npz"):
                name = os.path.join(task_dir, file)
                a = dict(np.load(name, allow_pickle=True))
                a["R_train"] = R_train
                a["F_train"] = F_train
                if "E_train" in a:
                    a["E_train"] = E_train
                a["idxs_train"] = train_indices
                np.savez_compressed(name, **a)

    ui.print_step_title("STEP 2", "Training")
    task_dir_arg = io.is_dir_with_file_type(task_dir, "task")
    args["task_dir"] = task_dir_arg
    model_dir_or_file_path = train(**args)

    ui.print_step_title("STEP 3", "Validation")
    model_dir_arg = io.is_dir_with_file_type(
        model_dir_or_file_path, "model", or_file=True
    )

    valid_dataset = args["valid_dataset"]
    validate(
        model_dir_arg,
        valid_dataset,
        overwrite=False,
        max_processes=args["max_processes"],
        use_torch=args["use_torch"],
    )

    ui.print_step_title("STEP 4", "Hyper-parameter selection")
    model_file_name = select(
        model_dir_arg, args["overwrite"], args["max_processes"], args["model_file"]
    )

    ui.print_step_title("STEP 5", "Testing")
    model_dir_arg = io.is_dir_with_file_type(model_file_name, "model", or_file=True)
    test_dataset = args["test_dataset"]

    test(
        model_dir_arg,
        test_dataset,
        args["n_test"],
        overwrite=False,
        max_processes=args["max_processes"],
        use_torch=args["use_torch"],
    )

    print(
        "\n"
        + ui.color_str("  DONE  ", fore_color=ui.BLACK, back_color=ui.GREEN, bold=True)
        + " Training assistant finished sucessfully."
    )
    print("         This is your model file: '{}'".format(model_file_name))

    if "glob" in globals():
        global glob
        del glob


def npz_prepredicted_F(self, indices):
    F = self.curr_model["F"][indices]
    return F.reshape(len(F), -1)


def sgdml_predict_F(self, R):
    model = self.curr_model
    _, F = model.predict(R)
    return F


def sgdml_predict_E(self, R):
    model = self.curr_model
    E, _ = model.predict(R)
    return E


def sgdml_model_info(self):
    model = self.curr_model
    dic = model.__dict__
    print(f"{'n_atoms':<10}{dic['n_atoms']}")
    print(f"{'n_train':<10}{dic['n_train']}")


def sgdml_train_default(self, train_indices, model_path, old_model_path, sgdml_args):

    args = sgdml_args.copy()
    dataset_tuple = (self.args["dataset_file"], self.dataset)
    if type(train_indices) == int:
        n_train = train_indices
    else:
        n_train = len(train_indices)

    if old_model_path is not None:
        model0 = (old_model_path, np.load(old_model_path, allow_pickle=True))
    else:
        model0 = None

    task_dir = os.path.join(self.storage_dir, f"sgdml_task_{n_train}")
    args["n_train"] = n_train
    args["task_dir"] = os.path.join(self.storage_dir, f"sgdml_task_{n_train}")
    args["valid_dataset"] = dataset_tuple
    args["test_dataset"] = dataset_tuple
    args["dataset"] = dataset_tuple
    args["model_file"] = model_path
    args["command"] = "all"
    args["max_processes"] = self.n_cores
    args["model0"] = model0

    if self.call_para("train_models", "suppress_sgdml_prints"):
        with sgdml_print_suppressor():
            sgdml_all_default(train_indices, args)

    else:
        sgdml_all_default(train_indices, args)

    # if self.call_para('train_models','suppress_sgdml_prints'):
    # 	with sgdml_print_suppressor():
    # 		cli.all(**args)
    # else:
    # 	cli.all(**args)


def sgdml_train_data(self, dataset_tuple, model_path, sgdml_args):

    args = sgdml_args.copy()

    task_dir = os.path.join(self.storage_dir, f"sgdml_task_temp")

    args["task_dir"] = task_dir
    args["valid_dataset"] = dataset_tuple
    args["test_dataset"] = dataset_tuple
    args["dataset"] = dataset_tuple
    args["model_file"] = model_path
    args["command"] = "all"
    args["max_processes"] = self.n_cores
    args["model0"] = None

    print_ongoing_process(f"Training model ({args['n_train']} points)")

    start_time = time.time()
    if self.call_para("train_models", "suppress_sgdml_prints"):
        with sgdml_print_suppressor():
            sgdml_all_default(None, args)

    else:
        sgdml_all_default(None, args)

    print_ongoing_process(
        f"Trained model ({args['n_train']} points)", True, time=time.time() - start_time
    )


def sgdml_train_data_flex(self, dataset_tuple, model_path, sgdml_args):
    args = sgdml_args.copy()

    n_train, n_valid, n_test = args["n_train"], args["n_valid"], args["n_test"]

    tot = n_train + n_valid + n_test
    N = len(dataset_tuple[1]["R"])

    if N < n_train:
        args["n_train"], args["n_valid"], args["n_test"] = N - 2, 1, 1
    elif N < tot:
        n_train = min(n_train, N - 2)
        args["n_train"], args["n_valid"], args["n_test"] = n_train, 1, 1

    sgdml_train_data(self, dataset_tuple, model_path, args)


def sgdml_path_predict_F(self, model_path, input_var, batch_size):
    from sgdml.predict import GDMLPredict

    N = len(input_var)
    n_batches = N // batch_size + 1

    if n_batches > 999:
        width = 20
    else:
        width = None

    npz = np.load(model_path)
    model = GDMLPredict(npz)

    message = f"Predicting {os.path.basename(model_path)} batches"

    predicts = []

    start_time, eta = time.time(), 0
    for i in range(n_batches):
        print_x_out_of_y_eta(message, i, n_batches, eta, width=width)
        R = input_var[i * batch_size : (i + 1) * batch_size]
        if len(R) == 0:
            break
        _, F = model.predict(R)
        predicts.append(F)

        avg_time = (time.time() - start_time) / (i + 1)
        eta = (n_batches - i + 1) * avg_time

    print_x_out_of_y_eta(
        message, n_batches, n_batches, time.time() - start_time, True, width=width
    )

    predicts = np.concatenate(predicts)
    return predicts


def get_sgdml_training_set(self, model):
    print(model.__dict__.keys())
    # sys.exit()


### SchNet ###


def schnet_train_default(self, train_indices, model_path, old_model_path, schnet_args):

    import schnetpack as spk
    import schnetpack.train as trn
    import torch

    n_val = schnet_args.get("n_val", 100)

    #  LOADING train, val, test
    if type(train_indices) == int:
        n_train = train_indices

        # Preparing storage
        storage = os.path.join(self.temp_dir, f"schnet_{n_train}")
        if not os.path.exists(storage):
            os.mkdir(storage)
        split_path = os.path.join(storage, "split.npz")

        train, val, test = spk.train_test_split(
            data=self.dataset, num_train=n_train, num_val=n_val, split_file=split_path
        )

    else:
        n_train = len(train_indices)

        # Preparing storage
        storage = os.path.join(self.temp_dir, f"schnet_{n_train}")
        if not os.path.exists(storage):
            os.mkdir(storage)
        split_path = os.path.join(storage, "split.npz")

        all_ind = np.arange(len(self.dataset))

        #  train
        train_ind = train_indices
        all_ind = np.delete(all_ind, train_ind)

        # val
        val_ind_ind = np.random.choice(np.arange(len(all_ind)), n_val, replace=False)
        val_ind = all_ind[val_ind_ind]
        all_ind = np.delete(all_ind, val_ind_ind)

        split_dict = {
            "train_idx": train_ind,
            "val_idx": val_ind,
            "test_idx": all_ind,
        }
        np.savez_compressed(split_path, **split_dict)

        train, val, test = spk.train_test_split(
            data=self.dataset, split_file=split_path
        )

    print_ongoing_process(f"Preparing SchNet training, {len(train)} points", True)

    data = self.dataset

    batch_size = schnet_args.get("batch_size", 10)
    n_features = schnet_args.get("n_features", 64)
    n_gaussians = schnet_args.get("n_gaussians", 25)
    n_interactions = schnet_args.get("n_interactions", 6)
    cutoff = schnet_args.get("cutoff", 5.0)
    learning_rate = schnet_args.get("learning_rate", 1e-3)
    rho_tradeoff = schnet_args.get("rho_tradeoff", 0.1)
    patience = schnet_args.get("patience", 5)
    n_epochs = schnet_args.get("n_epochs", 100)

    #  PRINTING INFO
    i = {}
    i["batch_size"], i["n_features"] = batch_size, n_features
    i["n_gaussians"], i["n_interactions"] = n_gaussians, n_interactions
    i["cutoff"], i["learning_rate"] = cutoff, learning_rate
    i["rho_tradeoff"], i["patience"] = rho_tradeoff, patience
    i["n_epochs"], i["n_val"] = n_epochs, n_val
    print_table("Parameters", None, None, i, width=20)
    print()

    train_loader = spk.AtomsLoader(train, shuffle=True, batch_size=batch_size)
    val_loader = spk.AtomsLoader(val, batch_size=batch_size)

    #  STATISTICS + PRINTS
    means, stddevs = train_loader.get_statistics("energy", divide_by_atoms=True)
    print_info(
        "Mean atomization energy / atom:      {:12.4f} [kcal/mol]".format(
            means["energy"][0]
        )
    )
    print_info(
        "Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]".format(
            stddevs["energy"][0]
        )
    )

    #  LOADING MODEL
    print_ongoing_process("Loading representation and model")
    schnet = spk.representation.SchNet(
        n_atom_basis=n_features,
        n_filters=n_features,
        n_gaussians=n_gaussians,
        n_interactions=n_interactions,
        cutoff=cutoff,
        cutoff_network=spk.nn.cutoff.CosineCutoff,
    )

    energy_model = spk.atomistic.Atomwise(
        n_in=n_features,
        property="energy",
        mean=means["energy"],
        stddev=stddevs["energy"],
        derivative="forces",
        negative_dr=True,
    )

    model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)
    print_ongoing_process("Loading representation and model", True)

    #  OPTIMIZER AND LOSS
    print_ongoing_process("Defining loss function and optimizer")
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=learning_rate)

    def loss(batch, result):

        # compute the mean squared error on the energies
        diff_energy = batch["energy"] - result["energy"]
        err_sq_energy = torch.mean(diff_energy ** 2)

        # compute the mean squared error on the forces
        diff_forces = batch["forces"] - result["forces"]
        err_sq_forces = torch.mean(diff_forces ** 2)

        # build the combined loss function
        err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

        return err_sq

    print_ongoing_process("Defining loss function and optimizer", True)

    # METRICS AND HOOKS
    print_ongoing_process("Setting up metrics and hooks")
    metrics = [
        spk.metrics.MeanAbsoluteError("energy"),
        spk.metrics.MeanAbsoluteError("forces"),
    ]

    hooks = [
        trn.CSVHook(log_path=storage, metrics=metrics),
        trn.ReduceLROnPlateauHook(
            optimizer, patience=5, factor=0.8, min_lr=1e-6, stop_after_min=True
        ),
    ]
    print_ongoing_process("Setting up metrics and hooks", True)

    print_ongoing_process("Setting up trainer")

    trainer = trn.Trainer(
        model_path=storage,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    print_ongoing_process("Setting up trainer", True)

    if torch.cuda.is_available():
        device = "cuda"
        print_info(f"Cuda cores found, training on GPU")

    else:
        device = "cpu"
        print_info(f"No cuda cores found, training on CPU")

    print_ongoing_process(f"Training {n_epochs} ecpochs, out in {storage}")
    trainer.train(device=device, n_epochs=n_epochs)
    print_ongoing_process(f"Training {n_epochs} epochs, out in {storage}", True)

    os.mkdir(model_path)

    os.rename(os.path.join(storage, "best_model"), os.path.join(model_path, "model"))
    shutil.copy(split_path, os.path.join(model_path, "split.npz"))


def load_schnet_model(self, path):
    import torch

    if path.split(".")[-1] == "npz":
        a = np.load(path)
        if "train_idx" not in dict(a).keys():
            print_error(f"Given split file {path} did not contain train_idx.")

        return None, a["train_idx"]

    if not os.path.isdir(path):
        print_error(
            f"{path} is not a directory. SchNet models need to be a "
            'directory containing the model as "model" and the split file as '
            '"split.npz". Alternatively, it can be the split file on its own '
            " in the .npz format to retrain from a given training set."
        )

    split, model = os.path.join(path, "split.npz"), os.path.join(path, "model")

    if not os.path.exists(split):
        print_error(f'"split.npz" file not found in {path}')

    if not os.path.exists(model):
        print_error(f'"model" file not found in {path}')

    if torch.cuda.is_available():
        m = torch.load(model, map_location=torch.device("cuda"))
    else:
        m = torch.load(model, map_location=torch.device("cpu"))
    training_indices = np.load(split)["train_idx"]

    return m, training_indices


def schnet_predict_F(self, indices):
    m = self.curr_model

    test = self.dataset.create_subset(indices)

    ind0 = indices[0]

    import schnetpack as spk

    test_loader = spk.AtomsLoader(test, batch_size=100)
    preds = []

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    for count, batch in enumerate(test_loader):
        print(f"{count}/{len(test_loader)}", end="\r")

        batch = {k: v.to(device) for k, v in batch.items()}
        preds.append(m(batch)["forces"].detach().cpu().numpy())

    F = np.concatenate(preds)
    return F.reshape(len(F), -1)


def schnet_predict_E(self, indices):
    m = self.curr_model
    test = self.dataset.create_subset(indices)

    import schnetpack as spk

    test_loader = spk.AtomsLoader(test, batch_size=100)
    preds = []

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    for count, batch in enumerate(test_loader):
        print(f"{count}/{len(test_loader)}", end="\r")

        batch = {k: v.to(device) for k, v in batch.items()}
        preds.append(m(batch)["energy"].detach().cpu().numpy())

    return np.concatenate(preds)


### GAP/SOAP ###


def gapsoap_train_default(
    self, train_indices, model_path, old_model_path, base_command
):

    from data_handling import indices_to_xyz_gap

    N = len(self.dataset["R"])

    if type(train_indices) == int:
        train_indices = np.random.choice(np.arange(N), train_indices)

    train_file = indices_to_xyz_gap(self, train_indices)
    train_file = train_file.replace(f"{self.storage_dir}/", "")

    base_command = (
        base_command.replace("__TRAIN_FILE__", train_file)
        .replace("__END_FILE__", "model.xml")
        .replace("__WORK_DIR__", self.storage_dir)
    )

    subprocess.call(base_command, shell=True)

    os.mkdir(model_path)
    os.rename(
        os.path.join(self.storage_dir, "model.xml"),
        os.path.join(model_path, "model.xml"),
    )
    np.save(os.path.join(model_path, "training_indices.npy"), train_indices)


def load_gap_model(self, path):
    from quippy.potential import Potential

    ind = np.load(os.path.join(path, "training_indices.npy"), allow_pickle=True)

    gap_path = os.path.join(path, "model.xml")

    return Potential(param_filename=gap_path), ind


def gap_predict_F(self, indices):
    from data_handling import indices_to_xyz_gap
    from ase.io import read

    model = self.curr_model
    F = []

    print(" ")
    temp_file = indices_to_xyz_gap(self, indices)
    data = read(temp_file, format="xyz", index=":")
    message = f"Predicting {len(indices)} atoms with GAP model"

    N = len(indices)
    start_time, eta = time.time(), 0
    for i in range(N):
        data[i].set_calculator(model)
        F.append(data[i].get_forces().flatten() / 1.88972)

        if i % 100 == 0:
            avg_time = (time.time() - start_time) / (i + 1)

        eta = (N - i + 1) * avg_time

        print_x_out_of_y_eta(message, i, N, eta, True, width=width)

    print_x_out_of_y_eta(message, N, N, time.time() - start_time, True, width=width)

    return np.array(F)
