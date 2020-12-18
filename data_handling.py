from util import *
from scipy.spatial.distance import pdist


def toDistance(R):
    """
    This function takes a numpy array containing positions and returns it as distances.

    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)

    Returns:
        -y:
            numpy array containing distances for every atom in every sample
            Dimensions: (n_samples,n_atoms*(n_atoms-1)/2)
    """

    shape = R.shape
    try:
        dim = shape[2]
    except:
        return
    if shape[1] < 2:
        return

    y = []

    for i in range(len(R)):  ##goes through samples
        y.append(pdist(R[i]))

    y = np.array(y)
    return y


def to_distance_minkowski_p(R, n):

    shape = R.shape
    try:
        dim = shape[2]
    except:
        return
    if shape[1] < 2:
        return

    y = []

    for i in range(len(R)):  ##goes through samples
        y.append(pdist(R[i], metric="minkowski", p=n))

    y = np.array(y)
    return y


def r_to_minkowski_m0p5(self, dataset):
    R = dataset["R"]
    return to_distance_minkowski_p(R, -0.5)


def r_to_inv_dist(self, dataset):
    R = dataset["R"]
    return 1.0 / toDistance(R)


def r_to_dist(self, dataset):
    R = dataset["R"]
    return toDistance(R)


def f_to_dist(self, dataset):
    F = dataset["F"]
    return toDistance(F)


def extract_E(self, dataset):
    E = dataset["E"]
    return np.array(E).reshape(-1, 1)


def extract_E_neg(self, dataset):
    E = dataset["E"]
    return -np.array(E).reshape(-1, 1)


def extract_R_concat(self, dataset):

    R = dataset["R"]
    n_samples, n_atoms, n_dim = R.shape
    R = np.reshape(R, (n_samples, n_atoms * n_dim))
    return np.array(R)


def extract_R(self, dataset):

    R = dataset["R"]
    n_samples, n_atoms, n_dim = R.shape
    # R=np.reshape(R,(n_samples,n_atoms*n_dim))
    return np.array(R)


def schnet_r_to_dist(self, dataset):

    if self.call_para("load_dataset", "schnet_preprocessed_npz"):

        path = self.dataset_path.replace(".db", ".npz")
        if os.path.exists(path):

            R = np.load(self.dataset_path.replace(".db", ".npz"))["R"]
            return toDistance(R)

    d = self.dataset
    N = len(d)

    R = []
    for i in range(N):
        print_x_out_of_y(f"Extracting positions", i, N)
        R.append(d[i]["_positions"].numpy())

    print_x_out_of_y(f"Extracting positions", N, N, True)
    return toDistance(np.array(R))


def schnet_extract_R_concat(self, dataset):
    d = self.dataset
    N = len(d)

    if self.call_para("load_dataset", "schnet_preprocessed_npz"):

        path = self.dataset_path.replace(".db", ".npz")
        if os.path.exists(path):

            R = np.load(self.dataset_path.replace(".db", ".npz"))["R"]
            return R.reshape((N, -1))

    R = []
    for i in range(N):
        print_x_out_of_y(f"Extracting positions", i, N)
        R.append(d[i]["_positions"].numpy())

    print_x_out_of_y(f"Extracting positions", N, N, True)
    return np.array(R).reshape((N, -1))


def schnet_extract_F_concat(self, dataset):
    d = self.dataset
    N = len(d)

    if self.call_para("load_dataset", "schnet_preprocessed_npz"):

        path = self.dataset_path.replace(".db", ".npz")
        if os.path.exists(path):

            F = np.load(self.dataset_path.replace(".db", ".npz"))["F"]
            return F.reshape((N, -1))

    F = []
    for i in range(N):
        print_x_out_of_y(f"Extracting forces", i, N)
        F.append(d[i]["forces"].numpy())
    print_x_out_of_y(f"Extracting forces", N, N, True)

    return np.array(F).reshape((N, -1))


def schnet_extract_E(self, dataset):

    if self.call_para("load_dataset", "schnet_preprocessed_npz"):

        path = self.dataset_path.replace(".db", ".npz")
        if os.path.exists(path):

            E = np.load(self.dataset_path.replace(".db", ".npz"))["E"]
            return E

    d = self.dataset
    N = len(d)

    E = []
    for i in range(N):
        print_x_out_of_y(f"Extracting energies", i, N)
        E.append(d[i]["energy"].numpy())

    print_x_out_of_y(f"Extracting forces", N, N, True)
    return np.array(E)


def schnet_indices(self, dataset):
    return np.arange(len(dataset))


def npz_indices(self, dataset):
    return np.arange(len(dataset["R"]))


def varfunc_dummy(self, dataset):

    return np.zeros(len(dataset["R"]))


def extract_F_concat(self, dataset):

    F = dataset["F"]
    n_samples, n_atoms, n_dim = F.shape
    F = np.reshape(F, (n_samples, n_atoms * n_dim))
    return np.array(F)


def mean_squared_error_sample_wise(x, y):
    err = (np.array(x) - np.array(y)) ** 2
    return err.mean(axis=1)


def indices_to_xyz_gap(self, indices=None):
    data = self.dataset

    if not type(data) == np.lib.npyio.NpzFile:
        print_error(
            f"Tried converting dataset to xyz gap, but only npz datasets currently supported, not {type(self.dataset)}"
        )

    kcalmol_to_eV = 4.3363e-2
    f_convert = 1.88972

    if indices is None:
        indices = np.arange(len(data["R"]))

    R = data["R"][indices]
    E = data["E"][indices].reshape(-1) * kcalmol_to_eV
    F = data["F"][indices] * f_convert

    z = [z_to_z_str_dict[x] for x in data["z"]]
    n_atoms = len(z)

    file_name = os.path.join(self.temp_dir, f"temp_xyz_gap_{len(indices)}")

    temp_file = open(file_name, "w+")
    header = 'Lattice="200.0 0.0 0.0 0.0 200.0 0.0 0.0 0.0 200.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=__E__ pbc="T T T"\n'

    for i in range(len(R)):
        temp_file.write(f"{n_atoms}\n")
        temp_file.write(header.replace("__E__", f"{E[i]:.6f}"))

        r, f = R[i], F[i]
        for j in range(n_atoms):
            temp_file.write(
                f"{z[j]:<20}{r[j][0]:<20.6f}{r[j][1]:<20.6f}{r[j][2]:<20.6f}{f[j][0]:<20.6f}{f[j][1]:<20.6f}{f[j][2]:<20.6f}"
            )

    temp_file.close()

    return file_name
