parameters = {
    "C": {
        "L": 24,
        "T": 96,
        "beta": '3.685',
        "kappa": '0.1394305',
        "kappa_add": "l",
        "csw": '2.095108',
        "a": 0.12*5.068,
        # "val_cfgs": range(8, 824+1, 16),
        "val_cfgs": range(100, 1000+1, 100),
        "NPR_cfgs": range(100, 1000+1, 100)
    },
    "M": {
        "L": 32,
        "T": 96,
        "beta": '3.80',
        "kappa": '0.138963',
        "kappa_add": "ls",
        "csw": '1.95524',
        "a": 0.094*5.068,
        # "val_cfgs": range(8, 824+1, 16),
        "val_cfgs": range(100, 600+1, 50),
        "NPR_cfgs": range(100, 600+1, 50)
    },
    "F": {
        "L": 48,
        "T": 96,
        "beta": '4.00',
        "kappa": '0.138272',
        "kappa_add": "l",
        "csw": '1.783303',
        "a": 0.064*5.068,
        # "val_cfgs": range(8, 584+1, 16),
        "val_cfgs": range(250, 550+1, 50),
        "NPR_cfgs": range(50, 550+1, 50)
    }
}
