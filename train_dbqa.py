import os

from dbqa.model import DBQA

home = os.path.expanduser("~")
outdir_prefix = home + "/work/dbqa/"
data_dir = home + f"/Development/large-data/"

options = {
    # "train": data_dir + "nlpcc-iccpol-2016.dbqa.training-data-segged",
    "train": data_dir + "nlpcc-iccpol-2016.dbqa.training-data",
    # "train": data_dir + "dbqa-small",
    # "dev": data_dir + "dbqa-small",
    "dev": data_dir + "nlpcc-iccpol-2016.dbqa.testing-data",
    "data-format": "character-based",
    "merger": "cnn",
    "use-bigram": 0,
    # "external-embedding": data_dir + "sample.win7.vector"
}

scheduler = DBQA.get_training_scheduler()
scheduler.add_options("test-segged-cnn-charbased-nobigram", options, outdir_prefix)
scheduler.run_parallel()
