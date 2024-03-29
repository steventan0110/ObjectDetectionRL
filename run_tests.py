from tests import test_split_data as tsd
from tests import test_transforms as tt
from tests import test_dataset as td
from tests import test_models as tm


def main():
    tsd.run_all()
    tt.run_all()
    td.run_all()
    tm.run_all()


if __name__ == '__main__':
    main()