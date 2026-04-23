from model import PKNODE
import utils


def main():
    config = utils.getConfig()
    print(config)
    dim_c = config["settings"]["nn"]["dim_c"]
    dim_V = config["settings"]["nn"]["dim_V"]

    print(dim_c)
    print(dim_V)

    model = PKNODE(dim_c, dim_V, 2)
    print(model)


if __name__ == "__main__":
    main()
