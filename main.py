from Resources.Constants import Constants
from Generation import Generator
from Training import Train

train = Train()
generator = Generator()
print(Constants.GREETING)

match input():
    case '1':
        train.train()
    case '2':
        generator.generate()
    case _:
        raise ValueError(Constants.ERROR)
