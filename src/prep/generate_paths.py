import argparse

################################################################################
#                       SECTION DES FONCTIONS D'ECRITURE                       #
################################################################################

def write_mnist(file, args):
    for i in range(60000):
        file.write(f'{args.path}mnist/image_{i}.png\n')


def write_custom(file, args):
    for i in range(5):
        for j in range(10):
            for n in range(124):
                file.write(f'{args.path}custom/raw-{i}-digit.{j}.{n}.png\n')


################################################################################
#                    SECTION DES FONCTIONS SUR LES ARGUMENTS                   #
################################################################################

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='',
        help='Prefix path to add to filenames')

    parser.add_argument('--output', required=True, type=str,
        help='Output file for paths.')

    parser.add_argument('--mode',
        choices=['custom', 'mnist'],
        default='custom')

    return parser.parse_args()


################################################################################
#                                SECTION MAIN                                  #
################################################################################

def main():

    args = parse_args()
    with open(args.output, 'w') as file:
        if(args.mode == 'custom'):
            write_custom(file, args)
        elif(args.mode == 'mnist'):
            write_mnist(file, args)


if __name__ == "__main__":
    main()
