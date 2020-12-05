# Author: Aqeel Anwar(ICSRL)
# Created: 9/20/2019, 12:43 PM
# Email: aqeel.anwar@gatech.edu

from configparser import ConfigParser

from dotmap import DotMap

PARSER_CACHE = None


def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        return input_string


def read_cfg(config_filename, mask_type="surgical", verbose=False):
    global PARSER_CACHE

    if PARSER_CACHE is None:
        PARSER_CACHE = ConfigParser()
        PARSER_CACHE.optionxform = str
        PARSER_CACHE.read(str(config_filename))

    cfg = DotMap()
    section_name = mask_type

    if verbose:
        hyphens = "-" * int((80 - len(str(config_filename))) / 2)
        print(hyphens + " " + str(config_filename) + " " + hyphens)

    # for section_name in parser.sections():

    if verbose:
        print("[" + section_name + "]")
    for name, value in PARSER_CACHE.items(section_name):
        value = ConvertIfStringIsInt(value)
        if name != "template":
            cfg[name] = tuple(int(s) for s in value.split(","))
        else:
            cfg[name] = value
        spaces = " " * (30 - len(name))
        if verbose:
            print(name + ":" + spaces + str(cfg[name]))

    return cfg
