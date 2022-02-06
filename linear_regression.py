import logging

logging.basicConfig(filename="linear_reg.log",
                            filemode='a',
                            format='%(asctime)s  %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logging.info("this is a test info")