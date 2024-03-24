class Learner(object):
    """ class that stores learner performance 
    """

    def __init__(self) -> None:

        self.__init_parameters()

    def __init_parameters(self,):
        self.score = 0
        self.satisfied = False

    def reset(self):

        self.__init_parameters()
