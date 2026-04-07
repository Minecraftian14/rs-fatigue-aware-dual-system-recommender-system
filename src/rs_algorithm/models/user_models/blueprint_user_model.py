from abc import ABC, abstractmethod


class UserModel(ABC):
    def react(self, time_step, device, item):
        pass

    def reset(self, seed=None):
        pass

    @abstractmethod
    def get_internal_representation(self):
        """
        Only for system study/debugging!
        These values can expose the user's private details like its current preference.
        """

    @abstractmethod
    def can_interact_today(self): """Is user not coming today to use our platform?"""

    @abstractmethod
    def is_gone_for_good(self): """Is user never going to come back?"""
