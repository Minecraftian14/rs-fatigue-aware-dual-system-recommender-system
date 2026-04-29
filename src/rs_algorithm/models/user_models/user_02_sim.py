import random
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from miscellaneous import clip, lerp

from .blueprint_user_model import UserModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cosine_based_relavance_calculation(pref, pref_bias, item, item_bias):
    relevance = cosine_similarity(pref, item).squeeze() / 2.0 + 0.5
    if pref_bias is not None: relevance += pref_bias
    if item_bias is not None: relevance += item_bias
    return relevance


def dot_product_based_relavance_calculation(pref, pref_bias, item, item_bias):
    relevance = (pref * item).sum()
    if pref_bias is not None: relevance = relevance + pref_bias
    if item_bias is not None: relevance = relevance + item_bias
    relevance = sigmoid(relevance).squeeze()
    # relevance = sigmoid(relevance / 0.62 * 60)
    return relevance


def norm_dot_product_based_relavance_calculation(pref, pref_bias, item, item_bias):
    pref = pref / np.linalg.norm(pref, axis=1, keepdims=True)
    item = item / np.linalg.norm(item, axis=1, keepdims=True)
    return dot_product_based_relavance_calculation(pref, pref_bias, item, item_bias)


def identity_preference_update_manager(preference):
    return preference


def norm_preference_update_manager(preference, target_std=1, target_mean=0):
    preference = (preference - preference.mean()) / preference.std()
    return preference * target_std + target_mean


def linearly_probabilistic_click_calculator(relevance):
    return random.random() < relevance


def power_scaled_probabilistic_click_calculator(relevance, power=0.5):
    return random.random() < relevance ** power


def linear_click_calculator(relevance):
    return relevance > 0.5


class DynamicPreferenceUserModel(UserModel):
    """

    * Ability to experience change in `preference`
        * There are several ways to model change in preference, for instance
            * After some time duration, the user randomly picks up a new interest
              During the gap, the actual preference gradually interpolates between current and the new target
            * Another method is to make it a function of current recommendations and interpolate slowly
        * This particular user models the latter approach, might add the former in future.

    """

    def __init__(self,
                 preference,
                 preference_bias=None,
                 preference_volatility=0.001854,

                 relevance_calculator=cosine_based_relavance_calculation,
                 preference_update_manager=identity_preference_update_manager,
                 click_calculator=linearly_probabilistic_click_calculator,
                 ):
        self._state_vector = {'preference': preference, 'preference_bias': preference_bias, 'preference_volatility': preference_volatility}
        self.preference = preference
        self.preference_bias = preference_bias
        self.preference_volatility = preference_volatility

        self.relevance_calculator = relevance_calculator
        self.preference_update_manager = preference_update_manager
        self.click_calculator = click_calculator

        self._relevance = None

    def reset(self, seed=None):
        for k, v in self._state_vector.items():
            setattr(self, k, v)

    def react(self, time_step: datetime, device, item, item_bias):
        self._relevance = self._get_relevance(time_step, device, item, item_bias)
        did_i_click = self.click_calculator(self._relevance)
        if did_i_click: self._update_preference(time_step, device, item, item_bias)
        return did_i_click

    def _get_relevance(self, time_step: datetime, device, item, item_bias=None):
        self._relevance = self.relevance_calculator(self.preference, self.preference_bias, item, item_bias)
        return self._relevance

    def _update_preference(self, time_step: datetime, device, item, item_bias):
        # TODO: Delta Time normalize the factor
        #  factor = self.preference_volatility / (Expected Delta) * (Actual Delta)
        self.preference = self.preference_update_manager(lerp(self.preference, item, self.preference_volatility))

    def get_internal_representation(self): return {'preference': self.preference, 'preference_bias': self.preference_bias}

    def can_interact_today(self, today: datetime): return True

    def is_gone_for_good(self): return False


class SimUserModel(DynamicPreferenceUserModel):
    """

    * Ability to experience change in `preference`
        Check out `DynamicPreferenceUserModel`
    * Maintains a `fatigue` level
        * Ranges between 0 and 1
        * If a good item is shown, fatigue reduces by a factor of `good_item_fatigue_weight`
        * If a bad item is shown, fatigue increases by a factor of `bad_item_fatigue_weight`
        * If fatigue reaches 1, no more interactions for the remaining of the day
          Next day, fatigue is reduced to a factor of `fatigue_overload_cooldown_factor`
    * Maintains a `discontent` level
        * Ranges between 0 and 1
        * If a good item is shown, discontent reduces by a factor of `good_item_discontent_weight`
        * If a bad item is shown, discontent increases by a factor of `bad_item_discontent_weight`
        * If discontent reaches 1, the episode ends in failure

    """

    def __init__(self,

                 preference,
                 preference_bias=None,
                 preference_volatility=0.001854,

                 fatigue=0.0,
                 good_item_fatigue_weight=0.11,
                 bad_item_fatigue_weight=0.1,
                 is_fatigued=False,
                 fatigue_overload_cooldown_factor=0.7,
                 fatigue_timeout_days=1,

                 discontent=0.0,
                 good_item_discontent_weight=0.04,
                 bad_item_discontent_weight=0.05,
                 is_discontent=False,

                 last_interaction_day=0,

                 relevance_calculator=cosine_based_relavance_calculation,
                 preference_update_manager=identity_preference_update_manager,
                 click_calculator=linearly_probabilistic_click_calculator,

                 ):
        super().__init__(preference, preference_bias, preference_volatility, relevance_calculator, preference_update_manager, click_calculator)
        self._state_vector.update({'fatigue': fatigue, 'good_item_fatigue_weight': good_item_fatigue_weight, 'bad_item_fatigue_weight': bad_item_fatigue_weight, 'is_fatigued': is_fatigued,
                                   'fatigue_overload_cooldown_factor': fatigue_overload_cooldown_factor, 'fatigue_timeout_days': fatigue_timeout_days, 'discontent': discontent, 'good_item_discontent_weight': good_item_discontent_weight,
                                   'bad_item_discontent_weight': bad_item_discontent_weight, 'is_discontent': is_discontent, 'last_interaction_day': last_interaction_day})
        self.fatigue = fatigue
        self.good_item_fatigue_weight = good_item_fatigue_weight
        self.bad_item_fatigue_weight = bad_item_fatigue_weight
        self.is_fatigued = is_fatigued
        self.fatigue_overload_cooldown_factor = fatigue_overload_cooldown_factor
        self.fatigue_timeout_days = fatigue_timeout_days
        self.discontent = discontent
        self.good_item_discontent_weight = good_item_discontent_weight
        self.bad_item_discontent_weight = bad_item_discontent_weight
        self.is_discontent = is_discontent
        self.last_interaction_day = last_interaction_day

    def reset(self, seed=None):
        for k, v in self._state_vector.items():
            setattr(self, k, v)

    def react(self, time_step: datetime, device, item, item_bias):
        #
        if self.is_discontent:
            raise Exception("User is discontent with the platform, and refuses to interact further.")

        if self.is_fatigued:
            if time_step.day >= self.last_interaction_day + self.fatigue_timeout_days:
                self.fatigue *= self.fatigue_overload_cooldown_factor
                self.is_fatigued = False
            else:
                raise Exception("User is fatigued with the platform, and refuses to interact today.")

        relevance = self._get_relevance(time_step, device, item)

        probability = lerp(relevance, 0, self.fatigue)
        did_i_click = self.click_calculator(probability)

        if did_i_click: self._update_preference(time_step, device, item, item_bias)

        if relevance > 0.5:
            self.fatigue += self.good_item_fatigue_weight
            self.discontent -= self.good_item_discontent_weight
        else:
            self.fatigue -= self.bad_item_fatigue_weight
            self.discontent += self.bad_item_discontent_weight

        if self.fatigue >= 1: self.is_fatigued = True
        if self.discontent >= 1: self.is_discontent = True

        self.fatigue, self.discontent = clip(self.fatigue), clip(self.discontent)
        self.last_interaction_day = time_step.day

        return did_i_click

    def get_internal_representation(self):
        drepr = super().get_internal_representation()
        drepr.update({'fatigue': self.fatigue, 'is_fatigued': self.is_fatigued, 'discontent': self.discontent, 'is_discontent': self.is_discontent})
        return drepr

    def can_interact_today(self, today: datetime):
        if self.is_discontent: return False
        if self.is_fatigued:
            return today.day >= self.last_interaction_day + self.fatigue_timeout_days
        return True

    def is_gone_for_good(self):
        return self.is_discontent
