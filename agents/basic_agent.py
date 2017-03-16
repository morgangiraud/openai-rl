import os, time

dir = os.path.dirname(os.path.realpath(__file__)) + '/..'

class BasicAgent(object):
    def __init__(self, observation_space, action_space, config):
        pass

    def initModel(self):
        self.result_folder = dir + '/results/' + str(int(time.time()))
        pass

    def act(self, obs, eps=None):
        pass

    def learnFromEpisode(self, env, episode_id):
        pass


