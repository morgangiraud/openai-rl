def CartPole0phi1(obs, done=False):
    if done:
        return 2**4

    phi =  [
        1 if obs[0] < 0 else 0,
        1 if obs[1] < 0 else 0,
        1 if obs[2] < 0 else 0,
        1 if obs[3] < 0 else 0,
    ]
    return 2**0 * phi[0] + 2**1 * phi[1] + 2**2 * phi[2] + 2**3 * phi[3]

def CartPole0phi2(obs, done=False):
    if done:
        return 2**8

    phi =  [
        1 if obs[0] < 0 else 0,
        1 if abs(obs[0]) > 1.8 else 0, # 3/4th of the field, danger zone
        1 if obs[1] < 0 else 0,
        1 if abs(obs[1]) > 0.4 else 0,
        1 if obs[2] < 0 else 0,
        1 if abs(obs[2]) > 31 else 0, # 3/4th of the rotation, danger zone
        1 if obs[3] < 0 else 0,
        1 if abs(obs[3]) > 0.4 else 0,
    ]

    return (
        2**0 * phi[0] + 2**1 * phi[1] + 2**2 * phi[2] + 2**3 * phi[3]
        + 2**4 * phi[4] + 2**5 * phi[5] + 2**6 * phi[6] + 2**7 * phi[7]
    )

def MountainCar0phi(obs, done=False):
    if done:
        return 19*15

    phi =  [
        12 + round(obs[0] * 10) # [-1.2, 0.6] -> 19 states
        , 7 + round(obs[1] * 100) # [-0.07, 0.07] -> 15 states
    ]

    return (
        19**0 * phi[0] + 19**1*15**0 * phi[1]
    )

def getPhiConfig(env_name, debug=False):
    if env_name == 'CartPole-v0' or env_name == 'CartPole-v1':
        if debug:
            return {
                'nb_state': 2**4 + 1,
                'phi': CartPole0phi1
            }
        else:
            return {
                'nb_state': 2**8 + 1,
                'phi': CartPole0phi2
            }
    elif env_name == 'MountainCar-v0':
        return {
            'nb_state': 19*15 + 1,
            'phi': MountainCar0phi
        }
    else:
        raise Exception('This env (%s) has not yet a feature mapping function' % env_name)
