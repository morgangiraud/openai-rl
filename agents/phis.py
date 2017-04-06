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

def getPhiConfig(env_name, debug=False):
    if env_name == 'CartPole-v0':
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
    else:
        raise Exception('This env (%s) has not yet a feature mapping function' % env_name)
