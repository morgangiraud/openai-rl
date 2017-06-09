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
        1 if obs[0] < 0 else 0,         # left/right side of the field
        1 if abs(obs[0]) > 1.2 else 0,  # center/border of the field range
        1 if obs[1] < 0 else 0,         # Moving left/right
        1 if abs(obs[1]) > 0.4 else 0,  # Moving slow/fast
        1 if obs[2] < 0 else 0,         # positive/negative angle
        1 if abs(obs[2]) > 0.10 else 0, # center/border of the anglular range
        1 if obs[3] < 0 else 0,         # Rotating left/right
        1 if abs(obs[3]) > 0.4 else 0,  # Rotating slow/fast
    ]
    if phi[0] == 0 and phi[1] == 1:
        base = 0 # extreme left side
    elif phi[0] == 0 and phi[1] == 0:
        base = 64 # left side
    elif phi[0] == 1 and phi[1] == 0:
        base = 128 # right side
    elif phi[0] == 1 and phi[1] == 1:
        base = 192 # extreme right side

    return ( 
        base + 2**0 * phi[2] + 2**1 * phi[3] + 2**2 * phi[4] 
        + 2**3 * phi[5] + 2**4 * phi[6] + 2**5 * phi[7] 
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

def Acrobot1phi(obs, done=False):
    if done:
        return 2*10

    phi =  [
        1 if obs[0] < 0.27 else 0 # cos([-1., 1.])~[0, 0.54] -> 2 states

        , 1 if abs(obs[1]) > 0.42 else 0 # sin([-1., 1.])~[-0.84, 0.84] -> 2 states
        , 1 if obs[1] < 0 else 0 # sin([-1., 1.])~[-0.84, 0.84] -> 2 states

        , 1 if obs[2] < 0.27 else 0 # cos([-1., 1.])~[0, 0.54] -> 2 states

        , 1 if abs(obs[3]) > 0.42 else 0 # sin([-1., 1.])~[-0.84, 0.84] -> 2 states
        , 1 if obs[3] < 0 else 0 # sin([-1., 1.])~[-0.84, 0.84] -> 2 states

        , 1 if abs(obs[4]) > 2 * 3.14 else 0 # [-4pi, 4pi] -> 2 states
        , 1 if obs[4] < 0 else 0 # [-4pi, 4pi] -> 2 states

        , 1 if abs(obs[5]) > 4.5 * 3.14 else 0 # [-9pi, 9pi] -> 2 states
        , 1 if obs[5] < 0 else 0 # [-9pi, 9pi] -> 2 states
    ]

    return (
        2**0 * phi[0] + 2**1 * phi[1] + 2**2 * phi[2] + 2**3 * phi[3]
        + 2**4 * phi[4] + 2**5 * phi[5] + 2**6 * phi[6] + 2**7 * phi[7]
        + 2**4 * phi[8] + 2**5 * phi[9]
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
    elif env_name == 'Acrobot-v1':
        return {
            'nb_state': 2**10 + 1,
            'phi': Acrobot1phi
        }
    else:
        raise Exception('This env (%s) has not yet a feature mapping function' % env_name)
