import argparse as ap
import json
import os
import random
import sys

import msgpack
from PIL import Image

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


def RandomTestParams(tid):
    seed = tid * 19997 + 997
    random.seed(seed)
    width = 50 + random.randint(0, 100)
    height = 50 + random.randint(0, 100)
    nr_cities = 4 + random.randint(0, (width + height) // 10)
    nr_trains = min(nr_cities * 20, 100 + random.randint(0, 100))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, 5)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)
    return (
        seed, width, height,
        nr_trains, nr_cities,
        max_rails_between_cities, max_rails_in_cities,
        malfunction_rate, malfunction_min_duration, malfunction_max_duration
    )


def RandomTestParams_small(tid):
    seed = tid * 19997 + 997
    random.seed(seed)

    nSize = random.randint(0, 5)

    width = 20 + nSize * 5
    height = 20 + nSize * 5
    nr_cities = 2 + nSize // 2 + random.randint(0, 2)
    nr_trains = min(nr_cities * 5, 5 + random.randint(0, 5))  # , 10 + random.randint(0, 10))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, nSize)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)
    return (
        seed, width, height,
        nr_trains, nr_cities,
        max_rails_between_cities, max_rails_in_cities,
        malfunction_rate, malfunction_min_duration, malfunction_max_duration
    )


def ShouldRunTest(tid):
    return tid >= 7
    # return tid >= 3
    return True


def create_test_env(fnParams, nTest, sDir):
    (seed, width, height,
     nr_trains, nr_cities,
     max_rails_between_cities, max_rails_in_cities,
     malfunction_rate, malfunction_min_duration, malfunction_max_duration) = fnParams(nTest)
    # if not ShouldRunTest(test_id):
    #    continue

    rail_generator = sparse_rail_generator(
        max_num_cities=nr_cities,
        seed=seed,
        grid_mode=False,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_cities,
    )

    # stochastic_data = {'malfunction_rate': malfunction_rate,
    #                    'min_duration': malfunction_min_duration,
    #                    'max_duration': malfunction_max_duration
    #                }

    stochastic_data = MalfunctionParameters(malfunction_rate=malfunction_rate,
                                            min_duration=malfunction_min_duration,
                                            max_duration=malfunction_max_duration
                                            )

    observation_builder = GlobalObsForRailEnv()

    DEFAULT_SPEED_RATIO_MAP = {
        1.: 0.25,
        1. / 2.: 0.25,
        1. / 3.: 0.25,
        1. / 4.: 0.25}

    schedule_generator = sparse_schedule_generator(DEFAULT_SPEED_RATIO_MAP)

    for iAttempt in range(5):
        try:
            env = RailEnv(
                width=width,
                height=height,
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=nr_trains,
                malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                obs_builder_object=observation_builder,
                remove_agents_at_target=True
            )
            obs = env.reset(random_seed=seed)
            break
        except ValueError as oErr:
            print("Error:", oErr)
            width += 5
            height += 5
            print("Try again with larger env: (w,h):", width, height)

    if not os.path.exists(sDir):
        os.makedirs(sDir)

    sfName = "{}/Level_{}.mpk".format(sDir, nTest)
    if os.path.exists(sfName):
        os.remove(sfName)
    env.save(sfName)

    sys.stdout.write(".")
    sys.stdout.flush()

    return env


# env = create_test_env(RandomTestParams_small, 0, "train-env-small/Test_0")

def createEnvSet(nStart, nEnd, sDir, bSmall=True):
    # print("Generate small env in train-env-small:")
    print(f"Generate env (small={bSmall}) in dir {sDir}:")

    sDirImages = "train-env-small/images/"
    if not os.path.exists(sDirImages):
        os.makedirs(sDirImages)

    for test_id in range(nStart, nEnd, 1):
        env = create_test_env(RandomTestParams_small, test_id, sDir)

        oRender = RenderTool(env, gl="PILSVG")

        # oRender.env = env
        # oRender.set_new_rail()
        oRender.render_env()
        g2img = oRender.get_image()
        imgPIL = Image.fromarray(g2img)
        # imgPIL.show()

        imgPIL.save(sDirImages + "Level_{}.png".format(test_id))

    # print("Generate large env in train-env-1000:")

    # for test_id in range(100):
    #     create_test_env(RandomTestParams, test_id, "train-env-1000/Test_0")


def merge(sfEpisode, sfEnv, sfEnvOut, bJson=False):
    if bJson:
        with open(sfEpisode, "rb") as fEp:
            oActions = json.load(fEp)
            oEp = {"actions": oActions}
            print("json oEp:", type(oEp), list(oEp.keys()))
    else:
        with open(sfEpisode, "rb") as fEp:
            oEp = msgpack.load(fEp)
            print("oEp:", type(oEp), list(oEp.keys()))

    with open(sfEnv, "rb") as fEnv:
        oEnv = msgpack.load(fEnv)
        print("oEnv:", type(oEnv), list(oEnv.keys()))

    # merge dicts
    oEnv2 = {**oEp, **oEnv}
    print("Merged keys:", list(oEnv2.keys()))

    with open(sfEnvOut, "wb") as fEnv:
        msgpack.dump(oEnv2, fEnv)


def printKeys1(sfEnv):
    with open(sfEnv, "rb") as fEnv:
        oEnv = msgpack.load(fEnv, encoding="utf-8")
        print(sfEnv, "keys:", list(oEnv.keys()))
        for sKey in oEnv.keys():
            print("key", sKey, len(oEnv[sKey]))
            if sKey == "shape":
                print("shape: ", oEnv[sKey])


def printKeys(sfEnvs):
    try:
        for sfEnv in sfEnvs:
            printKeys1(sfEnv)
    except:
        # assume single env
        printKeys1(sfEnvs)


def main2():
    parser = ap.ArgumentParser(description='Generate env, merge episodes into env files.')

    parser.add_argument("-c", '--createEnvs', type=int, nargs=2, action="append",
                        metavar=("nStart", "nEnd"),
                        help='merge episode into env')

    parser.add_argument("-d", "--outDir", type=str, nargs=1, default="./test-env-tmp")

    parser.add_argument("-m", '--merge', type=str, nargs=3, action="append",
                        metavar=("episode", "env", "output_env"),
                        help='merge episode into env')

    parser.add_argument("-j", '--mergejson', type=str, nargs=3, action="append",
                        metavar=("json", "env", "output_env"),
                        help='merge json actions into env, with key actions')

    parser.add_argument('-k', "--keys", type=str, action='append', nargs="+",
                        help='print the keys in a file')

    args = parser.parse_args()
    print(args)

    if args.merge:
        print("merge:", args.merge)
        merge(*args.merge[0])

    if args.mergejson:
        print("merge json:", args.mergejson)
        merge(*args.mergejson[0], bJson=True)

    if args.keys:
        print("keys:", args.keys)
        printKeys(args.keys[0])

    if args.outDir:
        print("outDir", args.outDir)

    if args.createEnvs:
        print("create Envs - ", *args.createEnvs[0])
        createEnvSet(*args.createEnvs[0], sDir=args.outDir)


if __name__ == "__main__":
    main2()
