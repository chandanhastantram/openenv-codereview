from openenv.core import OpenEnv

env = OpenEnv("codereview-env", url="http://localhost:7860")
env.reset("find-obvious-bug")
res = env.step({"action_type": "approve", "message": "LGTM"})
print(res)
