import argparse
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args(["--env_a", "1", "--env_b", "2"])

# unknown == ['--env_a', '5', '--env_b', '8']

env = {}
for i in range(0, len(unknown), 2):
    assert unknown[i+1].isdigit()
    env[unknown[i][len('--env_'):]] = int(unknown[i+1])

# env == {'a': 5, 'b': 8}
print(env)