RANDOM_STEPS=1e4
MAX_STEPS=1e6
MAX_ENV_STEPS=500
WARMUP=10000

REPLAY_BUFFER_SIZE = 30000
#REPLAY_BUFFER_SIZE = 6
BATCH_SIZE = 100
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
TAU = 5e-3
DISCOUNT=0.99
EXPL_NOISE=0.2


EVAL_EPISODE=500
EVAL_STEPS=1000
EVAL_LENGTH=10
